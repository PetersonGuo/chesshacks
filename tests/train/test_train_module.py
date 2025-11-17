import json
import os
from datetime import datetime

import pytest
import torch
import torch.nn as nn

from train.config import TrainingConfig
from train.train import (
    WarmupScheduler,
    ensure_data_exists,
    get_optimizer,
    get_scheduler,
    load_checkpoint,
    save_checkpoint,
    train_epoch,
    validate,
)


def make_config(tmp_path, **overrides):
    """Helper to create a CPU-only training config rooted in tmp_path."""
    params = {
        "train_data_path": str(tmp_path / "train.jsonl"),
        "val_data_path": str(tmp_path / "val.jsonl"),
        "checkpoint_dir": str(tmp_path / "checkpoints"),
        "device": "cpu",
        "auto_download": False,
        "verbose": False,
    }
    params.update(overrides)
    return TrainingConfig(**params)


class DummySparseModel(nn.Module):
    """Minimal model that mimics the sparse forward signature."""

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))

    def forward(
        self,
        white_indices,
        white_values,
        black_indices,
        black_values,
        batch_size=None,
    ):
        if batch_size is None:
            raise ValueError("batch_size must be provided for DummySparseModel")

        device = self.scale.device
        dtype = self.scale.dtype

        def scatter(indices, values):
            if indices.numel() == 0:
                return torch.zeros(batch_size, dtype=dtype, device=device)
            idx = indices.to(device)
            vals = values.to(device, dtype=dtype)
            acc = torch.zeros(batch_size, dtype=dtype, device=device)
            acc.index_add_(0, idx[0], vals)
            return acc

        white_sum = scatter(white_indices, white_values)
        black_sum = scatter(black_indices, black_values)
        total = white_sum + black_sum
        return total.unsqueeze(1) * self.scale


def test_warmup_scheduler_progression_and_state():
    model = nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    base_sched = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    scheduler = WarmupScheduler(
        optimizer=optimizer,
        base_scheduler=base_sched,
        warmup_epochs=2,
        target_lr=0.1,
        warmup_start_lr=0.0,
    )

    lrs = []
    for _ in range(4):
        optimizer.step()
        scheduler.step()
        lrs.append(optimizer.param_groups[0]["lr"])

    assert lrs[0] == pytest.approx(0.05)
    assert lrs[1] == pytest.approx(0.1)
    assert lrs[2] == pytest.approx(0.05)
    assert lrs[3] == pytest.approx(0.025)

    state = scheduler.state_dict()
    new_model = nn.Linear(1, 1)
    new_opt = torch.optim.SGD(new_model.parameters(), lr=0.1)
    new_base = torch.optim.lr_scheduler.StepLR(new_opt, step_size=1, gamma=0.5)
    restored = WarmupScheduler(
        optimizer=new_opt,
        base_scheduler=new_base,
        warmup_epochs=2,
        target_lr=0.1,
        warmup_start_lr=0.0,
    )
    restored.load_state_dict(state)
    assert restored.current_epoch == scheduler.current_epoch
    assert restored.base_scheduler.last_epoch == scheduler.base_scheduler.last_epoch


def test_get_optimizer_and_scheduler_variants(tmp_path):
    config = make_config(tmp_path, optimizer="adam")
    model = nn.Linear(2, 2)

    opt = get_optimizer(model, config)
    assert isinstance(opt, torch.optim.Adam)

    config.optimizer = "sgd"
    opt = get_optimizer(model, config)
    assert isinstance(opt, torch.optim.SGD)

    config.optimizer = "unknown"
    with pytest.raises(ValueError):
        get_optimizer(model, config)

    # Scheduler with warmup should wrap cosine annealing
    config.optimizer = "adam"
    config.scheduler = "cosine"
    config.warmup_epochs = 2
    config.learning_rate = 0.01
    sched = get_scheduler(get_optimizer(model, config), config)
    assert isinstance(sched, WarmupScheduler)
    assert isinstance(sched.base_scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)

    # Without warmup we should get the raw StepLR
    config.scheduler = "step"
    config.warmup_epochs = 0
    sched = get_scheduler(get_optimizer(model, config), config)
    assert isinstance(sched, torch.optim.lr_scheduler.StepLR)


def test_ensure_data_exists_handles_existing_files(tmp_path):
    train_path = tmp_path / "train.jsonl"
    val_path = tmp_path / "val.jsonl"
    for file_path in (train_path, val_path):
        with open(file_path, "w", encoding="utf-8") as fh:
            fh.write(json.dumps({"fen": "startpos", "eval": 0.0}) + "\n")

    config = make_config(
        tmp_path,
        train_data_path=str(train_path),
        val_data_path=str(val_path),
    )

    assert ensure_data_exists(config) is True


def test_ensure_data_exists_missing_without_download(tmp_path):
    config = make_config(
        tmp_path,
        train_data_path=str(tmp_path / "missing.jsonl"),
        val_data_path="",
    )
    assert ensure_data_exists(config) is False


def make_sparse_batch():
    white_indices = torch.tensor([[0, 0], [0, 1]], dtype=torch.long)
    white_values = torch.ones(2, dtype=torch.float32)
    black_indices = torch.tensor([[0], [2]], dtype=torch.long)
    black_values = torch.ones(1, dtype=torch.float32)
    targets = torch.tensor([[3.0]], dtype=torch.float32)
    return white_indices, white_values, black_indices, black_values, targets


def test_train_epoch_updates_parameters(tmp_path):
    config = make_config(tmp_path)
    model = DummySparseModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    criterion = nn.MSELoss()
    batch = make_sparse_batch()
    train_loader = [batch]

    initial_scale = model.scale.detach().clone()
    loss = train_epoch(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=torch.device("cpu"),
        config=config,
        epoch=0,
    )

    assert isinstance(loss, float)
    assert loss > 0
    assert not torch.allclose(model.scale.detach(), initial_scale)


def test_validate_matches_manual_loss(tmp_path):
    config = make_config(tmp_path)
    model = DummySparseModel()
    criterion = nn.MSELoss()
    batch = make_sparse_batch()
    val_loader = [batch]

    model.eval()
    with torch.no_grad():
        expected = criterion(
            model(
                batch[0],
                batch[1],
                batch[2],
                batch[3],
                batch_size=batch[4].shape[0],
            ),
            batch[4],
        ).item()

    val_loss = validate(
        model=model,
        val_loader=val_loader,
        criterion=criterion,
        device=torch.device("cpu"),
        config=config,
    )

    assert isinstance(val_loss, float)
    assert val_loss == pytest.approx(expected)


def test_save_and_load_checkpoint_roundtrip(tmp_path):
    config = make_config(tmp_path)
    model = DummySparseModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    filename = "unit_test_checkpoint.pt"
    train_loss = 1.23
    val_loss = 0.5
    best_val_loss = 0.42
    start_time = datetime.now().isoformat()

    save_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=0,
        train_loss=train_loss,
        val_loss=val_loss,
        config=config,
        filename=filename,
        upload_to_hf=False,
        scheduler=None,
        best_val_loss=best_val_loss,
        training_start_time=start_time,
    )

    checkpoint_path = os.path.join(config.checkpoint_dir, filename)
    assert os.path.exists(checkpoint_path)

    # Mutate parameters to ensure load_checkpoint restores them
    with torch.no_grad():
        model.scale.fill_(0.0)

    start_epoch, restored_best, restored_start = load_checkpoint(
        model=model,
        optimizer=optimizer,
        checkpoint_path=checkpoint_path,
        map_location=torch.device("cpu"),
    )

    assert start_epoch == 1
    assert restored_best == pytest.approx(best_val_loss)
    assert restored_start == start_time
    assert model.scale.item() != 0.0

