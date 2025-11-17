"""
NNUE integration tests rewritten for the native torch::nn loading path.

The C++ evaluator now expects checkpoints saved with torch.save(model.state_dict()).
This test generates such a file on the fly when a training checkpoint is available.
"""

from __future__ import annotations

from pathlib import Path

import c_helpers
import pytest


@pytest.fixture(scope="session")
def nnue_state_dict(tmp_path_factory):
    bundled_script = Path("tests/data/nnue_minimal_torchscript.pt")
    if bundled_script.exists():
        return bundled_script.resolve()

    checkpoint_dir = Path("train/nnue_model/checkpoints")
    candidate_files = ["best_model.pt", "final_model.pt"]
    pt_path = next(
        (
            checkpoint_dir / name
            for name in candidate_files
            if (checkpoint_dir / name).exists()
        ),
        None,
    )

    try:
        import torch
    except ImportError as exc:
        pytest.skip(
            f"Bundled NNUE state missing and PyTorch unavailable for fallback ({exc})."
        )

    from train.nnue_model.model import ChessNNUEModel  # type: ignore

    model = ChessNNUEModel()
    if pt_path:
        checkpoint = torch.load(pt_path, map_location="cpu")
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        cleaned = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(cleaned, strict=False)
    model.eval()

    export_dir = tmp_path_factory.mktemp("nnue_torchscript")
    export_path = export_dir / "nnue_model.pt"
    scripted = torch.jit.script(model)
    scripted.save(export_path)
    return export_path


def test_nnue_evaluation_pipeline(nnue_state_dict, make_state):
    assert c_helpers.init_nnue(
        str(nnue_state_dict)
    ), "Failed to initialize NNUE evaluator"

    fens = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
        "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
    ]

    for fen in fens:
        state = make_state(fen)
        nnue_eval = c_helpers.evaluate_nnue(state)
        auto_eval = c_helpers.evaluate(state)
        assert isinstance(nnue_eval, int)
        assert nnue_eval == auto_eval  # evaluate() should route to NNUE when loaded

        material_eval = c_helpers.evaluate_material(state)
        assert abs(nnue_eval - material_eval) < 4000  # sanity bound
