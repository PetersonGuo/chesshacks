"""
Lightweight sanity checks for NNUE integration.
Ensures the TorchScript checkpoint loads and evaluation matches NNUE path.
"""

import sys
from pathlib import Path

import chess
import pytest

import c_helpers

torch = pytest.importorskip("torch", reason="PyTorch is required for NNUE integration test")


@pytest.fixture(scope="session")
def model_bin_path(tmp_path_factory):
    checkpoint_dir = Path("train/nnue_model/checkpoints")
    pt_path = checkpoint_dir / "best_model.pt"
    if not pt_path.exists():
        pt_path = checkpoint_dir / "final_model.pt"

    if not pt_path.exists():
        pytest.skip(
            "Skipping NNUE tests: no checkpoint found in train/nnue_model/checkpoints",
            allow_module_level=True,
        )

    sys.path.insert(0, str(Path("train").resolve()))
    from train.nnue_model.model import ChessNNUEModel

    ckpt = torch.load(pt_path, map_location="cpu")
    state_dict = ckpt.get("model_state_dict", ckpt)
    cleaned = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

    model = ChessNNUEModel()
    model.load_state_dict(cleaned, strict=False)
    model.eval()

    scripted = torch.jit.script(model)
    output = tmp_path_factory.mktemp("nnue") / "exported_model.pt"
    scripted.save(str(output))
    return output


def test_nnue_evaluation(model_bin_path):
    assert c_helpers.init_nnue(str(model_bin_path)), "Failed to initialize NNUE"

    positions = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
        "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
    ]

    for fen in positions:
        pst_eval = c_helpers.evaluate_with_pst(fen)
        nnue_eval = c_helpers.evaluate_nnue(fen)
        assert isinstance(nnue_eval, int)
        # ensure NNUE path hooked into evaluate()
        assert c_helpers.evaluate(fen) == nnue_eval
        # evaluations should be in the same ballpark
        assert abs(nnue_eval - pst_eval) < 2000
