"""
Export a trained NNUE checkpoint to TorchScript for LibTorch inference.

This script loads the standard training checkpoint (state dict) and produces a
TorchScript module that can be consumed directly by the C++ evaluator.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train.nnue_model.model import ChessNNUEModel


def _load_state_dict(pt_path: Path):
    checkpoint = torch.load(pt_path, map_location="cpu")
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("model_state_dict", checkpoint)
    else:
        state_dict = checkpoint.state_dict()
    return {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}


def export_torchscript(source: Path, target: Path) -> Path:
    state_dict = _load_state_dict(source)
    model = ChessNNUEModel()
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    scripted = torch.jit.script(model)
    target.parent.mkdir(parents=True, exist_ok=True)
    scripted.save(str(target))
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description="Export NNUE checkpoint to TorchScript")
    parser.add_argument(
        "checkpoint",
        type=Path,
        help="Path to the training checkpoint (e.g., best_model.pt)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output TorchScript file (defaults to checkpoint.torchscript.pt)",
    )
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve()
    if args.output:
        target = args.output.resolve()
    else:
        target = checkpoint.with_suffix(".torchscript.pt")

    exported = export_torchscript(checkpoint, target)
    print(f"Exported TorchScript NNUE model to {exported}")


if __name__ == "__main__":
    main()

