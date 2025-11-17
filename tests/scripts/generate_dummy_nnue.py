#!/usr/bin/env python3
"""
Generate a tiny NNUE-compatible model checkpoint using Python + PyTorch.

The resulting file is saved with torch.save(model, path) so that the C++
NNUEEvaluator can ingest it directly via torch::load. A matching
nnue_stats.json is written alongside the checkpoint.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn


class TorchNNUE(nn.Module):
    def __init__(self, hidden: int, hidden2: int, hidden3: int):
        super().__init__()
        self.ft_friendly = nn.Linear(384, hidden)
        self.ft_enemy = nn.Linear(384, hidden)
        self.fc1 = nn.Linear(hidden, hidden2)
        self.res1 = nn.Linear(hidden2, hidden2)
        self.res2 = nn.Linear(hidden2, hidden2)
        self.fc2 = nn.Linear(hidden2, hidden3)
        self.fc3 = nn.Linear(hidden3, 1)

    @staticmethod
    def clipped_relu(tensor):
        return torch.clamp(tensor, 0.0, 1.0)

    def forward(self, features):
        friendly = features[:, :384]
        enemy = features[:, 384:]
        friendly = self.clipped_relu(self.ft_friendly(friendly))
        enemy = self.clipped_relu(self.ft_enemy(enemy))
        x = self.clipped_relu(self.fc1(friendly + enemy))
        residual = x
        x = self.clipped_relu(self.res1(x))
        x = self.res2(x)
        x = self.clipped_relu(x + residual)
        x = self.clipped_relu(self.fc2(x))
        return self.fc3(x)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Emit a minimal NNUE checkpoint for integration tests."
    )
    parser.add_argument(
        "output",
        type=Path,
        help="Path to the .pt file that should be written.",
    )
    parser.add_argument(
        "--hidden",
        type=int,
        default=512,
        help="Hidden layer size for the feature network (default: 512).",
    )
    parser.add_argument(
        "--hidden2",
        type=int,
        default=64,
        help="Hidden layer size for the first fully connected block (default: 64).",
    )
    parser.add_argument(
        "--hidden3",
        type=int,
        default=64,
        help="Hidden layer size for the penultimate layer (default: 64).",
    )
    return parser.parse_args()


def _build_model(hidden: int, hidden2: int, hidden3: int):
    model = TorchNNUE(hidden, hidden2, hidden3)
    model.eval()
    return model


def main() -> int:
    args = _parse_args()

    model = _build_model(args.hidden, args.hidden2, args.hidden3)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), args.output)

    stats_path = args.output.parent / "nnue_stats.json"
    stats_path.write_text(json.dumps({"eval_mean": 0.0, "eval_std": 1.0}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
