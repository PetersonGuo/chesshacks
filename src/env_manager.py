from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

_FALSE_VALUES = {"0", "false", "off", "no", ""}


def _env_int(name: str, default: int) -> int:
  try:
    value = int(os.getenv(name, default))
    return value if value > 0 else default
  except (TypeError, ValueError):
    return default


def _env_bool(name: str, default: bool) -> bool:
  raw = os.getenv(name)
  if raw is None:
    return default
  return raw.strip().lower() not in _FALSE_VALUES


@dataclass(frozen=True)
class EnvConfig:
  search_depth: int
  serve_port: int
  hf_token: Optional[str]
  cuda_enabled: bool


@lru_cache(maxsize=1)
def get_env_config() -> EnvConfig:
  return EnvConfig(
      search_depth=_env_int("CHESSHACKS_MAX_DEPTH", 4),
      serve_port=_env_int("SERVE_PORT", 5058),
      hf_token=os.getenv("HF_TOKEN"),
      cuda_enabled=_env_bool("CHESSHACKS_ENABLE_CUDA", False),
  )

