import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BUILD_DIR = ROOT / "build"


def _ensure_c_helpers() -> None:
  """Make sure the native extension exists before tests import it."""
  if any(BUILD_DIR.glob("c_helpers*.so")):
    return
  subprocess.run([str(ROOT / "build.sh")], cwd=ROOT, check=True)


try:
  _ensure_c_helpers()
except Exception as exc:  # pragma: no cover - surface original failure
  raise RuntimeError("Failed to build c_helpers before running tests") from exc


sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(BUILD_DIR))

