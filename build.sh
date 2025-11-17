#!/bin/bash
# ChessHacks environment bootstrapper.
# - Verifies the bundled native module matches the active Python ABI.
# - Downloads libtorch CPU/GPU if missing.
# - Installs Python requirements.
# - Writes an env helper file so `serve.py` can run immediately.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"
THIRD_PARTY_DIR="$SCRIPT_DIR/third_party"
LIBTORCH_DIR="$BUILD_DIR/third_party/libtorch"
ENV_FILE="$SCRIPT_DIR/.chesshacks_env"

PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 || true)}"
if [[ -z "$PYTHON_BIN" ]]; then
  echo "ERROR: python3 not found. Set PYTHON_BIN to your interpreter path." >&2
  exit 1
fi

if [[ ! -d "$SCRIPT_DIR/third_party/surge/src" ]]; then
  if command -v git >/dev/null 2>&1; then
    echo "Ensuring surge submodule is initialized..."
    git -C "$SCRIPT_DIR" submodule update --init --recursive third_party/surge
  else
    echo "ERROR: git is required to fetch the surge submodule." >&2
    exit 1
  fi
fi

echo "==============================================="
echo "ChessHacks environment bootstrap"
echo "Python: $PYTHON_BIN"
echo "Root  : $SCRIPT_DIR"
echo "==============================================="

mkdir -p "$BUILD_DIR"

echo "libtorch will be downloaded automatically by CMake via FetchContent."

echo "Installing Python requirements via pip..."
"$PYTHON_BIN" -m pip install -r "$SCRIPT_DIR/requirements.txt"

cat > "$ENV_FILE" <<'EOF'
# Source this file before running serve.py
export CHESSHACKS_ROOT="$SCRIPT_DIR"
export PYTHON_BIN="$PYTHON_BIN"
export PYTHONPATH="$SCRIPT_DIR:$SCRIPT_DIR/src:$BUILD_DIR:\${PYTHONPATH:-}"
export TORCH_HOME="$BUILD_DIR/third_party"

CHESSHACKS_LIBTORCH_LIB="$LIBTORCH_DIR/lib"
if [[ -n "\${LD_LIBRARY_PATH:-}" ]]; then
  _ch_clean_ld="\$("\$PYTHON_BIN" - <<'PY'
import os
from pathlib import Path
needle = Path(os.environ.get("CHESSHACKS_LIBTORCH_LIB", "")).resolve()
paths = []
for entry in os.environ.get("LD_LIBRARY_PATH", "").split(":"):
    chunks = [part for part in entry.strip().split() if part]
    for chunk in chunks:
        if Path(chunk).resolve() == needle:
            continue
        paths.append(chunk)
print(":".join(paths))
PY
)"
  if [[ -n "\$_ch_clean_ld" ]]; then
    export LD_LIBRARY_PATH="\$_ch_clean_ld"
  else
    unset LD_LIBRARY_PATH
  fi
  unset _ch_clean_ld
else
  unset LD_LIBRARY_PATH
fi
unset CHESSHACKS_LIBTORCH_LIB
EOF

echo ""
echo " Environment file written to $ENV_FILE"
echo "Next steps:"
echo "  source $ENV_FILE"
echo "  python serve.py"
echo "==============================================="

