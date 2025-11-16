#!/bin/bash
# Automated build script for ChessHacks C++ extensions
# Uses all available CPU cores and handles Python version detection

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"

# Detect number of CPU cores
if command -v nproc &>/dev/null; then
    NUM_CORES=$(nproc)
elif command -v sysctl &>/dev/null; then
    NUM_CORES=$(sysctl -n hw.ncpu)
else
    NUM_CORES=4
fi

echo "=================================================="
echo "ChessHacks Automated Build Script"
echo "=================================================="
echo "Using $NUM_CORES CPU cores for compilation"
echo ""

cd "$SCRIPT_DIR"

PYTHON_BIN=$(command -v python3)
PYTHON_ROOT=$(python3 -c "import sys, pathlib; print(pathlib.Path(sys.executable).parent.parent)")
PYTHON_INCLUDE=$(python3 -c "import sysconfig; print(sysconfig.get_path('include'))")

# Find Python library - handle both .so (Linux) and .dylib (macOS)
PYTHON_LIBRARY=$(python3 -c "
import sysconfig
import pathlib
import os

libdir = pathlib.Path(sysconfig.get_config_var('LIBDIR'))
ldversion = sysconfig.get_config_var('LDVERSION')

# Try .dylib first (macOS), then .so (Linux)
for ext in ['.dylib', '.so']:
    libname = f'libpython{ldversion}{ext}'
    libpath = libdir / libname
    if libpath.exists():
        print(libpath)
        break
else:
    # Fallback: try to find any libpython in the directory
    if libdir.exists():
        for f in os.listdir(libdir):
            if f.startswith('libpython') and (f.endswith('.dylib') or f.endswith('.so')):
                print(libdir / f)
                break
")
echo "Using Python interpreter: $PYTHON_BIN"
echo ""

echo "Installing Python dependencies from requirements.txt..."
# Check if we need --break-system-packages flag (Python 3.11+ with PEP 668)
PIP_FLAGS=""
if python3 -c "import sys; exit(0 if sys.version_info >= (3, 11) else 1)" 2>/dev/null; then
    # Try to install without flag first, but use flag if needed
    if ! python3 -m pip install -r "$SCRIPT_DIR/requirements.txt" 2>&1 | grep -q "externally-managed-environment"; then
        PIP_FLAGS=""
    else
        PIP_FLAGS="--break-system-packages"
        echo "Note: Using --break-system-packages flag for Python 3.11+"
    fi
fi

# Try installation (skip if packages already installed)
if ! python3 -m pip install $PIP_FLAGS -r "$SCRIPT_DIR/requirements.txt" 2>&1 | grep -q "Requirement already satisfied"; then
    if ! python3 -m pip install $PIP_FLAGS -r "$SCRIPT_DIR/requirements.txt"; then
        echo "WARNING: Some requirements may have failed to install, but continuing..."
        echo "If build fails, run: pip install -r requirements.txt --break-system-packages"
    fi
fi
echo "✓ Requirements check complete"
echo ""

if [[ "${1:-}" == "clean" ]]; then
    echo "Cleaning old build files..."
    rm -rf "$BUILD_DIR"
    echo "Clean complete."
    echo ""
fi

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

echo "Configuring CMake..."
cmake -S "$SCRIPT_DIR" -B . \
      -DCMAKE_BUILD_TYPE=Release \
      -DPython3_EXECUTABLE="$PYTHON_BIN" \
      -DPython3_ROOT_DIR="$PYTHON_ROOT" \
      -DPython3_INCLUDE_DIR="$PYTHON_INCLUDE" \
      -DPython3_LIBRARY="$PYTHON_LIBRARY" \
      -DPython_EXECUTABLE="$PYTHON_BIN" \
      -DPython_ROOT_DIR="$PYTHON_ROOT" \
      -DPython_INCLUDE_DIR="$PYTHON_INCLUDE" \
      -DPython_LIBRARY="$PYTHON_LIBRARY" \
      -DPython_FIND_VIRTUALENV=FIRST \
      -DPython_FIND_STRATEGY=LOCATION
echo ""

echo "Building C++ extensions..."
cmake --build . -j"$NUM_CORES"
echo ""

SO_FILE=$(find "$BUILD_DIR" -maxdepth 1 -name "c_helpers*.so" -type f | head -1)
if [[ -z "$SO_FILE" ]]; then
    echo "ERROR: Build succeeded but c_helpers*.so not found!"
    exit 1
fi

echo "Build successful: $SO_FILE"
echo ""
echo "Verifying module can be imported..."
if python3 -c "import sys; sys.path.insert(0, '$BUILD_DIR'); import c_helpers; print('✓ Module imported successfully')" 2>/dev/null; then
    echo "✓ Build verification passed"
else
    echo "⚠ Warning: Module built but import verification failed."
fi

echo ""
echo "=================================================="
echo "Build Complete!"
echo "=================================================="
echo "Module location: $SO_FILE"
echo ""

