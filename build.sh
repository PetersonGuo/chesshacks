#!/bin/bash
# Automated build script for ChessHacks C++ extensions
# Uses pure Python/setuptools - NO CMake required!
# Uses all available CPU cores and handles Python version detection

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"

# Detect number of CPU cores
if command -v nproc &>/dev/null; then
    NUM_CORES=$(nproc)
elif command -v sysctl &>/dev/null; then
    if NUM_CORES=$(sysctl -n hw.ncpu 2>/dev/null); then
        :
    else
        NUM_CORES=4
    fi
else
    NUM_CORES=4
fi

echo "=================================================="
echo "ChessHacks Automated Build Script"
echo "=================================================="
echo "Using $NUM_CORES CPU cores for compilation"
echo ""

cd "$SCRIPT_DIR"

# Detect Python interpreter (setup.py will handle the rest automatically)
if [[ -z "${PYTHON_BIN:-}" ]]; then
    PYTHON_BIN=$(command -v python3)
    if [[ -z "$PYTHON_BIN" ]]; then
        echo "ERROR: python3 not found in PATH"
        exit 1
    fi
fi

echo "Using Python interpreter: $PYTHON_BIN"
echo "Build system: Python setuptools (100% CMake-free!)"
echo ""

echo "Installing Python dependencies from requirements.txt..."
# Check if we're in a virtual environment
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    # Not in venv, try to install with --user or skip if it fails
    if ! "$PYTHON_BIN" -m pip install --user -r "$SCRIPT_DIR/requirements.txt" 2>/dev/null; then
        echo "⚠ Warning: Failed to install requirements (may already be installed or need venv)"
        echo "Continuing with build..."
    else
        echo "✓ Requirements installed"
    fi
else
    # In venv, normal install
    if ! "$PYTHON_BIN" -m pip install -r "$SCRIPT_DIR/requirements.txt"; then
        echo "⚠ Warning: Failed to install requirements (may already be installed)"
        echo "Continuing with build..."
    else
        echo "✓ Requirements installed"
    fi
fi
echo ""

if [[ "${1:-}" == "clean" ]]; then
    echo "Cleaning old build files..."
    rm -rf "$BUILD_DIR"
    # Also clean Python build artifacts
    rm -rf "$SCRIPT_DIR/build"
    rm -f "$SCRIPT_DIR"/*.so "$SCRIPT_DIR"/c_helpers*.so
    find "$SCRIPT_DIR" -name "*.so" -type f -delete 2>/dev/null || true
    echo "Clean complete."
    echo ""
fi

# Build using Python setuptools (NO CMake!)
echo "Building C++ extensions with Python setuptools..."
echo "(100% CMake-free build system)"
echo ""

cd "$SCRIPT_DIR"

# Build the extension using setup.py
# Use --inplace to build in the current directory
if "$PYTHON_BIN" setup.py build_ext --inplace 2>&1; then
    echo ""
    echo "✓ Build completed successfully"
else
    echo ""
    echo "ERROR: Build failed!"
    exit 1
fi

# Find the built module (could be in current dir or build/lib.*)
SO_FILE=$(find "$SCRIPT_DIR" -maxdepth 2 -name "c_helpers*.so" -type f | head -1)
if [[ -z "$SO_FILE" ]]; then
    # Also check build directory structure
    SO_FILE=$(find "$SCRIPT_DIR" -name "c_helpers*.so" -type f | head -1)
fi

if [[ -z "$SO_FILE" ]]; then
    echo "ERROR: Build succeeded but c_helpers*.so not found!"
    echo "Searched in: $SCRIPT_DIR"
    exit 1
fi

echo ""
echo "Build successful: $SO_FILE"
echo ""
echo "Verifying module can be imported..."
# Add the directory containing the .so to Python path
SO_DIR=$(dirname "$SO_FILE")
if "$PYTHON_BIN" -c "import sys; sys.path.insert(0, '$SO_DIR'); import c_helpers; print('✓ Module imported successfully')" 2>/dev/null; then
    echo "✓ Build verification passed"
else
    echo "⚠ Warning: Module built but import verification failed."
    echo "   Module location: $SO_FILE"
fi

echo ""
echo "=================================================="
echo "Build Complete!"
echo "=================================================="
echo "Module location: $SO_FILE"
echo ""
