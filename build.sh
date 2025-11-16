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

echo "Checking Python dependencies..."
if ! python3 -c "import nanobind" &>/dev/null; then
    echo "nanobind not found - installing dependencies..."
    if ! pip install -q nanobind; then
        echo "ERROR: Failed to install nanobind"
        echo "Please run: pip install nanobind"
        exit 1
    fi
    echo "✓ nanobind installed"
fi
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
cmake -S "$SCRIPT_DIR" -B . -DCMAKE_BUILD_TYPE=Release
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
if python3 -c "import sys; sys.path.insert(0, '$BUILD_DIR'); import c_helpers; print('✓ Module imported successfully')" &>/dev/null; then
    echo "✓ Build verification passed"
else
    echo "⚠ Warning: Module built but import failed. Ensure PYTHONPATH includes '$BUILD_DIR'."
fi

echo ""
echo "=================================================="
echo "Build Complete!"
echo "=================================================="
echo "Module location: $SO_FILE"
echo ""
echo "To use the module:"
echo "  import sys"
echo "  sys.path.insert(0, '$BUILD_DIR')"
echo "  import c_helpers"
echo ""

