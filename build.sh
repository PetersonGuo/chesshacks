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

if [[ -z "${PYTHON_BIN:-}" ]]; then
    PYTHON_BIN=$(command -v python3)
fi

PYTHON_ROOT=$("$PYTHON_BIN" -c "import sys, pathlib; print(pathlib.Path(sys.executable).parent.parent)")
PYTHON_INCLUDE=$("$PYTHON_BIN" -c "import sysconfig; print(sysconfig.get_path('include'))")

# Detect Python library - handle both .so (Linux) and .dylib (macOS) extensions
# Also handle framework-based installations on macOS
PYTHON_LIBRARY=$("$PYTHON_BIN" -c "
import sysconfig, pathlib, platform

libdir = pathlib.Path(sysconfig.get_config_var('LIBDIR'))
ldversion = sysconfig.get_config_var('LDVERSION')
ldlibrary = sysconfig.get_config_var('LDLIBRARY')

# On macOS, check for .dylib first, then framework
if platform.system() == 'Darwin':
    dylib = libdir / f'libpython{ldversion}.dylib'
    if dylib.exists():
        print(dylib)
    else:
        # Try framework path
        framework_path = pathlib.Path(sysconfig.get_config_var('LIBPL')).parent.parent / ldlibrary
        if framework_path.exists():
            print(framework_path)
        else:
            # Fallback to static library
            static_lib = libdir / f'libpython{ldversion}.a'
            if static_lib.exists():
                print(static_lib)
            else:
                # Last resort: construct expected path
                print(libdir / f'libpython{ldversion}.dylib')
else:
    # Linux: look for .so
    so_lib = libdir / f'libpython{ldversion}.so'
    if so_lib.exists():
        print(so_lib)
    else:
        # Fallback
        print(libdir / f'libpython{ldversion}.so')
")

echo "Using Python interpreter: $PYTHON_BIN"
echo "Python library: $PYTHON_LIBRARY"
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
if "$PYTHON_BIN" -c "import sys; sys.path.insert(0, '$BUILD_DIR'); import c_helpers; print('✓ Module imported successfully')" 2>/dev/null; then
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
