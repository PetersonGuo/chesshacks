#!/bin/bash
# Automated build script for ChessHacks C++ extensions
# Uses all available CPU cores and handles Python version detection

set -euo pipefail
export CHESSHACKS_MAX_DEPTH=6
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"
USER_LOCAL_BIN="$HOME/.local/bin"
if [[ -d "$USER_LOCAL_BIN" ]] && [[ ":$PATH:" != *":$USER_LOCAL_BIN:"* ]]; then
    export PATH="$USER_LOCAL_BIN:$PATH"
fi

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

ENABLE_CUDA="${CHESSHACKS_ENABLE_CUDA:-0}"
if [[ "$ENABLE_CUDA" != "0" && "$ENABLE_CUDA" != "OFF" ]]; then
    echo "CUDA build requested via CHESSHACKS_ENABLE_CUDA"
    CMAKE_CUDA_FLAG="-DCHESSHACKS_ENABLE_CUDA=ON"
else
    CMAKE_CUDA_FLAG="-DCHESSHACKS_ENABLE_CUDA=OFF"
fi

if [[ -z "${PYTHON_BIN:-}" ]]; then
    PYTHON_BIN=$(command -v python3)
fi

UV_BIN=$(command -v uv || true)
if [[ -n "$UV_BIN" ]]; then
    echo "Using uv for pip commands: $UV_BIN pip"
    PIP_RUN=("$UV_BIN" pip)
else
    echo "uv command not found; falling back to python -m pip"
    PIP_RUN=("$PYTHON_BIN" -m pip)
fi

pip_install() {
    "${PIP_RUN[@]}" "$@"
}

ensure_tool_via_pip() {
    local cmd="$1"
    local pkg="$2"
    if command -v "$cmd" >/dev/null 2>&1; then
        return
    fi
    echo "Installing $cmd via pip (--user) (package: $pkg)..."
    if pip_install --user "$pkg"; then
        hash -r
        if ! command -v "$cmd" >/dev/null 2>&1; then
            echo "ERROR: $cmd still not available after installing $pkg via pip" >&2
            exit 1
        fi
    else
        echo "ERROR: Failed to install $pkg via pip" >&2
        exit 1
    fi
}

ensure_clang_toolchain() {
    if command -v clang >/dev/null 2>&1 && command -v clang++ >/dev/null 2>&1; then
        return
    fi
    for pkg in clang-llvm clang; do
        echo "Attempting to install clang toolchain via pip package '$pkg'..."
        if pip_install --user "$pkg"; then
            hash -r
            if command -v clang >/dev/null 2>&1 && command -v clang++ >/dev/null 2>&1; then
                return
            fi
        fi
    done
    echo "ERROR: Unable to install clang toolchain via pip (--user)." >&2
    exit 1
}

# Run all tool checks in parallel for faster setup
ensure_tool_via_pip "cmake" "cmake" &
ensure_tool_via_pip "ninja" "ninja" &
ensure_clang_toolchain &
wait  # Wait for all background jobs to complete; fails if any job fails

if [[ ! -d "$SCRIPT_DIR/third_party/libtorch" ]]; then
    curl -LO https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-2.9.1%2Bcpu.zip
    npm install --silent --prefix "$HOME/.local/extractzip" extract-zip
    node "$HOME/.local/extractzip/node_modules/extract-zip/cli.js" libtorch-shared-with-deps-2.9.1%2Bcpu.zip `pwd`/third_party/
fi

TORCH_DIR="$SCRIPT_DIR/third_party/libtorch"
TORCH_CMAKE_DIR="$TORCH_DIR/share/cmake/Torch"
if [[ ! -d "$TORCH_CMAKE_DIR" ]]; then
    echo "ERROR: Vendored libtorch directory not found at $TORCH_CMAKE_DIR" >&2
    echo "Please place the libtorch CPU distribution under third_party/libtorch." >&2
    exit 1
fi
echo "Using vendored libtorch: $TORCH_DIR"
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-"8.0;8.6"}
export LD_LIBRARY_PATH="$TORCH_DIR/lib:${LD_LIBRARY_PATH:-}"

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

if command -v clang >/dev/null 2>&1 && command -v clang++ >/dev/null 2>&1; then
    C_COMPILER="clang"
    CXX_COMPILER="clang++"
    COMPILER_DESC="Clang"
else
    C_COMPILER="gcc"
    CXX_COMPILER="g++"
    COMPILER_DESC="GCC"
fi
echo "Selected C compiler: $C_COMPILER ($COMPILER_DESC preference)"
echo "Selected C++ compiler: $CXX_COMPILER"

if command -v ninja >/dev/null 2>&1; then
    CMAKE_GENERATOR="Ninja"
    echo "Build generator: Ninja"
else
    CMAKE_GENERATOR="Unix Makefiles"
    echo "Build generator: Unix Makefiles (make fallback)"
fi
echo ""

echo "Installing Python dependencies from requirements.txt..."
# Check if we're in a virtual environment
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    # Not in venv, try to install with --user or skip if it fails
    if ! pip_install --user -r "$SCRIPT_DIR/requirements.txt" 2>/dev/null; then
        echo "⚠ Warning: Failed to install requirements (may already be installed or need venv)"
        echo "Continuing with build..."
    else
        echo "✓ Requirements installed"
    fi
else
    # In venv, normal install
    if ! pip_install -r "$SCRIPT_DIR/requirements.txt"; then
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

if [[ -f "$BUILD_DIR/CMakeCache.txt" ]]; then
    CURRENT_GENERATOR=$(grep -m1 "^CMAKE_GENERATOR:" "$BUILD_DIR/CMakeCache.txt" | cut -d= -f2)
    if [[ "$CURRENT_GENERATOR" != "$CMAKE_GENERATOR" ]]; then
        echo "Generator changed ($CURRENT_GENERATOR -> $CMAKE_GENERATOR); clearing build directory..."
        rm -rf "$BUILD_DIR"
    fi
fi

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

echo "Configuring CMake..."
cmake -S "$SCRIPT_DIR" -B . \
      -G "$CMAKE_GENERATOR" \
      -DCMAKE_C_COMPILER="$C_COMPILER" \
      -DCMAKE_CXX_COMPILER="$CXX_COMPILER" \
      -DCMAKE_BUILD_TYPE=Release \
      "$CMAKE_CUDA_FLAG" \
      -DTorch_DIR="$TORCH_CMAKE_DIR" \
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
