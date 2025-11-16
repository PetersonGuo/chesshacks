#!/bin/bash
# Setup script to install all dependencies needed for ChessHacks

set -e

echo "=================================================="
echo "ChessHacks Dependency Setup"
echo "=================================================="
echo ""

# Check if we're on macOS
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Detected macOS"
    
    # Check for Homebrew
    if ! command -v brew &> /dev/null; then
        echo "ERROR: Homebrew is not installed."
        echo "Please install Homebrew first: https://brew.sh"
        exit 1
    fi
    
    echo "✓ Homebrew found"
    
    # Install cmake if not present
    if ! command -v cmake &> /dev/null; then
        echo ""
        echo "Installing cmake..."
        brew install cmake
    else
        echo "✓ cmake is already installed ($(cmake --version | head -n 1))"
    fi
    
    # Check for Python development headers
    # On macOS with Homebrew Python, headers should be included
    if ! python3-config --includes &> /dev/null; then
        echo ""
        echo "WARNING: Python development headers may not be available."
        echo "Installing Python with development headers..."
        brew install python@3.11 || brew install python@3.12 || brew install python@3.13
    else
        echo "✓ Python development headers available"
    fi
    
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "Detected Linux"
    
    # Detect package manager
    if command -v apt-get &> /dev/null; then
        PKG_MANAGER="apt-get"
        INSTALL_CMD="sudo apt-get install -y"
    elif command -v yum &> /dev/null; then
        PKG_MANAGER="yum"
        INSTALL_CMD="sudo yum install -y"
    elif command -v dnf &> /dev/null; then
        PKG_MANAGER="dnf"
        INSTALL_CMD="sudo dnf install -y"
    else
        echo "ERROR: Could not detect package manager (apt-get, yum, or dnf)"
        exit 1
    fi
    
    echo "Using $PKG_MANAGER"
    
    # Install cmake if not present
    if ! command -v cmake &> /dev/null; then
        echo ""
        echo "Installing cmake..."
        $INSTALL_CMD cmake
    else
        echo "✓ cmake is already installed ($(cmake --version | head -n 1))"
    fi
    
    # Install Python development headers
    if ! python3-config --includes &> /dev/null; then
        echo ""
        echo "Installing Python development headers..."
        $INSTALL_CMD python3-dev || $INSTALL_CMD python3-devel
    else
        echo "✓ Python development headers available"
    fi
else
    echo "WARNING: Unsupported OS: $OSTYPE"
    echo "Please install cmake and Python development headers manually"
fi

echo ""
echo "=================================================="
echo "Installing Python dependencies..."
echo "=================================================="

# Upgrade pip first
python3 -m pip install --upgrade pip

# Check if we need --break-system-packages flag (Python 3.11+ with PEP 668)
PIP_FLAGS=""
if python3 -c "import sys; exit(0 if sys.version_info >= (3, 11) else 1)" 2>/dev/null; then
    # Try without flag first, add if needed
    if ! python3 -m pip install --dry-run nanobind &>/dev/null 2>&1; then
        PIP_FLAGS="--break-system-packages"
        echo "Note: Using --break-system-packages flag for Python 3.11+"
    fi
fi

# Install from requirements.txt if it exists
if [ -f "requirements.txt" ]; then
    echo "Installing packages from requirements.txt..."
    python3 -m pip install $PIP_FLAGS -r requirements.txt
else
    echo "WARNING: requirements.txt not found, installing core dependencies..."
    python3 -m pip install $PIP_FLAGS torch numpy tqdm requests zstandard huggingface_hub
    python3 -m pip install $PIP_FLAGS fastapi uvicorn python-chess nanobind
fi

echo ""
echo "=================================================="
echo "Verifying installations..."
echo "=================================================="

# Check cmake
if command -v cmake &> /dev/null; then
    echo "✓ cmake: $(cmake --version | head -n 1)"
else
    echo "✗ cmake: NOT FOUND"
fi

# Check Python packages
echo ""
echo "Python packages:"
python3 -c "import torch; print(f'  ✓ torch: {torch.__version__}')" 2>/dev/null || echo "  ✗ torch: NOT FOUND"
python3 -c "import chess; print(f'  ✓ chess: {chess.__version__}')" 2>/dev/null || echo "  ✗ chess: NOT FOUND"
python3 -c "import nanobind; print(f'  ✓ nanobind: {nanobind.__version__}')" 2>/dev/null || echo "  ✗ nanobind: NOT FOUND"
python3 -c "import numpy; print(f'  ✓ numpy: {numpy.__version__}')" 2>/dev/null || echo "  ✗ numpy: NOT FOUND"
python3 -c "import fastapi; print(f'  ✓ fastapi: {fastapi.__version__}')" 2>/dev/null || echo "  ✗ fastapi: NOT FOUND"

echo ""
echo "=================================================="
echo "Setup complete!"
echo "=================================================="
echo ""
echo "Next steps:"
echo "  1. The C++ module will build automatically when you start the server"
echo "  2. Or build manually: ./build.sh"
echo ""

