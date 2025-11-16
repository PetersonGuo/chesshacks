# Building with Python (100% CMake-Free!)

This project now supports building C++ extensions using **pure Python and setuptools** - **absolutely no CMake required!**

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Build the extension
python setup.py build_ext --inplace

# Or use the convenience script
python build_python.py
```

## How It Works

The `setup.py` file uses Python's `setuptools` to:
- **Find nanobind by importing it** (no CMake queries)
- **Compile nanobind from source** if no pre-built library exists (pure setuptools)
- Automatically detect CUDA (if available)
- Compile C++ source files
- Compile CUDA `.cu` files (if CUDA is found)
- Link everything together into a Python extension module

**Zero CMake dependencies** - everything is done with Python and standard C++ compilers!

## Requirements

- Python 3.8+
- C++ compiler (g++, clang++, etc.)
- nanobind (installed via pip: `pip install nanobind`)
- CUDA toolkit (optional, for GPU acceleration)

## Building with CUDA

If CUDA is installed and `nvcc` is in your PATH, the build system will automatically:
- Detect CUDA
- Compile CUDA source files
- Link CUDA libraries
- Enable CUDA features in the extension

## Building without CUDA

The build will work fine without CUDA - it will just build a CPU-only version.

## Output

The compiled module `c_helpers.so` (or `c_helpers.cpython-*.so` on some systems) will be created in the project root directory.

## Troubleshooting

### nanobind not found
```bash
pip install nanobind
```

### Python development headers not found
- **macOS**: `xcode-select --install`
- **Ubuntu/Debian**: `sudo apt-get install python3-dev`
- **Fedora**: `sudo dnf install python3-devel`

### CUDA not detected
- Make sure `nvcc` is in your PATH
- Or set `CUDA_HOME` environment variable

## Comparison with CMake

| Feature | CMake | Python (setup.py) |
|---------|-------|-------------------|
| Dependencies | CMake 3.18+ | **Python + setuptools only** |
| CMake Required | ✅ Yes | ❌ **No!** |
| CUDA Support | ✅ | ✅ |
| Cross-platform | ✅ | ✅ |
| Build Speed | Fast | Fast |
| Ease of Use | Moderate | **Very Easy** |
| nanobind Build | Requires CMake | **Compiled from source with setuptools** |

## Notes

- **No CMake needed!** The build system finds nanobind by importing it and compiles it from source
- **By default, nanobind is always compiled from source** to ensure compatibility across different Python versions and platforms
- For faster local builds, you can use a pre-built library by setting `USE_NANOBIND_PREBUILT=1` environment variable (only if you're sure it's compatible)
- All compilation is done with standard C++ compilers (g++, clang++, etc.) - no build system dependencies
- This approach ensures the build works reliably on Linux, macOS, and Windows with any Python version

