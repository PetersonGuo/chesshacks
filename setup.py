#!/usr/bin/env python3
"""
100% CMake-Free Python build system for ChessHacks C++ extensions.

This build system uses pure Python and setuptools - absolutely no CMake required!
- Finds nanobind by importing it (no CMake queries)
- Compiles nanobind from source if needed (pure setuptools)
- Handles CUDA compilation without CMake
- All done with standard C++ compilers

Usage: python setup.py build_ext --inplace
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sysconfig


def find_nvcc():
    """Find nvcc compiler."""
    nvcc = shutil.which('nvcc')
    if nvcc:
        return nvcc
    # Try common CUDA installation paths
    cuda_paths = [
        '/usr/local/cuda/bin/nvcc',
        '/opt/cuda/bin/nvcc',
        os.path.expanduser('~/cuda/bin/nvcc'),
    ]
    for path in cuda_paths:
        if os.path.exists(path):
            return path
    return None


def get_cuda_paths():
    """Get CUDA include and library paths."""
    nvcc = find_nvcc()
    if not nvcc:
        return None, None
    
    # Get CUDA installation directory
    cuda_bin = os.path.dirname(nvcc)
    cuda_root = os.path.dirname(cuda_bin)
    
    include_dir = os.path.join(cuda_root, 'include')
    lib_dir = os.path.join(cuda_root, 'lib64')
    if not os.path.exists(lib_dir):
        lib_dir = os.path.join(cuda_root, 'lib')
    
    return include_dir, lib_dir


def get_nanobind_paths():
    """Get nanobind include directory and source files - NO CMake needed."""
    try:
        # Import nanobind to find its installation location
        import nanobind
        nanobind_dir = Path(nanobind.__file__).parent
        
        include_dir = nanobind_dir / 'include'
        src_dir = nanobind_dir / 'src'
        
        if not include_dir.exists():
            raise RuntimeError(f"nanobind include directory not found at {include_dir}")
        if not src_dir.exists():
            raise RuntimeError(f"nanobind src directory not found at {src_dir}")
        
        # Find all nanobind source files
        nanobind_sources = list(src_dir.glob('*.cpp'))
        if not nanobind_sources:
            raise RuntimeError(f"No nanobind source files found in {src_dir}")
        
        return str(include_dir), [str(s) for s in nanobind_sources]
    except ImportError:
        pass
    
    # Fallback: try to find nanobind in site-packages
    import site
    for site_pkg in site.getsitepackages():
        nanobind_path = Path(site_pkg) / 'nanobind'
        include_dir = nanobind_path / 'include'
        src_dir = nanobind_path / 'src'
        if include_dir.exists() and src_dir.exists():
            nanobind_sources = list(src_dir.glob('*.cpp'))
            if nanobind_sources:
                return str(include_dir), [str(s) for s in nanobind_sources]
    
    raise RuntimeError("Could not find nanobind. Please install it: pip install nanobind")


def find_nanobind_library():
    """Find nanobind static library if it exists - NO CMake needed."""
    # Check common locations in project
    project_root = Path(__file__).parent
    possible_paths = [
        project_root / 'libnanobind-static.a',
        project_root / 'build' / 'libnanobind-static.a',
    ]
    
    for path in possible_paths:
        if path.exists():
            return str(path)
    
    # Try to find in nanobind installation (if pre-built)
    try:
        import nanobind
        nanobind_dir = Path(nanobind.__file__).parent
        for lib_path in [
            nanobind_dir / 'libnanobind-static.a',
            nanobind_dir.parent / 'lib' / 'libnanobind-static.a',
        ]:
            if lib_path.exists():
                return str(lib_path)
    except ImportError:
        pass
    
    return None


class CUDABuildExt(build_ext):
    """Custom build extension that handles CUDA compilation."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.project_root = Path(__file__).parent
    
    def build_extensions(self):
        # Check for CUDA
        nvcc = find_nvcc()
        cuda_enabled = nvcc is not None
        
        if cuda_enabled:
            cuda_include, cuda_lib = get_cuda_paths()
            print(f"CUDA found: {nvcc}")
            print(f"CUDA include: {cuda_include}")
            print(f"CUDA lib: {cuda_lib}")
        else:
            print("CUDA not found - building CPU-only version")
        
        # Update compiler settings for each extension
        for ext in self.extensions:
            # Set C++ standard (nanobind requires C++17)
            ext.extra_compile_args = ['-std=c++17', '-O3']
            
            # Add CUDA support if available
            if cuda_enabled and hasattr(ext, 'cuda_sources') and ext.cuda_sources:
                ext.define_macros.append(('CUDA_ENABLED', '1'))
                if cuda_include:
                    ext.include_dirs.append(cuda_include)
                if cuda_lib:
                    ext.library_dirs.append(cuda_lib)
                ext.libraries.append('cudart')
                
                # Compile CUDA files separately
                self.compile_cuda_files(ext, nvcc, cuda_include)
        
        # Call parent build_extensions
        super().build_extensions()
    
    def compile_cuda_files(self, ext, nvcc, cuda_include):
        """Compile CUDA .cu files to object files."""
        if not hasattr(ext, 'cuda_sources') or not ext.cuda_sources:
            return
        
        # Get build directory
        build_dir = Path(self.build_temp)
        build_dir.mkdir(parents=True, exist_ok=True)
        
        # CUDA architectures to support
        cuda_archs = ['75', '80', '86', '89', '90']  # Modern GPUs
        arch_flags = []
        for arch in cuda_archs:
            arch_flags.extend(['-gencode', f'arch=compute_{arch},code=sm_{arch}'])
        
        # Get include directories for CUDA compilation
        include_args = []
        for inc_dir in ext.include_dirs:
            include_args.extend(['-I', inc_dir])
        if cuda_include:
            include_args.extend(['-I', cuda_include])
        
        # Compile each CUDA file
        cuda_objects = []
        for cuda_file in ext.cuda_sources:
            cuda_path = Path(cuda_file)
            if not cuda_path.is_absolute():
                cuda_path = self.project_root / cuda_path
            obj_name = cuda_path.stem + '.o'
            obj_path = build_dir / obj_name
            
            # Compile CUDA file
            cmd = [
                nvcc,
                '-c',
                '-std=c++17',
                '-O3',
                '--compiler-options', '-fPIC',
            ] + arch_flags + include_args + [
                '-o', str(obj_path),
                str(cuda_path)
            ]
            
            print(f"Compiling CUDA: {cuda_file}")
            print(f"Command: {' '.join(cmd)}")
            subprocess.run(cmd, check=True, cwd=str(self.project_root))
            cuda_objects.append(str(obj_path))
        
        # Add CUDA objects to extension (they'll be linked)
        if not hasattr(ext, 'extra_objects') or ext.extra_objects is None:
            ext.extra_objects = []
        ext.extra_objects.extend(cuda_objects)


# Get project root (needed in class method)
project_root = Path(__file__).parent
src_dir = project_root / 'src'

# Source files (always included)
cpp_sources = [
    'src/bindings.cpp',
    'src/chess_board.cpp',
    'src/move_ordering.cpp',
    'src/transposition_table.cpp',
    'src/evaluation.cpp',
    'src/nnue_evaluator.cpp',
    'src/opening_book.cpp',
    'src/search_and_utils.cpp',
    'src/cuda/cuda_utils.cpp',  # Always include (has fallback implementations)
]

# CUDA sources (optional)
cuda_sources = []
nvcc = find_nvcc()
if nvcc:
    cuda_sources = ['src/cuda/cuda_eval.cu']

# Get include directories
include_dirs = [
    str(src_dir),
    str(src_dir / 'cuda'),
]

# Add nanobind include and source files (NO CMake needed!)
try:
    nanobind_include, nanobind_sources = get_nanobind_paths()
    include_dirs.append(nanobind_include)
    print(f"Found nanobind include at: {nanobind_include}")
    print(f"Found {len(nanobind_sources)} nanobind source files")
except RuntimeError as e:
    print(f"Error: {e}")
    raise

# Get Python include
python_include = sysconfig.get_path('include')
include_dirs.append(python_include)

# Create extension - include nanobind sources directly (NO CMake needed!)
# First check if we should use pre-built library or compile from source
nanobind_lib = find_nanobind_library()
if nanobind_lib:
    # Use pre-built static library if available
    print(f"Using pre-built nanobind static library: {nanobind_lib}")
    ext_sources = cpp_sources
    extra_objects = [nanobind_lib]
else:
    # Compile nanobind from source - pure Python build!
    print("Compiling nanobind from source (no CMake needed)")
    ext_sources = cpp_sources + nanobind_sources
    extra_objects = []

ext = Extension(
    'c_helpers',
    sources=ext_sources,
    include_dirs=include_dirs,
    language='c++',
    extra_compile_args=['-std=c++17', '-O3'],
    extra_objects=extra_objects if extra_objects else None,
)

# Add CUDA sources as attribute (handled by custom build_ext)
if cuda_sources:
    ext.cuda_sources = cuda_sources
    if not ext.define_macros:
        ext.define_macros = []
    if nvcc:
        ext.define_macros.append(('CUDA_ENABLED', '1'))

setup(
    name='chesshacks',
    version='1.0.0',
    description='ChessHacks C++ extension module',
    ext_modules=[ext],
    cmdclass={'build_ext': CUDABuildExt},
    zip_safe=False,
)

