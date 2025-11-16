#!/usr/bin/env python3
"""
Simple Python build script - alternative to build.sh
Just run: python build_python.py
"""

import subprocess
import sys
from pathlib import Path

def main():
    project_root = Path(__file__).parent
    
    print("=" * 50)
    print("ChessHacks Python Build Script")
    print("=" * 50)
    print()
    
    # Build the extension
    print("Building C++ extension with setuptools...")
    print()
    
    cmd = [sys.executable, 'setup.py', 'build_ext', '--inplace']
    result = subprocess.run(cmd, cwd=str(project_root))
    
    if result.returncode != 0:
        print("Build failed!")
        sys.exit(1)
    
    print()
    print("=" * 50)
    print("Build Complete!")
    print("=" * 50)
    print()
    print("The c_helpers module should now be available.")
    print("Test it with: python -c 'import c_helpers; print(c_helpers)'")

if __name__ == '__main__':
    main()

