#!/usr/bin/env python3
"""Test CUDA availability detection from multiple sources"""

import os
import sys

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(parent_dir, "build"))
sys.path.insert(0, parent_dir)


def test_cpp_cuda_detection():
    """Test CUDA detection from C++ bindings"""
    print("=" * 70)
    print("CUDA Detection via C++ Bindings")
    print("=" * 70)

    try:
        import c_helpers

        is_available = c_helpers.is_cuda_available()
        cuda_info = c_helpers.get_cuda_info()

        print(f"\nCUDA Available: {is_available}")
        print(f"CUDA Info: {cuda_info}")

        if is_available:
            print("\n✓ CUDA is compiled and available!")
            print("  - Engine can use GPU acceleration")
            print("  - alpha_beta_cuda() will use GPU")
        else:
            print("\n✗ CUDA not available from C++")
            if "not compiled" in cuda_info:
                print("  - C++ code not compiled with nvcc")
                print("  - To enable: use CUDA compiler and toolkit")
            else:
                print("  - No CUDA devices detected")

        return is_available

    except Exception as e:
        print(f"\n✗ Error checking C++ CUDA: {e}")
        return False


def test_python_cuda_detection():
    """Test CUDA detection from Python utilities"""
    print("\n" + "=" * 70)
    print("CUDA Detection via Python (PyTorch)")
    print("=" * 70)

    try:
        from cuda_check import is_cuda_available, get_cuda_info

        cuda_available, msg = is_cuda_available()
        print(f"\nCUDA Available: {cuda_available}")
        print(f"Message: {msg}")

        if cuda_available:
            info = get_cuda_info()
            if info:
                print(f"\n✓ PyTorch CUDA Details:")
                print(f"  - CUDA Version: {info['cuda_version']}")
                print(f"  - Device Count: {info['device_count']}")
                for dev in info["devices"]:
                    print(f"  - GPU {dev['id']}: {dev['name']}")
                    print(f"    * Memory: {dev['memory_gb']:.2f} GB")
                    print(f"    * Compute: {dev['compute_capability']}")
        else:
            print("\n✗ CUDA not available from PyTorch")
            if "not installed" in msg:
                print("  - Install PyTorch with CUDA support:")
                print(
                    "    conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia"
                )

        return cuda_available

    except ImportError:
        print("\n✗ Python CUDA checker not available")
        print("  - Module src/utils/cuda_check.py not found")
        return False
    except Exception as e:
        print(f"\n✗ Error checking Python CUDA: {e}")
        return False


def test_nvidia_smi():
    """Test CUDA detection via nvidia-smi command"""
    print("\n" + "=" * 70)
    print("CUDA Detection via nvidia-smi")
    print("=" * 70)

    try:
        import subprocess

        result = subprocess.run(
            ["nvidia-smi"], capture_output=True, timeout=2, check=False
        )

        if result.returncode == 0:
            print("\n✓ nvidia-smi available")
            print("\nGPU Information:")
            output = result.stdout.decode("utf-8")
            # Print first few lines of nvidia-smi output
            lines = output.split("\n")[:10]
            for line in lines:
                if line.strip():
                    print(f"  {line}")
            return True
        else:
            print("\n✗ nvidia-smi returned error")
            return False

    except FileNotFoundError:
        print("\n✗ nvidia-smi not found")
        print("  - NVIDIA drivers not installed")
        print("  - No NVIDIA GPU present")
        return False
    except subprocess.TimeoutExpired:
        print("\n✗ nvidia-smi timeout")
        return False
    except Exception as e:
        print(f"\n✗ Error running nvidia-smi: {e}")
        return False


def test_integrated_detection():
    """Test the integrated detection from main.py"""
    print("\n" + "=" * 70)
    print("Integrated CUDA Detection (main.py logic)")
    print("=" * 70)

    try:
        import c_helpers
        from cuda_check import is_cuda_available as py_is_cuda_available

        # Method 1: C++ bindings
        cpp_cuda = False
        try:
            cpp_cuda = c_helpers.is_cuda_available()
            cpp_info = c_helpers.get_cuda_info()
        except AttributeError:
            cpp_info = "Not available"

        # Method 2: Python/PyTorch
        py_cuda, py_msg = py_is_cuda_available()

        # Method 3: nvidia-smi
        try:
            import subprocess

            smi_result = subprocess.run(
                ["nvidia-smi"], capture_output=True, timeout=1, check=False
            )
            smi_cuda = smi_result.returncode == 0
        except:
            smi_cuda = False

        print("\nDetection Methods:")
        print(f"  1. C++ bindings:    {cpp_cuda} - {cpp_info}")
        print(f"  2. PyTorch:         {py_cuda} - {py_msg}")
        print(f"  3. nvidia-smi:      {smi_cuda}")

        # Final decision (same logic as main.py)
        has_cuda = cpp_cuda or py_cuda or smi_cuda

        print(f"\n{'='*70}")
        print(f"Final Decision: CUDA Available = {has_cuda}")
        print(f"{'='*70}")

        if has_cuda:
            print("\n✓ Chess engine will use GPU acceleration when available")
            print("  - alpha_beta_cuda() will be preferred")
            print("  - Batch NNUE evaluation on GPU (future)")
        else:
            print("\n✓ Chess engine will use CPU optimizations")
            print("  - alpha_beta_optimized() with all 15 features")
            print("  - Parallel search with multiple CPU threads")
            print("  - Still very fast: 57-210x speedup over baseline")

        return has_cuda

    except Exception as e:
        print(f"\n✗ Error in integrated detection: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all CUDA detection tests"""
    print("\n" + "=" * 70)
    print(" " * 20 + "CUDA DETECTION TEST SUITE")
    print("=" * 70)

    results = {
        "C++ Bindings": test_cpp_cuda_detection(),
        "Python/PyTorch": test_python_cuda_detection(),
        "nvidia-smi": test_nvidia_smi(),
        "Integrated": test_integrated_detection(),
    }

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for method, available in results.items():
        status = "✓ Available" if available else "✗ Not Available"
        print(f"{method:20s}: {status}")

    any_cuda = any(results.values())
    print("\n" + "=" * 70)
    if any_cuda:
        print("RESULT: CUDA detected via at least one method")
        print("Chess engine can leverage GPU acceleration")
    else:
        print("RESULT: No CUDA detected")
        print("Chess engine will use optimized CPU implementation")
        print("(Still very fast with 15 advanced features!)")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
