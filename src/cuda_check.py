"""CUDA availability checker for chess engine"""


def is_cuda_available():
    """
    Check if CUDA is available via PyTorch.
    Returns (bool, str): (is_available, device_info)
    """
    try:
        import torch

        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device_name = (
                torch.cuda.get_device_name(0) if device_count > 0 else "Unknown"
            )
            cuda_version = torch.version.cuda
            return (
                True,
                f"{device_count} GPU(s) available: {device_name} (CUDA {cuda_version})",
            )
        else:
            return False, "PyTorch installed but no CUDA devices found"
    except ImportError:
        return False, "PyTorch not installed"
    except Exception as e:
        return False, f"Error checking CUDA: {str(e)}"


def get_cuda_info():
    """
    Get detailed CUDA information.
    Returns dict with CUDA details or None if not available.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return None

        info = {
            "available": True,
            "device_count": torch.cuda.device_count(),
            "cuda_version": torch.version.cuda,
            "devices": [],
        }

        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            device_info = {
                "id": i,
                "name": torch.cuda.get_device_name(i),
                "memory_gb": props.total_memory / 1e9,
                "compute_capability": f"{props.major}.{props.minor}",
            }
            info["devices"].append(device_info)

        return info
    except:
        return None


if __name__ == "__main__":
    # Test the checker
    available, msg = is_cuda_available()
    print(f"CUDA Available: {available}")
    print(f"Message: {msg}")

    if available:
        info = get_cuda_info()
        if info:
            print("\nDetailed CUDA Info:")
            print(f"  CUDA Version: {info['cuda_version']}")
            print(f"  Device Count: {info['device_count']}")
            for dev in info["devices"]:
                print(f"  GPU {dev['id']}: {dev['name']}")
                print(f"    Memory: {dev['memory_gb']:.2f} GB")
                print(f"    Compute: {dev['compute_capability']}")
