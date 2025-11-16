#include "cuda_utils.h"
#ifdef CUDA_ENABLED
#include <cuda_runtime.h>
#endif

bool is_cuda_available() {
#ifdef CUDA_ENABLED
  int device_count = 0;
  cudaError_t error = cudaGetDeviceCount(&device_count);
  return (error == cudaSuccess && device_count > 0);
#else
  return false;
#endif
}

std::string get_cuda_info() {
#ifdef CUDA_ENABLED
  int device_count = 0;
  cudaError_t error = cudaGetDeviceCount(&device_count);

  if (error != cudaSuccess || device_count == 0) {
    return "CUDA not available";
  }

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);

  return std::string(prop.name) + " (CUDA Compute " +
         std::to_string(prop.major) + "." + std::to_string(prop.minor) + ")";
#else
  return "CUDA not compiled (use nvcc to enable)";
#endif
}
