#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <string>

// Check if CUDA is available
bool is_cuda_available();

// Get CUDA device information
std::string get_cuda_info();

#endif // CUDA_UTILS_H
