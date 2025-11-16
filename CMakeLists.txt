cmake_minimum_required(VERSION 3.18...3.27)

# Initial project declaration with CXX
project(chesshacks LANGUAGES CXX)

# Try to find CUDA compiler using 'which nvcc'
if(NOT DEFINED CMAKE_CUDA_COMPILER)
    execute_process(
        COMMAND which nvcc
        OUTPUT_VARIABLE NVCC_PATH
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(NVCC_PATH)
        set(CMAKE_CUDA_COMPILER "${NVCC_PATH}")
        message(STATUS "Found nvcc at: ${NVCC_PATH}")
    endif()
endif()

# Check if CUDA is available
include(CheckLanguage)
check_language(CUDA)

# Enable CUDA by default if found
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CUDA_ENABLED ON)
    message(STATUS "CUDA found - GPU acceleration ENABLED")
    
    # Set CUDA architecture (auto-detect or use common values)
    # Common values: 75 (Turing), 80/86 (Ampere), 89 (Ada), 90 (Hopper), 100 (Blackwell)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89 90)  # Support modern GPUs
    endif()
    
    # Set CUDA standard
    set(CMAKE_CUDA_STANDARD 14)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    
    # Find CUDA Toolkit
    find_package(CUDAToolkit REQUIRED)
else()
    set(CUDA_ENABLED OFF)
    message(STATUS "CUDA not found - GPU acceleration disabled (CPU-only build)")
endif()

# Need Python interpreter + development headers/libs for nanobind.
find_package(Python 3.8 COMPONENTS Interpreter Development REQUIRED)

execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE nanobind_ROOT
)
find_package(nanobind CONFIG REQUIRED)

# Add the source directory to include paths
include_directories(${CMAKE_SOURCE_DIR}/src)

# Source files
set(SOURCE_FILES
    src/bindings.cpp
    src/chess_board.cpp
    src/move_ordering.cpp
    src/transposition_table.cpp
    src/evaluation.cpp
    src/opening_book.cpp
    src/search_and_utils.cpp
    src/cuda/cuda_utils.cpp  # Always compile (has fallback when CUDA_ENABLED is not defined)
)

# Add CUDA files if CUDA is enabled
if(CUDA_ENABLED)
    list(APPEND SOURCE_FILES 
        src/cuda/cuda_eval.cu
    )
    
    # Create the extension module with CUDA support
    nanobind_add_module(c_helpers ${SOURCE_FILES})
    
    # Set CUDA compilation flags and include directories
    target_compile_definitions(c_helpers PRIVATE CUDA_ENABLED)
    target_include_directories(c_helpers PRIVATE 
        ${CUDAToolkit_INCLUDE_DIRS}
        ${CMAKE_SOURCE_DIR}/src/cuda
    )
    target_link_libraries(c_helpers PRIVATE CUDA::cudart)
    set_target_properties(c_helpers PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
    
    message(STATUS "Building with CUDA support")
    message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
else()
    # Create the extension module without CUDA
    nanobind_add_module(c_helpers ${SOURCE_FILES})
    message(STATUS "Building CPU-only version")
endif()

# Create a symlink from build/tests to ../tests for easier test execution
add_custom_command(TARGET c_helpers POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E create_symlink
        ${CMAKE_SOURCE_DIR}/tests
        ${CMAKE_BINARY_DIR}/tests
    COMMENT "Creating symlink to tests directory"
)
