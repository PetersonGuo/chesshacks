if(NOT CUDA_ENABLED)
  message(STATUS "CUDA not enabled; skipping lib/cuda.")
  return()
endif()

add_library(chesshacks_cuda STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda_eval.cu
)

target_include_directories(chesshacks_cuda PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/cuda
    ${CMAKE_SOURCE_DIR}/third_party/surge/src
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_compile_definitions(chesshacks_cuda PRIVATE BUILDING_PY_MODULE CUDA_ENABLED)

set_target_properties(chesshacks_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    OUTPUT_NAME chesshacks_cuda
)

install(TARGETS chesshacks_cuda
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)

install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/cuda/
        DESTINATION include/chesshacks/cuda
        FILES_MATCHING PATTERN "*.h" PATTERN "*.cuh")

set(CHESSHACKS_CUDA_LIBRARY chesshacks_cuda PARENT_SCOPE)

