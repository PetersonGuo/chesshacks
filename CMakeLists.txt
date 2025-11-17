cmake_minimum_required(VERSION 3.18...3.27)

# Initial project declaration with CXX
project(chesshacks LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(CHESSHACKS_ENABLE_CUDA "Enable CUDA acceleration" OFF)
option(CHESSHACKS_ENABLE_ROCM "Prefer ROCm-flavoured libtorch download (Linux only)" OFF)
option(CHESSHACKS_USE_SYSTEM_TORCH
    "Skip vendored libtorch download and rely on a pre-installed Torch_DIR"
    OFF)

include(FetchContent)

if(CHESSHACKS_ENABLE_ROCM AND NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(FATAL_ERROR "ROCm libtorch builds are only supported on Linux hosts.")
endif()

set(_libtorch_version_env "$ENV{CHESSHACKS_LIBTORCH_VERSION}")
if(_libtorch_version_env STREQUAL "")
    set(_libtorch_version_env "2.3.0")
endif()
set(CHESSHACKS_LIBTORCH_VERSION "${_libtorch_version_env}" CACHE STRING "libtorch version to fetch (e.g. 2.3.0)")

set(_libtorch_variant_env "$ENV{CHESSHACKS_LIBTORCH_VARIANT}")
set(CHESSHACKS_LIBTORCH_VARIANT "${_libtorch_variant_env}" CACHE STRING "Override libtorch variant (cpu, cu121, macos-arm64, rocm6.0, ...)")

set(_libtorch_rocm_env "$ENV{CHESSHACKS_LIBTORCH_ROCM_VERSION}")
if(_libtorch_rocm_env STREQUAL "")
    set(_libtorch_rocm_env "6.0")
endif()
set(CHESSHACKS_LIBTORCH_ROCM_VERSION "${_libtorch_rocm_env}" CACHE STRING "ROCm version suffix used when CHESSHACKS_ENABLE_ROCM=ON (e.g. 6.0)")

set(CUDA_ENABLED OFF)
if(CHESSHACKS_ENABLE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        set(CUDA_ENABLED ON)
        message(STATUS "CUDA enabled explicitly")

        if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
            set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89 90)
        endif()

        set(CMAKE_CUDA_STANDARD 17)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        find_package(CUDAToolkit REQUIRED)
    else()
        message(FATAL_ERROR "CUDA requested via CHESSHACKS_ENABLE_CUDA, but no CUDA compiler was found.")
    endif()
else()
    message(STATUS "CUDA disabled (set CHESSHACKS_ENABLE_CUDA=ON to enable)")
endif()

# Need Python interpreter + development headers/libs for nanobind.
# nanobind requires find_package(Python ...) to be called first
# Try to find Python with Development component first, but be flexible on macOS
find_package(Python 3.8 COMPONENTS Interpreter Development)

# If Development component not found, try to find just Interpreter and use provided paths
if(NOT Python_Development_FOUND)
    message(STATUS "Python Development component not found, trying Interpreter only...")
    find_package(Python 3.8 COMPONENTS Interpreter REQUIRED)

    # Use the paths provided via CMake variables (from build.sh)
    # Check both Python_* and Python3_* variables
    if(DEFINED Python3_LIBRARY AND NOT DEFINED Python_LIBRARY)
        set(Python_LIBRARY ${Python3_LIBRARY})
        message(STATUS "Using provided Python library: ${Python_LIBRARY}")
    elseif(DEFINED Python_LIBRARY)
        message(STATUS "Using provided Python library: ${Python_LIBRARY}")
    endif()
    if(DEFINED Python3_INCLUDE_DIR AND NOT DEFINED Python_INCLUDE_DIR)
        set(Python_INCLUDE_DIR ${Python3_INCLUDE_DIR})
        message(STATUS "Using provided Python include: ${Python_INCLUDE_DIR}")
    elseif(DEFINED Python_INCLUDE_DIR)
        message(STATUS "Using provided Python include: ${Python_INCLUDE_DIR}")
    endif()

    # Verify we have what we need
    if(NOT Python_LIBRARY OR NOT Python_INCLUDE_DIR)
        message(FATAL_ERROR "Python development libraries not found. Please ensure Python development headers are installed.")
    endif()
else()
    message(STATUS "Python Development component found successfully")
endif()

execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE nanobind_ROOT
)
find_package(nanobind CONFIG REQUIRED)
function(chesshacks_compute_libtorch_download out_url out_variant)
    string(TOLOWER "${CMAKE_SYSTEM_NAME}" platform_lower)
    string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" arch_lower)

    set(resolved_variant "${CHESSHACKS_LIBTORCH_VARIANT}")
    if(resolved_variant STREQUAL "")
        if(platform_lower STREQUAL "darwin")
            if(arch_lower MATCHES "arm64|aarch64")
                set(resolved_variant "macos-arm64")
            else()
                set(resolved_variant "macos-x86_64")
            endif()
        elseif(platform_lower STREQUAL "windows")
            if(CHESSHACKS_ENABLE_CUDA)
                set(resolved_variant "cu121")
            else()
                set(resolved_variant "cpu")
            endif()
        else()
            if(CHESSHACKS_ENABLE_ROCM)
                set(resolved_variant "rocm${CHESSHACKS_LIBTORCH_ROCM_VERSION}")
            elseif(CHESSHACKS_ENABLE_CUDA)
                set(resolved_variant "cu121")
            else()
                set(resolved_variant "cpu")
            endif()
        endif()
    endif()

    set(channel "")
    set(archive_name "")

    if(platform_lower STREQUAL "linux")
        if(resolved_variant STREQUAL "cpu")
            set(channel "cpu")
            set(archive_name "libtorch-shared-with-deps-${CHESSHACKS_LIBTORCH_VERSION}%2Bcpu.zip")
        elseif(resolved_variant MATCHES "^cu")
            set(channel "${resolved_variant}")
            set(archive_name "libtorch-cxx11-abi-shared-with-deps-${CHESSHACKS_LIBTORCH_VERSION}%2B${resolved_variant}.zip")
        elseif(resolved_variant MATCHES "^rocm")
            set(channel "${resolved_variant}")
            set(archive_name "libtorch-cxx11-abi-shared-with-deps-${CHESSHACKS_LIBTORCH_VERSION}%2B${resolved_variant}.zip")
        else()
            message(FATAL_ERROR "Unsupported Linux libtorch variant '${resolved_variant}'.")
        endif()
    elseif(platform_lower STREQUAL "darwin")
        set(channel "macos")
        if(resolved_variant STREQUAL "macos-arm64")
            set(archive_name "libtorch-macos-arm64-${CHESSHACKS_LIBTORCH_VERSION}.zip")
        elseif(resolved_variant STREQUAL "macos-x86_64")
            set(archive_name "libtorch-macos-${CHESSHACKS_LIBTORCH_VERSION}.zip")
        else()
            message(FATAL_ERROR "Unsupported macOS libtorch variant '${resolved_variant}'.")
        endif()
    elseif(platform_lower STREQUAL "windows")
        if(resolved_variant STREQUAL "cpu")
            set(channel "cpu")
            set(archive_name "libtorch-win-shared-with-deps-${CHESSHACKS_LIBTORCH_VERSION}%2Bcpu.zip")
        elseif(resolved_variant MATCHES "^cu")
            set(channel "${resolved_variant}")
            set(archive_name "libtorch-win-shared-with-deps-${CHESSHACKS_LIBTORCH_VERSION}%2B${resolved_variant}.zip")
        else()
            message(FATAL_ERROR "Unsupported Windows libtorch variant '${resolved_variant}'.")
        endif()
    else()
        message(FATAL_ERROR "Unsupported host platform '${CMAKE_SYSTEM_NAME}' for libtorch auto-download.")
    endif()

    if(channel STREQUAL "" OR archive_name STREQUAL "")
        message(FATAL_ERROR "Failed to resolve libtorch download channel/archive.")
    endif()

    set(download_url "https://download.pytorch.org/libtorch/${channel}/${archive_name}")
    set(${out_url} "${download_url}" PARENT_SCOPE)
    set(${out_variant} "${resolved_variant}" PARENT_SCOPE)
endfunction()

function(chesshacks_ensure_libtorch)
    chesshacks_compute_libtorch_download(_libtorch_url _libtorch_variant)
    set(_stage_dir "${CMAKE_BINARY_DIR}/third_party/libtorch")

    FetchContent_Declare(
        chesshacks_libtorch
        URL "${_libtorch_url}"
        SOURCE_DIR "${_stage_dir}"
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )

    FetchContent_GetProperties(chesshacks_libtorch)
    if(NOT chesshacks_libtorch_POPULATED)
        message(STATUS "Fetching libtorch ${CHESSHACKS_LIBTORCH_VERSION} (${_libtorch_variant}) from ${_libtorch_url}")
        FetchContent_Populate(chesshacks_libtorch)
    endif()

    set(_libtorch_root "${_stage_dir}/libtorch")
    if(NOT EXISTS "${_libtorch_root}")
        set(_libtorch_root "${_stage_dir}")
    endif()

    if(NOT EXISTS "${_libtorch_root}/include/torch/script.h")
        message(FATAL_ERROR "libtorch download failed: expected headers not found in ${_libtorch_root}")
    endif()

    set(Torch_DIR "${_libtorch_root}/share/cmake/Torch" CACHE PATH "Torch CMake config dir" FORCE)
    set(CHESSHACKS_LIBTORCH_ROOT "${_libtorch_root}" CACHE PATH "Resolved libtorch root directory" FORCE)
endfunction()

function(chesshacks_clear_cached_torch_vars)
    foreach(_var
        Torch_FOUND
        Torch_VERSION
        TORCH_LIBRARIES
        TORCH_LIBRARY
        TORCH_LIBRARY_TYPE
        torch_LIBRARY
        torch_cpu_LIBRARY
        torch_cuda_LIBRARY
        c10_LIBRARY
        c10_cuda_LIBRARY
        kineto_LIBRARY
        Caffe2_DIR)
        if(DEFINED ${_var})
            unset(${_var} CACHE)
        endif()
    endforeach()
endfunction()

if(CHESSHACKS_USE_SYSTEM_TORCH)
    if(NOT Torch_DIR)
        message(FATAL_ERROR
            "CHESSHACKS_USE_SYSTEM_TORCH=ON requires Torch_DIR to be set.")
    endif()
    message(STATUS "Using system-provided libtorch at ${Torch_DIR}")
else()
    chesshacks_ensure_libtorch()
endif()
chesshacks_clear_cached_torch_vars()
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Add include paths
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/third_party/surge/src)

if(CUDA_ENABLED)
    add_subdirectory(lib/cuda)
endif()

FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)
FetchContent_MakeAvailable(googletest)

# Core engine sources (shared by module and native tests)
set(CHESS_CORE_SOURCES
    lib/core/move_ordering.cpp
    lib/core/transposition_table.cpp
    lib/core/evaluation.cpp
    lib/core/nnue_evaluator.cpp
    lib/core/search_common.cpp
    lib/core/search_alpha_beta.cpp
    lib/core/search_pgn.cpp
    lib/bitboard/bitboard_state.cpp
    lib/cuda/cuda_utils.cpp
    lib/third_party/surge_inline.cpp
    third_party/surge/src/position.cpp
    third_party/surge/src/tables.cpp
    third_party/surge/src/types.cpp
)

set(SOURCE_FILES
    lib/core/bindings.cpp
    lib/core/search_best_move.cpp
    ${CHESS_CORE_SOURCES}
)

set_source_files_properties(third_party/surge/src/tables.cpp
    PROPERTIES COMPILE_FLAGS "-include cstring")

nanobind_add_module(c_helpers ${SOURCE_FILES})
target_compile_definitions(c_helpers PRIVATE BUILDING_PY_MODULE)
target_include_directories(c_helpers PRIVATE
    ${CMAKE_SOURCE_DIR}/include/cuda
)

if(CUDA_ENABLED)
    if(NOT CHESSHACKS_CUDA_LIBRARY)
        message(FATAL_ERROR "CUDA enabled but cuda helpers were not configured.")
    endif()
    target_compile_definitions(c_helpers PRIVATE CUDA_ENABLED)
    target_include_directories(c_helpers PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    target_link_libraries(c_helpers PRIVATE ${CHESSHACKS_CUDA_LIBRARY} CUDA::cudart ${TORCH_LIBRARIES})
    set_target_properties(c_helpers PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
    message(STATUS "Building with CUDA support")
    message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
else()
    target_link_libraries(c_helpers PRIVATE ${TORCH_LIBRARIES})
    message(STATUS "Building CPU-only version")
endif()

set(_torch_rpath "")
if(DEFINED CHESSHACKS_LIBTORCH_ROOT)
    set(_torch_rpath "${CHESSHACKS_LIBTORCH_ROOT}/lib")
elseif(Torch_DIR)
    get_filename_component(_torch_root "${Torch_DIR}/../.." ABSOLUTE)
    set(_torch_rpath "${_torch_root}/lib")
endif()
if(_torch_rpath)
    set_target_properties(c_helpers PROPERTIES
        BUILD_RPATH "${_torch_rpath}"
        INSTALL_RPATH "${_torch_rpath}"
    )
endif()

add_subdirectory(tests ${CMAKE_BINARY_DIR}/tests)

# Create a symlink from build/tests to ../tests for easier test execution
set(CHESSHACKS_TESTS_SYMLINK "${CMAKE_BINARY_DIR}/tests_src")
set(CHESSHACKS_BENCH_SYMLINK "${CMAKE_BINARY_DIR}/benchmarks_src")

add_custom_command(TARGET c_helpers POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E remove -f ${CHESSHACKS_TESTS_SYMLINK}
    COMMAND ${CMAKE_COMMAND} -E create_symlink
    ${CMAKE_SOURCE_DIR}/tests
    ${CHESSHACKS_TESTS_SYMLINK}
    COMMENT "Creating symlink to tests directory"
)

add_custom_command(TARGET c_helpers POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E remove -f ${CHESSHACKS_BENCH_SYMLINK}
    COMMAND ${CMAKE_COMMAND} -E create_symlink
    ${CMAKE_SOURCE_DIR}/benchmarks
    ${CHESSHACKS_BENCH_SYMLINK}
    COMMENT "Creating symlink to benchmarks directory"
)
