cmake_minimum_required(VERSION 3.15...3.27)
project(chesshacks)

if (CMAKE_VERSION VERSION_LESS 3.18)
    set(DEV_MODULE Development)
else()
    set(DEV_MODULE Development.Module)
endif()
find_package(Python 3.8 COMPONENTS Interpreter ${DEV_MODULE} REQUIRED)

execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE nanobind_ROOT
)
find_package(nanobind CONFIG REQUIRED)

# Add the source directory to include paths
include_directories(${CMAKE_SOURCE_DIR}/src)

# Create the extension module with bindings and implementation files
nanobind_add_module(c_helpers
    src/bindings.cpp
    src/chess_board.cpp
    src/cuda_utils.cpp
    src/move_ordering.cpp
    src/transposition_table.cpp
    src/evaluation.cpp
    src/opening_book.cpp
    src/search_and_utils.cpp
)

# Create a symlink from build/tests to ../tests for easier test execution
add_custom_command(TARGET c_helpers POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E create_symlink
        ${CMAKE_SOURCE_DIR}/tests
        ${CMAKE_BINARY_DIR}/tests
    COMMENT "Creating symlink to tests directory"
)