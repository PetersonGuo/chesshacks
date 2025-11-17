if(NOT CHESS_CORE_SOURCES)
  message(FATAL_ERROR "CHESS_CORE_SOURCES is not defined; include tests after core sources.")
endif()

enable_testing()

set_source_files_properties(
  ${CMAKE_SOURCE_DIR}/third_party/surge/src/tables.cpp
  PROPERTIES COMPILE_FLAGS "-include cstring"
)

set(CPP_TEST_SOURCES
  cpp/search_eval_tests.cpp
  cpp/pgn_tests.cpp
  cpp/cuda_tests.cpp
  cpp/nnue_tests.cpp
)

set(_core_sources_abs)
foreach(src ${CHESS_CORE_SOURCES})
  if(IS_ABSOLUTE "${src}")
    list(APPEND _core_sources_abs "${src}")
  else()
    list(APPEND _core_sources_abs "${CMAKE_SOURCE_DIR}/${src}")
  endif()
endforeach()

add_executable(cpp_tests
  ${CPP_TEST_SOURCES}
  ${_core_sources_abs}
)

target_include_directories(cpp_tests PRIVATE
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/third_party/surge/src
  ${CMAKE_SOURCE_DIR}/include/cuda
)

target_link_libraries(cpp_tests PRIVATE gtest_main ${TORCH_LIBRARIES})
target_compile_definitions(cpp_tests PRIVATE CHESSHACKS_SOURCE_DIR=\"${CMAKE_SOURCE_DIR}\")

if(CUDA_ENABLED)
  target_compile_definitions(cpp_tests PRIVATE CUDA_ENABLED)
  target_include_directories(cpp_tests PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
endif()

set_target_properties(cpp_tests PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

add_test(NAME cpp_tests COMMAND cpp_tests)

