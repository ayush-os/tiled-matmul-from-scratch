cmake_minimum_required(VERSION 3.10)
project(matmul C CXX CUDA)

find_package(CUDA REQUIRED)

set(CMAKE_CUDA_ARCHITECTURES "80" CACHE STRING "Target CUDA architectures")

set(CUDA_SEPARABLE_COMPILATION ON)
set(CUDA_PROPAGATE_HOST_FLAGS OFF)

add_executable(matmul.cu)

target_include_directories(matmul PRIVATE ${CUDA_INCLUDE_DIRS})

target_compile_options(matmul PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-std=c++14>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

target_link_libraries(matmul PRIVATE
    ${CUDA_LIBRARIES}
)

set_target_properties(matmul PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)