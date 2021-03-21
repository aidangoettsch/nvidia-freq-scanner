cmake_minimum_required(VERSION 3.17)
project(benchmark_kernel LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)

add_library(benchmark_kernel OBJECT
        kernel.cu)

set_property(TARGET benchmark_kernel PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET benchmark_kernel PROPERTY CUDA_PTX_COMPILATION ON)
target_compile_options(
        benchmark_kernel
        PRIVATE
        "--keep"
)
