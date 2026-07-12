cmake_minimum_required(VERSION 3.18)
project(md2d CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

add_executable(md2d
    src/main.cu
    src/vec3.cuh
    src/cells.cuh
    src/assign_cells.cuh
    src/init_neighborhood.h
    src/particle_device.cuh
    src/particle_host.h
    src/domain.h
    src/halo_exchange.h
    src/force_kernels.cuh
    src/integration.cuh
    src/migration.h
    src/vtk_output.h
    src/input.h
    src/json.hpp)

target_include_directories(md2d PRIVATE src)

set_target_properties(md2d PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON)
