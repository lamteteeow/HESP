cmake_minimum_required(VERSION 3.18)
project(md2d CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

add_executable(md2d
    main.cu
    check_cuda.h
    vec3.cuh
    assign_cells.cuh
    energy_diagnostics.cuh
    init_neighborhood.h
    particle_device.cuh
    particle_host.h
    domain.h
    halo_exchange.h
    force_kernels.cuh
    integration.cuh
    migration.h
    vtk_output.h
    input.h
    json.hpp)

set_target_properties(md2d PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON)
