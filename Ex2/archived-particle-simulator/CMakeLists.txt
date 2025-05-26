cmake_minimum_required(VERSION 3.10)
project(ParticleSimulator)

set(CMAKE_CUDA_STANDARD 14)

# Specify the source files
set(SOURCES
    src/main.cu
    src/particle_sim.cu
    src/kernels/force_calculation.cu
    src/kernels/integration.cu
    src/utils/file_io.cpp
    src/utils/vtk_writer.cpp
    src/utils/config_parser.cpp
)

# Include directories
include_directories(
    src/headers
)

# Add executable
add_executable(ParticleSimulator ${SOURCES})

# Set CUDA flags
set_target_properties(ParticleSimulator PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Optionally link against any required libraries (e.g., for VTK)
# find_package(VTK REQUIRED)
# target_link_libraries(ParticleSimulator ${VTK_LIBRARIES})