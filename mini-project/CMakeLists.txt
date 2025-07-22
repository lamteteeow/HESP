cmake_minimum_required(VERSION 3.31)
project(rigid_body CUDA)

set(CMAKE_CUDA_STANDARD 17)

add_executable(rigid_body main.cu
        input_file.h
        VTK.h
        assign_cells.cuh
        init_neighborhood.h
        symplecticEuler.cuh
        Vec3.cuh
        read_input.h
        Quaternion.cuh
        SphereHost.h
        SceneHost.h
        SphereDevice.cuh
        SphereVTK.h
        Cells.cuh
        PlaneHost.h
        PlaneDevice.cuh
        PlaneVTK.h
        MeshHost.h
        MeshVTK.h
        MeshDevice.cuh
        forceComputations.cuh
        debug.cuh
        contact.cuh
        simplex.cuh
        face.cuh)

set_target_properties(rigid_body PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

