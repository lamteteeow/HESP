# Agent Context — rigid_body mini-project

## What this project is

A GPU-accelerated **DEM (Discrete Element Method) rigid body simulator** written in CUDA C++17.

Simulates two object types:
- **Spheres** — spring-dashpot contact forces with Coulomb friction
- **Convex meshes** (boxes, etc.) — GJK + EPA collision detection, same force model

Each simulation step:
1. Assign particles to uniform-grid cells (`assign_cells.cuh`)
2. Compute contact forces checking only the 27 neighboring cells (`symplecticEuler.cuh`)
3. Symplectic Euler integration for linear motion; quaternion integration for rotation
4. Every `stepsPerFrame` steps: download state to CPU and write VTK files

Output goes to `out_vtk_<scenario><steps>/` directories, readable by ParaView.

## File map

| File(s) | Role |
|---|---|
| `main.cu` | Entry point, simulation loop |
| `SceneHost.h` | Top-level scene: loads JSON, owns SphereHost / PlaneHost / MeshHost |
| `SphereHost.h`, `PlaneHost.h`, `MeshHost.h` | CPU-side data + `upload()` / `download()` |
| `SphereDevice.cuh`, `PlaneDevice.cuh`, `MeshDevice.cuh` | Plain GPU pointer structs |
| `SphereVTK.h`, `PlaneVTK.h`, `MeshVTK.h` | CPU structs for VTK download |
| `symplecticEuler.cuh` | All CUDA kernels: `cfSphereOnSphere`, `cfSphereOnPlane`, `cfMeshOnMesh`, `cfMeshOnPlane`, `IntegrateVelAndPos`, `IntegrateVelAndPosMesh` |
| `contact.cuh` | GJK + EPA collision detection for mesh-mesh |
| `simplex.cuh`, `face.cuh` | GJK/EPA data structures |
| `assign_cells.cuh` | `assignCell` kernel + `computeCellIndex` |
| `init_neighborhood.h` | CPU-side cell neighbor list init |
| `forceComputations.cuh` | Helper device functions (currently not `#include`'d anywhere — dead code) |
| `Vec3.cuh`, `Quaternion.cuh` | Math types, `__host__ __device__` throughout |
| `VTK.h` | `writeVTK`, `writeMeshVTK`, `writeBoundingBoxWallsVTK` |
| `read_input.h`, `input_file.h` | JSON parsing helpers |
| `json.hpp` | nlohmann/json single-header library |
| `*.json` | Example scenes (sphere blocks, box tower, friction test) |
| `*.py` | Python scripts to generate scene JSON files |

## Build

**Preferred (CMake):**
```bash
cmake -B build -S .
cmake --build build
```

**Quick (Makefile, Linux/WSL):**
```bash
make          # produces ./rigid_body
make clean
```

**Run:**
```bash
./rigid_body <scene.json> [maxSteps]
# e.g.
./rigid_body boxTower.json 50000
```

Compiler: `nvcc`, CUDA standard C++17, separable compilation enabled.

## Coding conventions

- `d_` prefix on all device pointers (e.g. `d_positions`, `d_forces`)
- `h_` prefix on temporary host-side arrays
- `.cuh` for files with CUDA device code; `.h` for pure C++ headers
- Host structs own the data lifecycle; device structs are plain pointer bags
- Kernels live in `symplecticEuler.cuh`; math helpers in `Vec3.cuh` / `Quaternion.cuh`
- Use `__host__ __device__ __forceinline__` for any function called from both sides
- Error-check every kernel launch with `cudaGetLastError()` immediately after `<<<>>>`

## Known issues / TODOs

- `forceComputations.cuh` — `sphereOnSphere` and `sphereOnPlane` are implemented but the file is never `#include`'d. Either delete it or refactor the kernels in `symplecticEuler.cuh` to call these helpers.
- `forceComputations.cuh` — `sphereOnPlane` body is a stub (marked TODO).
- `cfSphereOnPlane` — Baumgarte position correction is commented out (lines ~371–374 in `symplecticEuler.cuh`).
- `MeshHost.h` — `resizeAll` has a blank comment block where mesh-offset resizing was meant to go.
- No CUDA error checking on `cudaMalloc` / `cudaMemcpy` calls in host structs.
- No memory is freed (`cudaFree` / `delete[]`) — fine for a short-lived simulation binary but worth noting.

---

## Tasks

<!-- Add your next tasks here. Be as specific as possible: what to implement, which files to touch, and any constraints. -->

- [ ] TODO

