# Agent Context — ss26-project (md2d)

## What this project is

A GPU-accelerated 2D molecular dynamics / DEM simulator with **N-GPU domain decomposition**.

The simulation domain is split evenly along the X axis into N slices, one per GPU.
Each GPU owns particles in its slice and exchanges halo particles with its left
and right neighbors. Interior GPUs have two halo neighbors; edge GPUs have one.

At each step:
1. **Halo exchange** — each GPU sends boundary strips to left/right neighbors and receives their strips as ghost/read-only particles, so cross-boundary contacts are captured.
2. **Cell assignment** — all particles (owned + halo) are inserted into a uniform cell grid local to each GPU.
3. **Force computation** — spring-dashpot (DEM) contact forces, computed only for owned particles.
4. **Integration** — symplectic Euler, z=0 constraint enforced, reflective wall BCs.
5. **Particle migration** — particles that crossed their owning slice are redistributed across all GPUs.
6. **VTK output** — every `steps_per_frame` steps, all GPUs dump to a single VTK file.

For 2D, `num_cells.z = 1` and all z-coordinates are clamped to 0. Extension to 3D only requires removing the z-constraint and setting `num_cells.z > 1`.

## File map

| File | Role |
|---|---|
| `main.cu` | Entry point; simulation loop |
| `Domain.h` | `Domain` struct + `buildDomains()` for N-GPU X-split |
| `ParticleDevice.cuh` | GPU pointer struct (owned + halo layout) |
| `ParticleHost.h` | CPU buffer; `upload()` / `download()` / `freeParticleDevice()` |
| `halo_exchange.h` | `collectHalo()` / `uploadHalo()` / `exchangeHalos()` — N-GPU |
| `migration.h` | `migrateParticles()` — full CPU round-trip redistribution, N-GPU |
| `force_kernels.cuh` | `computeContactForces` kernel (spring-dashpot DEM) |
| `integration.cuh` | `integrate` kernel (symplectic Euler, reflective walls) |
| `assign_cells.cuh` | `assignCell` kernel + `computeCellIndex` (from mini-project) |
| `init_neighborhood.h` | `initCellNeighborhood()` — CPU-side 27-neighbor table (from mini-project) |
| `Cells.cuh` | `Cells` helper struct (from mini-project) |
| `Vec3.cuh` | `Vec3` math type + free functions (from mini-project) |
| `vtk_output.h` | `writeParticlesVTK()` — positions, velocities, radii |
| `input.h` | `loadScene()` / `splitAt()` — JSON parsing |
| `json.hpp` | nlohmann/json single-header (from mini-project) |
| `scenes/two_discs.json` | Two particles colliding at the domain boundary |
| `gen_lattice.py` | Python script: generates a 2D hexagonal lattice JSON scene |

## Build

```bash
cmake -B build -S .
cmake --build build
```

Binary: `build/md2d`

## Run

```bash
# Uses all available GPUs by default
./build/md2d scenes/two_discs.json 10000

# Explicitly request N GPUs (3rd argument)
./build/md2d scenes/lattice.json 50000 4

# Generate a lattice scene and run it
python gen_lattice.py > scenes/lattice.json
./build/md2d scenes/lattice.json 50000
```

Usage: `./build/md2d <scene.json> [max_steps] [num_gpus]`
- `max_steps` defaults to 100000
- `num_gpus` defaults to all available CUDA devices

Output goes to `out_vtk_<scene>_<steps>/`. Open in ParaView; use "Glyph" filter with sphere glyph scaled by the `radius` scalar.

## JSON scene format

```json
{
    "dt":         0.00005,
    "gravity":    [0.0, -9.81, 0.0],
    "domain":     { "min": [0,0,0], "max": [10,10,0] },
    "cell_size":  1.0,
    "particles": [
        { "position": [x,y,0], "velocity": [vx,vy,0],
          "radius": r, "mass": m, "kn": k, "gamma_n": gn, "gamma_t": gt, "mu": mu }
    ]
}
```

## Architecture notes

- **Owned vs halo layout**: `d_positions[0..n)` = owned (read-write); `d_positions[n..n_total)` = halo (read-only after exchange). Both are passed to `assignCell` so the cell grid contains all particles. Only owned particles are force-computed and integrated.
- **Capacity**: each GPU pre-allocates `2 * total_N` slots so the worst-case migration (all particles on one GPU) fits without reallocation.
- **N-GPU halo exchange**: each GPU collects left-boundary and right-boundary strips independently, then uploads strips received from its left and right neighbors as halo. Edge GPUs exchange on one side only.
- **N-GPU migration**: downloads all owned particles from all GPUs, merges, re-splits by x-coordinate across N domain slices, and re-uploads.
- **Material properties**: force kernel uses particle `i`'s `kn`, `gamma_n`, `gamma_t`, `mu` for both sides of a contact — valid for uniform materials. For mixed materials, use effective (harmonic mean) values.
- **Neighborhood table** (one `d_nb` per GPU): built once at startup per domain and never changes. Only `d_cellHeads` is reset each step.
- **Migration cost**: current implementation downloads all owned particles from all GPUs every time any particle crosses a boundary. This is the dominant cost for high-migration scenarios.

---

## Naming conventions

### Files
| Category | Convention | Examples |
|---|---|---|
| Header files (host-only) | `snake_case.h` | `Domain.h`, `input.h`, `vtk_output.h` |
| Header files (device/host) | `snake_case.cuh` | `Vec3.cuh`, `force_kernels.cuh` |
| Source files | `snake_case.cu` | `main.cu` |
| Python scripts | `snake_case.py` | `gen_lattice.py` |
| Scene files | `snake_case.json` | `two_discs.json` |

### Code (C++ / CUDA)
| Category | Convention | Examples |
|---|---|---|
| Types (structs) | `PascalCase` | `Domain`, `ParticleDevice`, `ParticleHost`, `SceneConfig` |
| Constants / macros | `SCREAMING_SNAKE_CASE` | `BLOCK` (dim3 constant) |
| Functions (free) | `camelCase` | `buildDomains()`, `computeCellIndex()`, `exchangeHalos()` |
| Functions (methods) | `camelCase` | `upload()`, `download()`, `removeAt()` |
| Variables (local/param) | `snake_case` | `max_steps`, `cell_size`, `num_gpus` |
| Members (struct fields) | `snake_case` | `d_positions`, `num_cells`, `halo_width` |
| Device pointers prefix | `d_` | `d_positions`, `d_forces`, `d_cellHeads` |
| Host vectors prefix | (none) or `h_` in halo exchange | `positions`, `h_pos` |

### Inconsistencies to fix
- `init_neighborhood.h` uses `lowerCamelCase` and `snake_case` mixed for function names (e.g. `initCellNeighborhood` vs `get_cell_index_for_periodic_boundary`). Prefer consistent `camelCase` for functions.
- `assign_cells.cuh` has include guard `NEIGHBORHOOD_CUH` but the file is `assign_cells.cuh`. Should be `ASSIGN_CELLS_CUH`.
- `Vec3.cuh` uses `__host__`-only JSON function (`Vec3FromJson`) inside a `.cuh` header; move to a host-only utility or `input.h`.
- `computeCellIndex` parameter `numOfCellsPerAxis` uses PascalCase parameter name — inconsistent with the rest of the code (prefer `snake_case` params).
- Variable naming conflict: `computeCellIndex()` parameter `cellL` collides in style with `cell_size` used elsewhere. Pick one.

---

## NHR@FAU TinyGPU cluster (tinyx)

### Overview
- **Frontend node**: `tinyx.nhr.fau.de`
- **Documentation**: https://doc.nhr.fau.de/clusters/tinygpu/
- **Access**: Tier-3 Grundversorgung accounts only (NOT NHR project accounts)
- **OS**: Ubuntu 20.04 LTS
- **Software**: Environment modules (Spack-based), Apptainer for containers

### Node types

| Partition | Nodes | GPUs per node | GPU type | GPU memory | CPU | Host RAM |
|---|---|---|---|---|---|---|
| `work` (default) | 8 | 4× | RTX 2080 Ti | 11 GB | Intel Xeon Gold 6134 (2×16c) | 96 GB |
| `work` (default) | 7 | 8× | RTX 3080 | 10 GB | Intel Xeon Gold 6226R (2×32c) | 384 GB |
| `v100` | 4 | 4× | Tesla V100 | 32 GB | Intel Xeon Gold 6134 (2×16c) | 96 GB |
| `a100` | 8 | 4× | A100 SXM4 (NVLink) | 40 GB | AMD EPYC 7662 (128c) | 512 GB |

### GPU compute capabilities (for NVCC `-gencode`)

| GPU | Compute Capability | NVCC flags |
|---|---|---|
| A100 | 8.0 | `-gencode arch=compute_80,code=sm_80` |
| RTX 3080 | 8.6 | `-gencode arch=compute_86,code=sm_86` |
| RTX 2080 Ti | 7.5 | `-gencode arch=compute_75,code=sm_75` |
| V100 | 7.0 | `-gencode arch=compute_70,code=sm_70` |

Build a multi-architecture binary for TinyGPU:
```bash
# Add to CMakeLists.txt:
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86")
```

### CPU optimization flags (for host code)

| Partition | Microarchitecture | GCC/LLVM flags |
|---|---|---|
| All (safe) | x86-64-v3 | `-mavx2 -mfma` or `-march=x86-64-v3` |
| `work` | Skylake / Cascade Lake | `-march=skylake-avx512` |
| `rtx3080` | Cascade Lake | `-march=cascadelake` |
| `v100` | Skylake | `-march=skylake-avx512` |
| `a100` | Zen2 | `-march=znver2` |

**Important**: Host code compiled with Intel-specific flags (`-march=skylake-avx512`) will NOT run on the A100 partition (AMD EPYC). Use `-mavx2 -mfma` for portable binaries.

### Slurm commands (use `.tinygpu` suffix)

```bash
# Interactive job (1 GPU, 1 hour)
salloc.tinygpu --gres=gpu:1 --time=01:00:00

# Batch job submission
sbatch.tinygpu job_script.sh

# Queue status
squeue.tinygpu

# Node info
sinfo.tinygpu
```

### Batch job script template for md2d (2 GPUs, A100 partition)

```bash
#!/bin/bash -l
#SBATCH --gres=gpu:a100:2
#SBATCH --partition=a100
#SBATCH --time=6:00:00
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV

# Load CUDA module if needed
module load cuda

# Run with the scene file passed as argument
./build/md2d scenes/lattice.json 50000
```

### Requesting specific GPU types
```bash
# Generic GPU (any available type)
--gres=gpu:2

# Specific GPU type
--gres=gpu:a100:2
--gres=gpu:rtx3080:2
--gres=gpu:rtx2080ti:2
--gres=gpu:v100:2
```

### Key constraints
- Max walltime: 24 hours (all partitions)
- Max interactive: 4 hours
- CPUs and host RAM are allocated automatically proportional to GPU count
- Nodes use `$TMPDIR` (local SSD, 1.8–5.8 TB) for fast scratch storage — deleted after job ends
- `$HOME`, `$HPCVAULT`, `$WORK` filesystems available on all nodes
- P2P / NVLink available on A100 nodes — ideal for `cudaMemcpyPeer` optimization

---

## Known issues / TODOs

- `migration.h`: full CPU round-trip on every migration event. Replace with in-GPU compaction (thrust or hand-written prefix-sum) and `cudaMemcpyPeer`.
- `halo_exchange.h`: CPU-mediated. Replace `collectHalo` download + `uploadHalo` with `cudaMemcpyPeer` for direct GPU-to-GPU transfer.
- `force_kernels.cuh`: uses particle `i`'s material properties only. Implement harmonic-mean effective `kn` and `gamma_n` for multi-material simulations.
- No periodic boundary conditions in y. Currently reflective walls only.
- No energy / momentum diagnostics. Add a reduction kernel to monitor conservation.
- `main.cu`: the neighborhood table (`d_nb0/d_nb1`) is rebuilt after migration even though the cell grid structure never changes. Cache it.

---

## Potential errors & code issues

### Critical / High severity
1. **No CUDA error checking anywhere.** After every `cudaMalloc`, `cudaMemcpy`, and kernel launch, there is no `cudaGetLastError()` or `cudaDeviceSynchronize()` + error check. A silent failure on one GPU will produce garbage results with no diagnostic. Add `CHECK_CUDA(err)` macro after all CUDA API calls and kernel invocations.

2. **No peer access enabled.** The code never calls `cudaDeviceEnablePeerAccess()`. On A100 nodes (NVLink-connected), this is a missed optimization and could cause correctness issues if `cudaMemcpyPeer` is later introduced without enabling peer access first. Add peer access enable/disable at startup/cleanup.

3. **`main.cu` — no error check on `cudaSetDevice`.** If device 0 or 1 is unavailable or in prohibited mode, subsequent operations silently operate on the wrong device or fail.

### Medium severity
4. **`assign_cells.cuh` include guard mismatch.** The guard uses `NEIGHBORHOOD_CUH` but the file is `assign_cells.cuh`. This is misleading and could cause subtle issues if a file named `neighborhood.cuh` is added later.

5. **`Vec3.cuh` includes `json.hpp`.** A math utility header pulls in the entire nlohmann/json library (~25k lines) for a single host-only function (`Vec3FromJson`). This increases compile times for every translation unit that includes `Vec3.cuh`. Move `Vec3FromJson` to `input.h` or a dedicated utility header.

6. **`Vec3::ceil()` uses `std::ceil`** which may not be available in device code on all CUDA toolkit versions (though it is supported since CUDA 10+). Use plain `ceilf()` for maximum portability.

7. **`computeContactForces` comment says `d_gamma_t`/`d_mu` are "size n/2"** but they are actually allocated as `capacity/2 = total_n` which is >= `n`. The comment is misleading — the actual allocation is correct but could confuse maintainers.

### Low severity
8. **`assign_cells.cuh` includes `<cstdio>`** unnecessarily. Remove dead include.

9. **Inconsistent naming in `init_neighborhood.h`**: function `initCellNeighborhood` (camelCase) calls `get_cell_index_for_fixed_boundary` (snake_case). Standardize.

10. **`integration.cuh` `reflectWalls` function**: the reflective wall correction only checks against global domain boundaries. If a particle on GPU 0 is pushed past `split_x` by wall reflection, it will be caught by migration — correct, but the two-step reflection + migration is slightly less efficient than handling the wall at the halo boundary.

11. **`halo_exchange.h` `collectHalo`**: downloads ALL owned particles to CPU just to filter a subset. This transfers unnecessary data (all velocities, masses, radii, kn, gamma_n). A GPU-side filter kernel writing to a contiguous output buffer would be significantly faster, even without peer-to-peer transfer.

12. **`gen_lattice.py` domain size**: computes domain_max from particle positions + spacing, but the generated domain may not be an even multiple of cell_size, leaving a partial cell at the boundary. This is handled by `computeCellIndex` clamping but may waste memory.

---

## Tasks

- [ ] **Validate 2D, 2-GPU**: run `two_discs.json`, verify the two particles collide correctly, bounce, and migrate between GPUs without losing energy unexpectedly. Compare against a single-GPU reference if possible.

- [ ] **Lattice test**: generate a 20×20 hex lattice with `gen_lattice.py`, run for 10 000 steps, check VTK output in ParaView.

- [ ] **Add CUDA error checking**: wrap all CUDA API calls and kernel launches with a `CHECK_CUDA` macro. This is the single most impactful reliability improvement.

- [ ] **Periodic BCs in y**: replace the reflective y-walls with periodic boundaries. Requires wrapping positions and adjusting `computeCellIndex` (use `get_cell_index_for_periodic_boundary` already in `init_neighborhood.h`).

- [ ] **Optimize migration**: replace `migrateParticles()` full round-trip with GPU-side stream compaction + `cudaMemcpyPeer`. Target: migration cost < halo exchange cost.

- [ ] **Optimize halo exchange**: replace `collectHalo` CPU download with `cudaMemcpyPeer` (direct GPU-to-GPU). Enable peer access at startup with `cudaDeviceEnablePeerAccess`.

- [ ] **Extend to 3D**: remove `z=0` constraint in `integration.cuh`, set `num_cells.z > 1` in `buildDomains()`, update JSON scenes. No other file changes required.

- [x] **Extend to N GPUs**: generalize `buildDomains()` to split the X axis into N equal slices, one per GPU. Each interior GPU then has two halo neighbors (left and right); adjust `exchangeHalos()` accordingly.

- [ ] **Energy diagnostics**: add a device reduction (e.g., thrust::transform_reduce) to compute total kinetic energy each frame. Print it alongside the step count.

- [ ] **Fix include guard in `assign_cells.cuh`**: rename `NEIGHBORHOOD_CUH` to `ASSIGN_CELLS_CUH`.

- [ ] **Decouple `Vec3.cuh` from `json.hpp`**: move `Vec3FromJson` to a host-only header or `input.h`.

- [ ] **Standardize naming conventions**: rename functions in `init_neighborhood.h` to camelCase; fix `numOfCellsPerAxis` parameter casing in `computeCellIndex`.
