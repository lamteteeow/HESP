# Agent Context — ss26-project (md2d)

## What this project is

A GPU-accelerated 2D molecular dynamics / DEM simulator with **N-GPU domain decomposition**.

The simulation domain is split evenly along the X axis into N slices, one per GPU.
Each GPU owns particles in its slice and exchanges halo particles with its left
and right neighbors. Interior GPUs have two halo neighbors; edge GPUs have one.

In 3D mode (`make md3d`, `-DMD3D`), the domain is split into a **2D nx×ny grid**
(factored from N GPUs). Each GPU has up to 4 neighbors (L/R/B/T). 2D mode uses
X-only split.

At each step:
1. **Halo exchange** — each GPU sends boundary strips (X and Y) to neighbors and receives their strips as ghost/read-only particles.
2. **Cell assignment** — all particles (owned + halo) are inserted into a uniform cell grid local to each GPU.
3. **Force computation** — spring-dashpot (DEM) contact forces, computed only for owned particles.
4. **Integration** — symplectic Euler, z=0 constraint enforced in 2D, reflective wall BCs on all axes.
5. **Particle migration** — particles that crossed their owning region are redistributed across the GPU grid.
6. **VTK output** — every `steps_per_frame` steps, all GPUs dump to a single VTK file with `gpu_owner`, `border`, and domain decomposition visuals.

For 2D, `num_cells.z = 1` and all z-coordinates are clamped to 0. For 3D, compile with `-DMD3D` (via `make md3d`) to remove the z-constraint and auto-decompose into a 2D grid.

## File map

| File | Role |
|---|---|
| `main.cu` | Entry point; simulation loop |
| `domain.h` | `Domain` struct + `buildDomains()` for N-GPU X-split |
| `particle_device.cuh` | GPU pointer struct (owned + halo layout) |
| `particle_host.h` | CPU buffer; `upload()` / `download()` / `freeParticleDevice()` |
| `halo_exchange.h` | `collectHalo()` / `uploadHalo()` / `exchangeHalos()` — N-GPU |
| `migration.h` | `migrateParticles()` — full CPU round-trip redistribution, N-GPU |
| `force_kernels.cuh` | `computeContactForces` kernel (spring-dashpot DEM) |
| `integration.cuh` | `integrate` kernel (symplectic Euler, reflective walls) |
| `assign_cells.cuh` | `assignCell` kernel + `computeCellIndex` (from mini-project) |
| `init_neighborhood.h` | `initCellNeighborhood()` — CPU-side 27-neighbor table (from mini-project) |
| `cells.cuh` | `Cells` helper struct (from mini-project) |
| `vec3.cuh` | `Vec3` math type + free functions (from mini-project) |
| `vtk_output.h` | `writeParticlesVTK()` — positions, velocities, radii |
| `input.h` | `loadScene()` / `splitAt()` / `splitIntoN()` — JSON parsing and particle distribution |
| `json.hpp` | nlohmann/json single-header (from mini-project) |
| `check_cuda.h` | `CHECK_CUDA` / `CHECK_LAST_CUDA` macros — CUDA error checking |
| `energy_diagnostics.cuh` | `computeEnergyAndMomentum` kernel + `computeDiagnostics` helper — reduction for KE and momentum |
| `scenes/two_discs.json` | Two particles colliding at the domain boundary |
| `scripts/gen_lattice.py` | Python script: generates a 2D hexagonal lattice JSON scene |
| `scripts/gen_random.py` | Python script: generates a 2D random particle scene |
| `scripts/gen_random3d.py` | Python script: generates a 3D random particle scene |
| `scripts/sbatch_a100.sh` | Slurm batch script for A100 partition (2 GPUs) |
| `scripts/sbatch_work.sh` | Slurm batch script for work partition (1 GPU) |
| `Makefile` | Build system (`make` / `make clean`) |
| `README.md` | Project overview and quick-start |

## Build

### Local / workstation (CUDA already in PATH)

```bash
make
```

### TinyGPU cluster

The cluster uses Lmod environment modules. CUDA is **not** available on the
login node (`tinyx`) — build on a compute node, or from a job script.

```bash
# Get a compute node first
salloc.tinygpu --gres=gpu:1 --time=01:00:00

# Load GCC first (CUDA 12.8 requires host compiler ≥ GCC 10)
module load gcc/11.5.0
module load cuda/12.8.0

make
```

**Recommended module versions:**

| Module | Version | Why |
|---|---|---|
| `cuda` | `12.8.0` | Latest toolkit — best nvcc optimizer, full sm_70–sm_86 support |
| `gcc` | `11.5.0` | CUDA 12.8 requires host compiler ≥ GCC 10; 11.5 is the most compatible version on TinyGPU. **GCC 14 is not supported by CUDA 12.8.** |

The project uses only core CUDA Runtime API calls (`cudaMalloc`, `cudaMemcpy`,
`cudaSetDevice`, `cudaDeviceEnablePeerAccess`, etc.) — no cuBLAS, cuFFT, CUB,
or Thrust. Any CUDA ≥ 11.0 would work, but 12.8.0 produces the best codegen
for A100 (sm_80) and RTX 3080 (sm_86).

Binary: `./md2d`

## Run

```bash
# Uses all available GPUs by default
./md2d scenes/two_discs.json 10000

# Explicitly request N GPUs (3rd argument)
./md2d scenes/lattice.json 50000 4

# Generate a lattice scene and run it
python3 scripts/gen_lattice.py > scenes/lattice.json
./md2d scenes/lattice.json 50000

# Generate a random scene and run it
python3 scripts/gen_random.py > scenes/random.json
./md2d scenes/random.json 50000
```

Usage: `./md2d <scene.json> [max_steps] [num_gpus]`
- `max_steps` defaults to 100000
- `num_gpus` defaults to all available CUDA devices

Output goes to `output/<scene>_<steps>/`. Open in ParaView; use "Glyph" filter with sphere glyph scaled by the `radius` scalar.

### Visualization

1. Download the `output/` directory to your local machine.
2. Open ParaView, `File → Open` → select all VTK files in the output directory
   (they load as a time series). Include `domain_boundary.vtk` to see the domain
   wireframe.
3. Click `Apply`.
4. Add a `Glyph` filter, set glyph type to `Sphere`, scale by the `radius` scalar.
   In the Glyph properties, set **Masking → Glyph Mode** to `All Points`
   to avoid radius interpolation artifacts.
5. Color particles: in **Properties → Coloring** select `gpu_owner` to see
   which GPU owns each particle, or `border` for the halo-overlap gradient.
6. For the domain boundary: select `domain_boundary.vtk` in the pipeline,
   set **Coloring** to `region_type` (0=boundary, 1=split, 2=halo), and
   reduce **Opacity** to ~0.3 to see particles through the halo volumes.
7. Click `Play` to animate.

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
- **N-GPU halo exchange**: each GPU collects boundary strips (X and Y) independently, then uploads strips received from neighbors as halo. Edge GPUs exchange on one side only. In 3D mode, exchanges in up to 4 directions.
- **N-GPU migration**: downloads all owned particles from all GPUs, merges, re-splits across the 2D grid (X and Y), and re-uploads.
- **Material properties**: force kernel uses particle `i`'s `kn`, `gamma_n`, `gamma_t`, `mu` for both sides of a contact — valid for uniform materials. For mixed materials, use effective (harmonic mean) values.
- **Neighborhood table** (one `d_nb` per GPU): built once at startup per domain and never changes. Only `d_cellHeads` is reset each step.
- **Migration cost**: current implementation downloads all owned particles from all GPUs every time any particle crosses a boundary. This is the dominant cost for high-migration scenarios.

---

## Naming conventions

### Files
| Category | Convention | Examples |
|---|---|---|
| Header files (host-only) | `snake_case.h` | `domain.h`, `input.h`, `vtk_output.h` |
| Header files (device/host) | `snake_case.cuh` | `vec3.cuh`, `force_kernels.cuh` |
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
- `vec3.cuh` includes `json.hpp` (~25k lines) for a single `__host__`-only function (`vec3FromJson`). This increases compile times for every translation unit that includes `vec3.cuh`. Move `vec3FromJson` to a host-only header or `input.h`.

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

Multi-architecture support is already configured in the `Makefile` (all four archs
are compiled into a single fat binary). No extra flags needed.

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

### Batch job scripts

Pre-built Slurm scripts are provided in `scripts/`:

| Script | Partition | Default GPUs | Use case |
|---|---|---|---|
| `scripts/sbatch_a100.sh` | `a100` | 2× A100 | NVLink, best multi-GPU perf |
| `scripts/sbatch_work.sh` | `work` | 1× (RTX 2080 Ti / 3080) | Quick tests, single GPU |

Usage:
```bash
# A100 partition, 2 GPUs, default scene
sbatch.tinygpu scripts/sbatch_a100.sh

# Custom scene, steps, and GPU count (override --gres on command line)
sbatch.tinygpu --gres=gpu:a100:4 scripts/sbatch_a100.sh scenes/random20.json 100000 4

# Work partition, 1 GPU
sbatch.tinygpu scripts/sbatch_work.sh scenes/two_discs.json 10000
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

- `halo_exchange.h` + `migration.h`: Both use full CPU round-trips for particle data movement — this is the dominant bottleneck. **Particle packing** (GPU-side filtering via stream compaction into contiguous output buffers) and **particle unpacking** (receiving directly via `cudaMemcpyPeer` instead of CPU staging) are the two key techniques needed. See the dedicated task below.
- No periodic boundary conditions in y. Currently reflective walls only.
- `vec3::ceil()` uses `std::ceil` on older builds — already migrated to `ceilf()` for portable device code.

---

## Potential errors & code issues

### Critical / High severity
1. ~~**No CUDA error checking applied.**~~ Fixed: `check_cuda.h` macros now wrap all CUDA API calls and kernel launches in `main.cu`, `particle_host.h`, and `energy_diagnostics.cuh`.

2. ~~**No peer access enabled.**~~ Fixed: `cudaDeviceEnablePeerAccess()` called for all GPU pairs at startup.

3. ~~**`main.cu` — no error check on `cudaSetDevice`.**~~ Fixed: all `cudaSetDevice` calls wrapped with `CHECK_CUDA`.

### Medium severity
4. ~~**`vec3.cuh` includes `json.hpp`.**~~ Fixed: `vec3FromJson` moved to `input.h`; `json.hpp` include removed from `vec3.cuh`.

5. ~~**`vec3::ceil()` uses `std::ceil`**~~ Fixed: uses `ceilf()` for portable device code.

7. ~~**`computeContactForces` comment says `d_gamma_t`/`d_mu` are "size n/2"**~~ Fixed: comment now says "capacity/2" to match actual allocation.

### Low severity
8. ~~**`assign_cells.cuh` includes `<cstdio>`**~~ Fixed: dead include removed.

9. ~~**Inconsistent naming in `init_neighborhood.h`**~~ Fixed: all functions now use camelCase consistently.

10. **`integration.cuh` `reflectWalls` function**: the reflective wall correction only checks against global domain boundaries. If a particle on GPU 0 is pushed past `split_x` by wall reflection, it will be caught by migration — correct, but the two-step reflection + migration is slightly less efficient than handling the wall at the halo boundary.

11. **`halo_exchange.h` `collectHalo`**: downloads ALL owned particles to CPU just to filter a subset. This transfers unnecessary data (all velocities, masses, radii, kn, gamma_n). A GPU-side filter kernel writing to a contiguous output buffer would be significantly faster, even without peer-to-peer transfer.

12. **`gen_lattice.py` domain size**: computes domain_max from particle positions + spacing, but the generated domain may not be an even multiple of cell_size, leaving a partial cell at the boundary. This is handled by `computeCellIndex` clamping but may waste memory.

---

## Tasks

- [ ] **Validate 2D, 2-GPU**: run `two_discs.json`, verify the two particles collide correctly, bounce, and migrate between GPUs without losing energy unexpectedly. Compare against a single-GPU reference if possible.

- [ ] **Lattice test**: generate a 20×20 hex lattice with `gen_lattice.py`, run for 10 000 steps, check VTK output in ParaView.

- [x] **Integrate CUDA error checking**: `check_cuda.h` provides the macros — applied `CHECK_CUDA` / `CHECK_LAST_CUDA` to all CUDA API calls and kernel launches in `main.cu`, `particle_host.h`, and `energy_diagnostics.cuh`.

- [ ] **Periodic BCs in y**: replace the reflective y-walls with periodic boundaries. Requires wrapping positions and adjusting `computeCellIndex` (use `get_cell_index_for_periodic_boundary` already in `init_neighborhood.h`).

- [ ] **Particle packing/unpacking for GPU-to-GPU transfer** — this is the single highest-impact optimization for multi-GPU performance. The current `collectHalo` + `uploadHalo` and `migrateParticles` both round-trip all particle data through the CPU, which dominates runtime. Replace them with:
    1. **Particle packing** (on source GPU): a GPU-side filter kernel that writes boundary/crossed particles into a compact contiguous output buffer (use a parallel prefix-sum / scan to compute output offsets). Only the packed buffer is copied to the target GPU via `cudaMemcpyPeer` — no CPU involvement.
    2. **Particle unpacking** (on target GPU): receive the packed buffer directly into the halo/migration slot. For halo exchange, append packed strips from left+right neighbors after owned particles. For migration, compact owned particles that stay in-place first, then append migrated-in particles.
    3. **Enable peer access** at startup with `cudaDeviceEnablePeerAccess()`, especially critical on A100 nodes where NVLink makes `cudaMemcpyPeer` nearly free.
  Reference techniques: GPU stream compaction (CUB/thrust `copy_if` or hand-rolled prefix-sum), `cudaMemcpyPeerAsync` for overlap, double-buffering for pipelining. Target: eliminate all CPU-mediated particle data movement from the hot path.

- [ ] **Optimize migration**: replace `migrateParticles()` full round-trip with GPU-side stream compaction + `cudaMemcpyPeer`. Target: migration cost < halo exchange cost.

- [ ] **Optimize halo exchange**: replace `collectHalo` CPU download with `cudaMemcpyPeer` (direct GPU-to-GPU). Enable peer access at startup with `cudaDeviceEnablePeerAccess`.

- [ ] **Extend to 3D**: remove `z=0` constraint in `integration.cuh`, set `num_cells.z > 1` in `buildDomains()`, update JSON scenes. No other file changes required.

- [x] **Extend to N GPUs**: generalize `buildDomains()` to split the X axis into N equal slices, one per GPU. Each interior GPU then has two halo neighbors (left and right); adjust `exchangeHalos()` accordingly.

- [x] **Energy diagnostics**: `energy_diagnostics.cuh` provides the reduction kernel and helper. Wired into the simulation loop — KE and momentum are printed each frame.

- [x] **Fix include guard in `assign_cells.cuh`**: rename `NEIGHBORHOOD_CUH` to `ASSIGN_CELLS_CUH`.

- [x] **Decouple `Vec3.cuh` from `json.hpp`**: move `Vec3FromJson` to a host-only header or `input.h`.

- [x] **Standardize naming conventions**: rename functions in `init_neighborhood.h` to camelCase; fix `numOfCellsPerAxis` parameter casing in `computeCellIndex`.
