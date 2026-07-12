# md2d — GPU-accelerated 2D/3D molecular dynamics simulator

A spring-dashpot DEM (Discrete Element Method) simulator with **N-GPU domain
decomposition**.

In 3D mode (`make md3d`, `-DMD3D`), the domain is split into a **3D nx×ny×nz grid**
(factored from N GPUs). Each GPU has up to 6 neighbors (±X, ±Y, ±Z) and exchanges
halo particles in all directions. 2D mode uses X-only split.

## Build

```bash
make                     # 2D binary (z=0 enforced, X-only split)
make md3d                # 3D binary (full 3D, 2D grid decomposition)
```

On the TinyGPU cluster:
```bash
module load gcc/11.5.0 cuda/12.8.0
make
```

## Run

```bash
./md2d scenes/random20.json 50000 2     # scene, max steps, num GPUs
./md3d scenes/cube8.json   5000  4     # 3D, auto-decomposes into 2×2 grid
```

Usage: `./md2d <scene.json> [max_steps] [num_gpus] [vtk_interval]`
- `max_steps` defaults to 100000
- `num_gpus` defaults to all available CUDA devices
- `vtk_interval` defaults to 20 (write VTK every N steps; lower = more collision detail)

## Algorithm

### Per-step loop

1. **Halo exchange** — each GPU packs boundary strips via a GPU-side
   filter kernel (atomic-add compaction into contiguous buffers),
   then exchanges directly via `cudaMemcpyPeer`. Only 4-byte
   particle counts cross the CPU bus. Up to 6 directions in 3D.
2. **Cell assignment** — all particles (owned + halo) are inserted into a
   uniform cell grid via atomic linked-list prepend. 27-neighbor lookup table
   is precomputed at startup.
3. **Force computation** — spring-dashpot contact forces (normal spring +
   dashpot, tangential viscous + Coulomb friction) computed for owned
   particles only, against all particles in neighboring cells (including
   halo). Material properties use harmonic mean for mixed contacts.
4. **Integration** — symplectic Euler: velocity update `v += dt × F/m`,
   position update `x += dt × v`. Reflective walls on all axes. 2D mode
   clamps z=0.
5. **Particle migration** — if any particle crosses its owning region,
   all particles are downloaded, merged, re-split across the GPU grid
   by (x, y, z) position, and re-uploaded.

### Domain decomposition

**3D mode** (`make md3d`): N GPUs are factored into a near-cube **nx×ny×nz grid**.
Each GPU owns a 3D sub-volume.

| GPUs | Grid | Per-GPU neighbors |
|---|---|---|
| 1 | 1×1×1 | none |
| 2 | 2×1×1 | ±X |
| 3 | 3×1×1 | ±X |
| 4 | 2×2×1 | ±X, ±Y |
| 5 | 5×1×1 | ±X |
| 6 | 3×2×1 | ±X, ±Y |
| 7 | 7×1×1 | ±X |
| 8 | 2×2×2 | ±X, ±Y, ±Z |

Prime GPU counts fall back to X-only split. Composite counts use 2D or
full 3D decomposition for best load balance.

**2D mode** (`make`): X-only split into N equal slices.

- Halo width = `2 × r_max` — keeps a margin wide enough that any
  cross-boundary contact is captured.
- Edge GPUs have no halo padding on the domain-wall side.

### Memory layout

- `d_positions[0..n)` = owned particles (read-write, integrated).
- `d_positions[n..n_total)` = halo particles (read-only after exchange,
  used only for contact detection).
- Capacity = `2 × total_N` — worst-case scenario where all particles
  migrate to a single GPU.
- Forces, gamma_t, mu are owned-only (sized `capacity/2`).

## Features

- **Cell-based neighbor search** — O(N) per step with 27-neighbor lookup
- **GPU-direct halo exchange** — GPU-side particle packing + `cudaMemcpyPeer`;
  no CPU staging of particle data in the hot path
- **Persistent particle IDs** — assigned at load, survive migration,
  enable stable ParaView animation without flicker
- **Energy diagnostics** — kinetic energy and momentum (x, y, z)
  accumulated across all GPUs via parallel reduction, printed per frame
- **P2P peer access** — enabled between all GPU pairs at startup

## Scene format

```json
{
    "dt":         0.0002,
    "gravity":    [0.0, 0.0, 0.0],
    "domain":     { "min": [0,0,0], "max": [10,10,8] },
    "cell_size":  0.6,
    "particles": [
        { "position": [x,y,z], "velocity": [vx,vy,vz],
          "radius": r, "mass": m, "kn": 5000,
          "gamma_n": 20, "gamma_t": 10, "mu": 0.3 }
    ]
}
```

Generate scenes with the bundled scripts:

```bash
python3 scripts/gen_random.py   > scenes/my_2d.json    # 2D random
python3 scripts/gen_lattice.py  > scenes/lattice.json  # 2D hexagonal lattice
python3 scripts/gen_random3d.py > scenes/my_3d.json    # 3D random
```

## Visualization

VTK output lands in `output/<scene>_<steps>/`. Each frame is one file;
load them all as a ParaView time series.

1. Download `output/` to your local machine.
2. `File → Open` → select all `.vtk` files. Include `domain_boundary.vtk`
   for the domain wireframe, GPU split lines, and halo volume boxes.
3. Click `Apply`.
4. `Filters → Glyph`, set **Glyph Type** to `Sphere`, scale by `radius`.
   In Glyph properties: **Masking → Glyph Mode** → `All Points`.
5. Color particles:
   - `gpu_owner` — discrete per-GPU coloring
   - `border` — 0→1 gradient showing halo proximity at GPU boundaries
   - `velocity` — vector field for arrow/direction coloring
6. For `domain_boundary.vtk`: set **Coloring** to `region_type`
   (0=boundary, 1=split, 2=halo), reduce **Opacity** to ~0.3 to see
   particles through the halo volumes.
7. Click `Play` to animate.

### File map

| File | Role |
|---|---|
| `main.cu` | Entry point; simulation loop |
| `domain.h` | `Domain` struct + `buildDomains()` for N-GPU 3D grid decomposition |
| `particle_device.cuh` | GPU pointer struct (owned + halo layout) |
| `particle_host.h` | CPU buffer; `upload()` / `download()` / `freeParticleDevice()` |
| `halo_exchange.h` | `packStrip()` / `exchangeHalos()` — GPU-side packing + `cudaMemcpyPeer`, N-GPU ±X±Y±Z |
| `pack_halo.cuh` | `packHaloParticles` kernel — GPU-side filter/compaction into contiguous buffers |
| `migration.h` | `migrateParticles()` — full CPU round-trip redistribution, XYZ grid |
| `force_kernels.cuh` | `computeContactForces` kernel (spring-dashpot DEM) |
| `integration.cuh` | `integrate` kernel (symplectic Euler, reflective walls) |
| `assign_cells.cuh` | `assignCell` kernel + `computeCellIndex` |
| `init_neighborhood.h` | `initCellNeighborhood()` — CPU-side 27-neighbor table |
| `cells.cuh` | `Cells` helper struct |
| `vec3.cuh` | `Vec3` math type + free functions |
| `vtk_output.h` | `writeParticlesVTK()` + `writeDomainBoundaryVTK()` |
| `input.h` | `loadScene()` / `splitAt()` / `splitIntoN()` — JSON parsing |
| `json.hpp` | nlohmann/json single-header |
| `check_cuda.h` | `CHECK_CUDA` / `CHECK_LAST_CUDA` macros |
| `energy_diagnostics.cuh` | `computeEnergyAndMomentum` kernel + helper |
| `Makefile` | Build system (`make` / `make md3d` / `make clean`) |
| `scripts/gen_lattice.py` | 2D hexagonal lattice scene generator |
| `scripts/gen_random.py` | 2D random particle scene generator |
| `scripts/gen_random3d.py` | 3D random particle scene generator |
| `scripts/sbatch_a100.sh` | Slurm batch script for A100 partition |
| `scripts/sbatch_work.sh` | Slurm batch script for work partition |
