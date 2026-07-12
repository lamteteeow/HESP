# md3d — Multi-GPU 3D DEM simulator

Spring-dashpot DEM (Discrete Element Method) with **nx×ny×nz grid decomposition**
(factored from N GPUs). Each GPU exchanges halo particles in up to 6 directions
via GPU-side packing + `cudaMemcpyPeer`.

## Build

```bash
make              # 3D binary (default)
make md2d         # 2D binary (z=0 enforced, X-only split)
```

### TinyGPU cluster (NHR@FAU)

```bash
# Get a compute node
salloc.tinygpu --gres=gpu:1 --time=01:00:00

# Load modules and build
module load cuda/12.8.0
make
```

Batch submission:
```bash
# A100 (4 GPUs, NVLink)
sbatch.tinygpu scripts/sbatch_a100.sh scenes/cube256.json 50000 4

# RTX 3080 (8 GPUs)
sbatch.tinygpu --gres=gpu:rtx3080:8 scripts/sbatch_work.sh scenes/cube256.json 50000 8

# Quick test (1 GPU)
sbatch.tinygpu scripts/sbatch_work.sh scenes/cube8.json 5000
```

## Run

```bash
./md3d scenes/cube256.json 5000 8 5   # scene, max steps, GPUs, VTK interval
```

## Algorithm

Per step: **halo exchange** (GPU-side packing → `cudaMemcpyPeer`) →
**cell assignment** (27-neighbor grid) → **force computation**
(spring-dashpot DEM) → **integration** (symplectic Euler, reflective walls) →
**migration** (redistribution across GPU grid).

### Domain decomposition

N GPUs factored into near-cube nx×ny×nz grid:

| GPUs | Grid | Neighbors |
|---|---|---|
| 1 | 1×1×1 | — |
| 2 | 2×1×1 | ±X |
| 4 | 2×2×1 | ±X, ±Y |
| 6 | 3×2×1 | ±X, ±Y |
| 8 | 2×2×2 | ±X, ±Y, ±Z |

Halo width = `2×r_max`. Edge GPUs have no halo on the domain-wall side.

## Scene format

```json
{
    "dt": 0.0002, "gravity": [0,0,0], "cell_size": 1.0,
    "domain": { "min": [0,0,0], "max": [16,16,16] },
    "particles": [
        { "position": [x,y,z], "velocity": [vx,vy,vz],
          "radius": r, "mass": m, "kn": 1000,
          "gamma_n": 5, "gamma_t": 2.5, "mu": 0.3 }
    ]
}
```

Generate scenes:

```bash
python3 scripts/gen_random3d.py > scenes/my_3d.json
python3 scripts/gen_random.py   > scenes/my_2d.json
```

## ParaView

1. Download `output/` to your machine
2. `File → Open` → all `.vtk` files + `domain_boundary.vtk`
3. `Glyph` filter → Sphere, scale by `radius`, **Masking → All Points**
4. Color by `gpu_owner` (per-GPU) or `border` (halo gradient)
5. `domain_boundary.vtk`: Coloring → `region_type`, Opacity ~0.3

## Key files

| File | Role |
|---|---|
| `main.cu` | Entry point, simulation loop |
| `domain.h` | 3D grid decomposition |
| `halo_exchange.h` | GPU-side packing + `cudaMemcpyPeer` |
| `pack_halo.cuh` | Pack kernel (atomic-add compaction) |
| `force_kernels.cuh` | DEM contact forces |
| `integration.cuh` | Symplectic Euler, reflective walls |
| `migration.h` | CPU round-trip redistribution |
| `vtk_output.h` | VTK output + domain boundary viz |
| `input.h` | JSON scene loading |
| `energy_diagnostics.cuh` | KE + momentum reduction |
| `scripts/gen_random3d.py` | 3D scene generator |
| `scripts/sbatch_*.sh` | Slurm batch scripts |

## Cluster hardware

| Partition | GPUs/node | Type |
|---|---|---|
| `a100` | 4× | A100 SXM4 (NVLink, 40 GB) |
| `work` | 4× | RTX 2080 Ti (11 GB) |
| `work` | 8× | RTX 3080 (10 GB) |
| `v100` | 4× | Tesla V100 (32 GB) |
