# md2d — GPU-accelerated 2D/3D molecular dynamics

Spring-dashpot DEM simulator with **N-GPU 3D domain decomposition**
(nx×ny×nz grid, factored from N GPUs).

## Build

```bash
make              # 3D binary (default)
make md2d         # 2D binary (z=0 enforced, X-only split)
```

## Run

```bash
./md2d scenes/cube256.json 5000 8 5   # scene, max steps, GPUs, VTK interval
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
