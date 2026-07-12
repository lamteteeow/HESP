# md2d — GPU-accelerated 2D/3D molecular dynamics simulator

A spring-dashpot DEM (Discrete Element Method) simulator with **N-GPU domain
decomposition**.

## Build

```bash
make                     # 2D binary (CUDA toolkit ≥ 11.0 required)
make md3d                # 3D binary (defines MD3D)
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

## Features

- **2D grid domain decomposition** — 3D mode factors N GPUs into a near-square
  nx×ny grid (e.g. 4→2×2, 6→3×2). Each GPU exchanges halos with up to 4
  neighbors. 2D mode uses X-only split.
- **Cell-based neighbor search** — uniform grid with 27-neighbor lookup table,
  O(N) per step.
- **Spring-dashpot DEM** — normal and tangential contact forces with Coulomb
  friction.
- **Symplectic Euler integration** — reflective domain walls on all axes.
  2D enforces z=0; 3D allows free z motion.
- **Particle migration** — particles crossing their owning region are
  redistributed across the GPU grid.
- **Energy diagnostics** — kinetic energy and momentum printed per frame.
- **VTK output** — per-frame ParaView time series with:
  - Positions, velocities, radii (glyph as spheres)
  - `gpu_owner` scalar — color particles by owning GPU
  - `border` scalar — 0→1 gradient showing halo proximity at GPU boundaries
  - `domain_boundary.vtk` — wireframe box, GPU split lines, filled halo strips

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
python3 scripts/gen_lattice.py  > scenes/lattice.json  # 2D lattice
python3 scripts/gen_random3d.py > scenes/my_3d.json    # 3D random
```

## Visualization

1. Download the `output/` directory to your local machine.
2. Open ParaView, `File → Open` → select all `.vtk` files in the output
   directory (they load as a time series). Include `domain_boundary.vtk`
   to see the domain wireframe, GPU split lines, and halo strips.
3. Click `Apply`.
4. Add a `Glyph` filter, set **Glyph Type** to `Sphere`, scale by the
   `radius` scalar. In Glyph properties, set **Masking → Glyph Mode**
   to `All Points` to avoid radius interpolation artifacts.
5. Color particles by `gpu_owner` to see which GPU owns each, or by
   `border` to see the halo overlap gradient at GPU boundaries.
6. Click `Play` to animate.
