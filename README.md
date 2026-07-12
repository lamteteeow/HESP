# md2d — GPU-accelerated 2D molecular dynamics simulator

A spring-dashpot DEM (Discrete Element Method) simulator with **N-GPU domain
decomposition** across the X axis.

## Build

```bash
make                     # CUDA toolkit required (≥ 11.0)
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
```

## Features

- **N-GPU domain decomposition** — X-axis split with halo exchange for
  cross-boundary contacts. Supports any number of GPUs.
- **Cell-based neighbor search** — uniform grid with 27-neighbor lookup table,
  O(N) per step.
- **Spring-dashpot DEM** — normal and tangential contact forces with Coulomb
  friction.
- **Symplectic Euler integration** — z=0 enforced for 2D, reflective domain
  walls.
- **Particle migration** — particles that cross their owning slice are
  redistributed across GPUs.
- **VTK output** — positions, velocities, radii, and GPU ownership scalars per
  frame. ParaView-compatible time series with domain boundary wireframe.
- **Energy diagnostics** — kinetic energy and momentum printed per frame.

## Scene format

```json
{
    "dt":         0.0002,
    "gravity":    [0.0, 0.0, 0.0],
    "domain":     { "min": [0,0,0], "max": [10,10,0] },
    "cell_size":  0.6,
    "particles": [
        { "position": [x,y,0], "velocity": [vx,vy,0],
          "radius": r, "mass": m, "kn": 5000,
          "gamma_n": 20, "gamma_t": 10, "mu": 0.3 }
    ]
}
```

Generate scenes with the bundled scripts:

```bash
python3 scripts/gen_random.py > scenes/my_scene.json    # 2D random
python3 scripts/gen_lattice.py > scenes/lattice.json    # 2D lattice
python3 scripts/gen_random3d.py > scenes/cube.json      # 3D random
```

## Visualization

1. Download the `output/` directory to your local machine.
2. Open ParaView, `File → Open` → select all `.vtk` files in the output
   directory (they load as a time series). Include `domain_boundary.vtk`
   to see the domain wireframe and halo regions.
3. Click `Apply`.
4. Add a `Glyph` filter, set **Glyph Type** to `Sphere`, scale by the
   `radius` scalar. In the Glyph properties, set **Masking → Glyph Mode**
   to `All Points` to avoid radius interpolation artifacts.
5. Click `Play` to animate.
