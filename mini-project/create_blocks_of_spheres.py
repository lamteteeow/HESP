# ─── Example Usage ─────────────────────────────────────────────────────────────
from data_structures import generate_cube, Scene

if __name__ == "__main__":
    # scenario settings
    dt = 1e-3
    gravity = (0.0, -9.81, 0)

    # common params
    mass = 1.0
    nx, ny, nz = 3, 3, 3
    dims = (nx, ny, nz)
    radius = 1.0
    spacing = 0.0

    # uniform‐radius grids
    radii_cube = [[[radius] * nz for _ in range(ny)] for _ in range(nx)]
    radii_plane = [[radius] * ny for _ in range(nx)]

    cube = generate_cube(1, (-39.0, -9.0, -10.0), dims, spacing, radius, radii_cube, mass, (-15.0, 1.0, 1.0))
    cube2 = generate_cube(len(cube) + 1, (-9.0, -9.0, -9.0), dims, spacing, radius, radii_cube, mass, (0.0, 0.0, 0.0))
    cube3 = generate_cube(len(cube2) + 1, (21.0, -9.0, -8.0), dims, spacing, radius, radii_cube, mass, (-20.0, -1.0, -1.0))
    spheres = cube + cube2 + cube3

    scene = Scene(
        dt=dt,
        gravity=gravity,
        margin=10,
        spheres=spheres
    )
    scene.write_scenario("blocksOfSpheres.json")