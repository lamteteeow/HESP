#!/usr/bin/env python3
import numpy as np
from pydantic import BaseModel, ConfigDict, model_validator, Field


# ─── Helpers ────────────────────────────────────────────────────────────────────

def compute_offsets(
        radii_line: list[float],
        origin: float,
        spacing: float
) -> list[float]:
    """
    Given radii [r0, r1, …], returns center‐positions [x0, x1, …]
    along one axis, starting at origin, with
      Δ = r_prev + r_curr + spacing
    """
    offsets = [origin]
    for r_prev, r_curr in zip(radii_line, radii_line[1:]):
        offsets.append(offsets[-1] + r_prev + r_curr + spacing)
    return offsets


def compute_inertia(mass: float, radius: float) -> float:
    """Solid‐sphere moment of inertia I = 0.4·m·r²"""
    return 0.4 * mass * radius * radius


def precompute_3d_offsets(
        radii: list[list[list[float]]],
        dims: tuple[int, int, int],
        origin: tuple[float, float, float],
        spacing: float
):
    nx, ny, nz = dims
    ox, oy, oz = origin

    x_off = {
        (j, k): compute_offsets(
            [radii[i][j][k] for i in range(nx)], ox, spacing
        )
        for j in range(ny) for k in range(nz)
    }
    y_off = {
        (i, k): compute_offsets(
            [radii[i][j][k] for j in range(ny)], oy, spacing
        )
        for i in range(nx) for k in range(nz)
    }
    z_off = {
        (i, j): compute_offsets(
            [radii[i][j][k] for k in range(nz)], oz, spacing
        )
        for i in range(nx) for j in range(ny)
    }
    return x_off, y_off, z_off


def precompute_2d_offsets(
        radii: list[list[float]],
        dims: tuple[int, int],
        origin: tuple[float, float, float],
        spacing: float
):
    nx, ny = dims
    ox, oy, _ = origin

    x_off = {
        j: compute_offsets([radii[i][j] for i in range(nx)], ox, spacing)
        for j in range(ny)
    }
    y_off = {
        i: compute_offsets([radii[i][j] for j in range(ny)], oy, spacing)
        for i in range(nx)
    }
    return x_off, y_off


# ─── Models ─────────────────────────────────────────────────────────────────────

class Sphere(BaseModel):
    id: int
    mass: float
    radius: float
    inertia: float
    position: tuple[float, float, float]
    velocity: tuple[float, float, float]
    force: tuple[float, float, float] = (0.0, 0.0, 0.0)
    orientation: tuple[float, float, float, float] = (
        1.0, 0.0, 0.0, 0.0
    )
    angularVelocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    torque: tuple[float, float, float] = (0.0, 0.0, 0.0)
    kn: float
    gamma_n: float
    mu: float
    gamma_t: float

class Plane(BaseModel):
    id: int
    normal: tuple[float, float, float]
    distance: float

class ConvexMesh(BaseModel):
    id: int
    vertices: list[tuple[float, float, float]]
    faces: list[tuple[int, int, int]]

class MeshInstance(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True,
                              json_encoders={np.ndarray: lambda v: v.tolist()})
    id: int
    mesh_id: int  # ConvexMesh.id
    mass: float
    scale: tuple[float, float, float]
    position: tuple[float, float, float]
    velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    force: tuple[float, float, float] = (0.0, 0.0, 0.0)
    orientation: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    angular_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    torque: tuple[float, float, float] = (0.0, 0.0, 0.0)
    inertia: np.ndarray
    inertia_inv: np.ndarray
    kn: float
    gamma_n: float
    mu: float
    gamma_t: float

class Scene(BaseModel):
    dt: float
    gravity: tuple[float, float, float]
    margin: float
    length: tuple[float, float, float] = None
    offset: tuple[float, float, float] = None
    planes: list[Plane] = Field(default_factory=list)
    spheres: list[Sphere] = Field(default_factory=list)
    complex_objects: list[MeshInstance] = Field(default_factory=list)
    meshes: list[ConvexMesh] = Field(default_factory=list)

    @model_validator(mode='after')
    def compute_all(self):
        self.compute_bounds()
        self.create_boundary_planes()
        return self

    def compute_bounds(self):
        pts = [*(s.position for s in self.spheres),
               *(b.position for b in self.complex_objects)]
        if not pts:
            self.offset = self.length = (0.0, 0.0, 0.0)
            return
        mins = tuple(min(axis) - self.margin for axis in zip(*pts))
        maxs = tuple(max(axis) + self.margin for axis in zip(*pts))
        self.offset = mins
        self.length = tuple(M - m for M, m in zip(maxs, mins))

    def create_boundary_planes(self):
        ox, oy, oz = self.offset
        lx, ly, lz = self.length
        self.planes = [Plane(id=666, normal=(1, 0, 0), distance=ox),
                       Plane(id=666, normal=(-1, 0, 0), distance=-(lx + ox)),
                       Plane(id=666, normal=(0, 1, 0), distance=oy),
                       Plane(id=666, normal=(0, -1, 0), distance=-(ly + oy)),
                       Plane(id=666, normal=(0, 0, 1), distance=oz),
                       Plane(id=666, normal=(0, 0, -1), distance=-(lz + oz))]

    def write_scenario(self, filename: str):
        with open(filename, "w") as f:
            f.write(self.model_dump_json(indent=2))
        print(f"Wrote '{filename}'")


# ─── Generators ─────────────────────────────────────────────────────────────────

def generate_cube(
        start_id: int,
        origin: tuple[float, float, float],
        dims: tuple[int, int, int],
        spacing: float,
        radius: float,
        radii_grid: list[list[list[float]]],
        mass: float,
        velocity: tuple[float, float, float],
        kn: float = 1e4,
        gamma_n: float = 50.0,
        mu: float = 0.3,
        gamma_t: float = 20.0
) -> list[Sphere]:
    x_off, y_off, z_off = precompute_3d_offsets(
        radii_grid, dims, origin, spacing
    )
    spheres: list[Sphere] = []
    pid = start_id

    nx, ny, nz = dims
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                r = radii_grid[i][j][k]
                pos = (x_off[(j, k)][i], y_off[(i, k)][j], z_off[(i, j)][k])
                spheres.append(Sphere(
                    id=pid, mass=mass, radius=r, inertia=compute_inertia(mass, radius), position=pos,
                    velocity=velocity, kn=kn, gamma_n=gamma_n, mu=mu, gamma_t=gamma_t
                ))
                pid += 1

    return spheres


def generate_plane_variable_radii(
        start_id: int,
        origin: tuple[float, float, float],
        dims: tuple[int, int],
        spacing: float,
        radius: float,
        radii_grid: list[list[float]],
        mass: float,
        velocity: tuple[float, float, float],
        kn: float = 1e4,
        gamma_n: float = 50.0,
        mu: float = 0.3,
        gamma_t: float = 20.0
) -> list[Sphere]:
    x_off, y_off = precompute_2d_offsets(
        radii_grid, dims, origin, spacing
    )
    particles: list[Sphere] = []
    pid = start_id

    nx, ny = dims
    _, _, oz = origin
    for i in range(nx):
        for j in range(ny):
            r = radii_grid[i][j]
            pos = (x_off[j][i], y_off[i][j], oz)
            particles.append(Sphere(
                id=pid, mass=mass, radius=r, inertia=compute_inertia(mass, radius), position=pos,
                velocity=velocity, kn=kn, gamma_n=gamma_n, mu=mu, gamma_t=gamma_t
            ))
            pid += 1

    return particles

def compute_unit_inertia_polyhedron(vertices: list[tuple], faces: list[tuple]) -> np.ndarray:
    """
    Mirtich ’96: compute inertia of a closed convex polyhedron of unit mass & unit scale.
    vertices: list of 3-tuples
    faces:    list of index-triples
    Returns 3×3 inertia matrix about the origin.
    """
    I = np.zeros((3,3))
    vol_accum = 0.0

    for (i0,i1,i2) in faces:
        v0 = np.array(vertices[i0])
        v1 = np.array(vertices[i1])
        v2 = np.array(vertices[i2])

        # tetra volume from origin
        vol = abs(np.linalg.det(np.stack((v0, v1, v2), axis=1)) / 6.0)

        vol_accum += vol

        # ∫(r·r) dV over tetra = vol/10 * sum(dot(vi,vi) + dot(vi,vj))
        dots = np.array([
            v0.dot(v0), v1.dot(v1), v2.dot(v2),
            v0.dot(v1), v1.dot(v2), v2.dot(v0),
        ])
        S = vol/10.0 * dots.sum()

        # approximate ∫(r r^T) dV ≈ (vol/20) * sum(vi⊗vi)
        M = np.outer(v0,v0) + np.outer(v1,v1) + np.outer(v2,v2)

        I += S * np.eye(3) - (vol/20.0) * M

    return I / vol_accum


def inertia_mesh(mass: float,
                 scale: tuple[float,float,float],
                 vertices: list[tuple],
                 faces: list[tuple]) -> np.ndarray:
    """
    Full inertia for *any* convex mesh:
      1. compute unit-mass, unit-scale inertia via Mirtich
      2. apply the scale-law: I_scaled = mass * S @ I_unit @ S
         where S = diag(sy*sz, sx*sz, sx*sy)
    """
    # 1) unit inertia
    I1 = compute_unit_inertia_polyhedron(vertices, faces)

    # 2) build S
    sx, sy, sz = scale
    S = np.diag([sy*sz, sx*sz, sx*sy])

    # 3) scaled inertia
    I_full = mass * (S @ I1 @ S)
    return np.diag(I_full)

def inverse_inertia_diagonal(I: np.ndarray) -> np.ndarray:
    """
    Invert a diagonal inertia by taking 1/x on each diagonal.
    """
    return 1.0 / I

