from data_structures import Scene, ConvexMesh, MeshInstance, inertia_mesh, inverse_inertia_diagonal

if __name__ == "__main__":
    dt = 1e-3
    gravity = (0.0, 0.0, 0)
    unit_cube = ConvexMesh(
        id=1,
        vertices=[
            (-0.5, -0.5, -0.5),  # 0
            (0.5, -0.5, -0.5),  # 1
            (0.5, 0.5, -0.5),  # 2
            (-0.5, 0.5, -0.5),  # 3
            (-0.5, -0.5, 0.5),  # 4
            (0.5, -0.5, 0.5),  # 5
            (0.5, 0.5, 0.5),  # 6
            (-0.5, 0.5, 0.5),  # 7
        ],
        faces=[
            # bottom (z = –0.5)
            (0, 1, 2), (0, 2, 3),
            # top (z = +0.5)
            (4, 6, 5), (4, 7, 6),
            # front (y = –0.5)
            (0, 5, 1), (0, 4, 5),
            # back  (y = +0.5)
            (3, 2, 6), (3, 6, 7),
            # right (x = +0.5)
            (1, 5, 6), (1, 6, 2),
            # left  (x = –0.5)
            (0, 3, 7), (0, 7, 4),
        ]
    )
    unit_pyramid = ConvexMesh(
        id=2,
        vertices=[
            (-0.5, -0.5, -0.5),  # 0: base corner
            (0.5, -0.5, -0.5),  # 1
            (0.5, 0.5, -0.5),  # 2
            (-0.5, 0.5, -0.5),  # 3
            (0.0, 0.0, 0.5),  # 4: apex
        ],
        faces=[
            # base (z = –0.5), split into two triangles
            (0, 1, 2), (0, 2, 3),
            # four triangular side faces
            (4, 0, 1),
            (4, 1, 2),
            (4, 2, 3),
            (4, 3, 0),
        ]
    )
    tetrahedron = ConvexMesh(
        id=3,
        vertices=[
            (0.0, 0.0, 0.0),  # 0
            (1.0, 0.0, 0.0),  # 1
            (0.0, 2.0, 0.0),  # 2
            (0.0, 0.0, 3.0),  # 3
        ],
        faces=[
            (0, 1, 2),  # base
            (0, 1, 3),
            (1, 2, 3),
            (2, 0, 3),
        ]
    )
    regular_octahedron = ConvexMesh(
        id=5,
        vertices=[
            (1.0, 0.0, 0.0),  # +X
            (-1.0, 0.0, 0.0),  # -X
            (0.0, 1.0, 0.0),  # +Y
            (0.0, -1.0, 0.0),  # -Y
            (0.0, 0.0, 1.0),  # +Z (top)
            (0.0, 0.0, -1.0),  # -Z (bottom)
        ],
        faces=[
            # top‐cap triangles (apex = vertex 4)
            (4, 0, 2),
            (4, 2, 1),
            (4, 1, 3),
            (4, 3, 0),
            # bottom‐cap triangles (apex = vertex 5)
            (5, 2, 0),
            (5, 1, 2),
            (5, 3, 1),
            (5, 0, 3),
        ]
    )
    kn: float = 1e4
    gamma_n: float = 50.0
    mu: float = 0.3
    gamma_t: float = 20.0
    in1 = inertia_mesh(1.0, (1.0, 1.0, 1.0), unit_cube.vertices, unit_cube.faces)
    inv1 = inverse_inertia_diagonal(in1)
    in2 = inertia_mesh(1.0, (2.0, 2.0, 2.0), unit_cube.vertices, unit_cube.faces)
    inv2 = inverse_inertia_diagonal(in2)
    cubes = [
        MeshInstance(id=666, mesh_id=1, mass=1.0, scale=(1.0, 1.0, 1.0), position=(5.0, 3.0, 5.0), inertia=in1,
                     inertia_inv=inv1, kn=kn, gamma_n=gamma_n, mu=mu, gamma_t=gamma_t),
        MeshInstance(id=666, mesh_id=1, mass=1.0, scale=(1.0, 1.0, 1.0), position=(5.0, 4.0, 5.0),
                     velocity=(0.0, -1.0, 0.0), inertia=in1,
                     inertia_inv=inv1, kn=kn, gamma_n=gamma_n, mu=mu, gamma_t=gamma_t),
        MeshInstance(id=666, mesh_id=1, mass=1.0, scale=(1.0, 1.0, 1.0), position=(5.0, 1.0, 5.0), inertia=in1,
                     inertia_inv=inv1, kn=kn, gamma_n=gamma_n, mu=mu, gamma_t=gamma_t),
        MeshInstance(id=666, mesh_id=1, mass=2.0, scale=(2.0, 2.0, 2.0), position=(1.0, 5.0, 5.0),
                     velocity=(10.0, 1.0, 1.0), inertia=in2, inertia_inv=inv2, kn=kn, gamma_n=gamma_n, mu=mu,
                     gamma_t=gamma_t)
    ]
    scene = Scene(
        dt=dt,
        gravity=gravity,
        margin=10,
        complex_objects=cubes,
        meshes=[unit_cube]
    )
    scene.write_scenario("boxTower.json")

    MeshInstance(id=666, mesh_id=1, mass=1.0, scale=(1.0, 1.0, 1.0), position=(5.0, 1.0, 5.0), inertia=in1,
                 inertia_inv=inv1, kn=kn, gamma_n=gamma_n, mu=mu, gamma_t=gamma_t),
    MeshInstance(id=666, mesh_id=1, mass=2.0, scale=(2.0, 2.0, 2.0), position=(1.0, 5.0, 5.0),
                 velocity=(10.0, 1.0, 1.0), inertia=in2, inertia_inv=inv2, kn=kn, gamma_n=gamma_n, mu=mu,
                 gamma_t=gamma_t)
