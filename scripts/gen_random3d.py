#!/usr/bin/env python3
"""Generate a random 3D particle scene for md3d."""

import json, random, sys

N       = 8
SEED    = 42
DOMAIN  = [8.0, 8.0, 8.0]   # x, y, z
DT      = 0.0002
VEL_MAX = 4.0
R_MIN   = 0.3
R_MAX   = 0.5

random.seed(SEED)

particles = []
for _ in range(N):
    r = random.uniform(R_MIN, R_MAX)
    x = random.uniform(r, DOMAIN[0] - r)
    y = random.uniform(r, DOMAIN[1] - r)
    z = random.uniform(r, DOMAIN[2] - r)
    vx = random.uniform(-VEL_MAX, VEL_MAX)
    vy = random.uniform(-VEL_MAX, VEL_MAX)
    vz = random.uniform(-VEL_MAX, VEL_MAX)
    particles.append({
        "position":  [round(x,3), round(y,3), round(z,3)],
        "velocity":  [round(vx,3), round(vy,3), round(vz,3)],
        "radius":    round(r, 3),
        "mass":      round(random.uniform(0.5, 1.5), 3),
        "kn":        5000.0,
        "gamma_n":   20.0,
        "gamma_t":   10.0,
        "mu":        0.3,
    })

scene = {
    "_comment": f"{N} random particles in 3D, seed={SEED}",
    "dt":        DT,
    "gravity":   [0.0, 0.0, 0.0],
    "domain":    {"min": [0.0, 0.0, 0.0],
                  "max": DOMAIN},
    "cell_size": 2.0 * R_MAX,
    "particles": particles,
}

json.dump(scene, sys.stdout, indent=2)
