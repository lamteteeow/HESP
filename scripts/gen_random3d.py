#!/usr/bin/env python3
"""Generate a random 3D particle scene for md3d."""

import json, random, sys, math

N       = 16
SEED    = 42
DOMAIN  = [16.0, 16.0, 16.0]   # x, y, z
DT      = 0.0002
VEL_MAX = 8.0
R_MIN   = 0.3
R_MAX   = 0.5
MASS_MIN = 0.5
MASS_MAX = 1.5

# Softer DEM parameters for visible deformation on collision
KN      = 1000.0
GAMMA_N = 5.0
GAMMA_T = 2.5
MU      = 0.3

random.seed(SEED)

def overlaps(particles, x, y, z, r):
    for p in particles:
        dx = x - p["position"][0]
        dy = y - p["position"][1]
        dz = z - p["position"][2]
        dist = math.sqrt(dx*dx + dy*dy + dz*dz)
        if dist < r + p["radius"]:
            return True
    return False

particles = []
attempts = 0
while len(particles) < N and attempts < 5000:
    r = random.uniform(R_MIN, R_MAX)
    x = random.uniform(r, DOMAIN[0] - r)
    y = random.uniform(r, DOMAIN[1] - r)
    z = random.uniform(r, DOMAIN[2] - r)
    if not overlaps(particles, x, y, z, r):
        vx = random.uniform(-VEL_MAX, VEL_MAX)
        vy = random.uniform(-VEL_MAX, VEL_MAX)
        vz = random.uniform(-VEL_MAX, VEL_MAX)
        particles.append({
            "position":  [round(x,3), round(y,3), round(z,3)],
            "velocity":  [round(vx,3), round(vy,3), round(vz,3)],
            "radius":    round(r, 3),
            "mass":      round(random.uniform(0.5, 1.5), 3),
            "kn":        KN,
            "gamma_n":   GAMMA_N,
            "gamma_t":   GAMMA_T,
            "mu":        MU,
        })
    attempts += 1

scene = {
    "_comment": f"{len(particles)} non-overlapping particles in 3D, seed={SEED}",
    "dt":        DT,
    "gravity":   [0.0, 0.0, 0.0],
    "domain":    {"min": [0.0, 0.0, 0.0],
                  "max": DOMAIN},
    "cell_size": 2.0 * R_MAX,
    "particles": particles,
}

json.dump(scene, sys.stdout, indent=2)
