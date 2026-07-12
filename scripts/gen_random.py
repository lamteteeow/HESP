#!/usr/bin/env python3
"""Generate a random 2D particle scene for md2d."""

import json
import random
import sys

# --- Parameters ---
N      = 20       # number of particles
SEED   = 42       # random seed (change for different layouts)
DOMAIN = [10.0, 10.0]  # domain size in x, y

# DEM material properties
DT       = 0.00005
KN       = 5000.0
GAMMA_N  = 20.0
GAMMA_T  = 10.0
MU       = 0.3

# Particle ranges
R_MIN    = 0.2
R_MAX    = 0.35
MASS_MIN = 0.5
MASS_MAX = 1.5
VEL_MAX  = 5.0          # max initial speed per component

random.seed(SEED)

particles = []
for _ in range(N):
    r = random.uniform(R_MIN, R_MAX)
    m = random.uniform(MASS_MIN, MASS_MAX)

    # Position: keep particles away from walls (margin = r)
    x = random.uniform(r, DOMAIN[0] - r)
    y = random.uniform(r, DOMAIN[1] - r)

    # Random velocity
    vx = random.uniform(-VEL_MAX, VEL_MAX)
    vy = random.uniform(-VEL_MAX, VEL_MAX)

    particles.append({
        "position":  [round(x, 3), round(y, 3), 0.0],
        "velocity":  [round(vx, 3), round(vy, 3), 0.0],
        "radius":    round(r, 3),
        "mass":      round(m, 3),
        "kn":        KN,
        "gamma_n":   GAMMA_N,
        "gamma_t":   GAMMA_T,
        "mu":        MU,
    })

scene = {
    "_comment": f"{N} random particles, seed={SEED}, domain={DOMAIN[0]}×{DOMAIN[1]}",
    "dt":        DT,
    "gravity":   [0.0, 0.0, 0.0],
    "domain":    {"min": [0.0, 0.0, 0.0],
                  "max": [DOMAIN[0], DOMAIN[1], 0.0]},
    "cell_size": 2.0 * R_MAX,
    "particles": particles,
}

json.dump(scene, sys.stdout, indent=2)
