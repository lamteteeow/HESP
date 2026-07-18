#!/usr/bin/env python3
"""Generate a large scene that also stresses migration — particles biased
 toward domain boundaries with high crossing velocity, plus background fill.

 10,000 particles, radius 0.7 (halo width 1.4), ~10% packing fraction,
 50% crossers at 60 m/s near X/Y boundaries, 50% lattice fill.

 Usage:
     python3 scripts/gen_scale_crossing.py 10000 > scenes/scale10k_crossing.json
 """

import json
import random
import sys
import math

N         = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
SEED      = 42
RADIUS    = 0.7
DT        = 5e-5
KN        = 5000.0
GAMMA_N   = 20.0
GAMMA_T   = 10.0
MU        = 0.3
MASS_MIN  = 0.5
MASS_MAX  = 1.5
GRAVITY   = [0.0, 0.0, 0.0]

# Size domain for ~10% packing fraction
packing_frac = 0.10
vol_per_particle = (4.0 / 3.0) * math.pi * RADIUS**3
domain_vol = N * vol_per_particle / packing_frac
side = round(domain_vol ** (1.0 / 3.0), 1)
DOMAIN = [side, side, side]

random.seed(SEED)

# ── Determine boundaries for 4-GPU 2×2×1 split ──────────────────────────
# X boundary at side/2, Y boundary at side/2
BX = side / 2.0
BY = side / 2.0
STRIP_WIDTH = 0.6  # half-width of boundary strip

# 50% of particles are "crossers" placed near boundaries with high velocity
N_CROSS = N * 50 // 100
N_FILL  = N - N_CROSS

particles = []

def add_particle(x, y, z, vx, vy, vz):
    particles.append({
        "position":  [round(x, 4), round(y, 4), round(z, 4)],
        "velocity":  [round(vx, 4), round(vy, 4), round(vz, 4)],
        "radius":    RADIUS,
        "mass":      round(random.uniform(MASS_MIN, MASS_MAX), 4),
        "kn":        KN, "gamma_n": GAMMA_N, "gamma_t": GAMMA_T, "mu": MU,
    })

# ── Crossing particles ───────────────────────────────────────────────────
# Place near X and Y boundaries with velocity perpendicular to the boundary
CROSS_VEL = 60.0  # m/s — bounce off walls rapidly, many crossings per run
boundaries = [
    ("x", BX),  # X boundary
    ("y", BY),  # Y boundary
]
per_bdry = N_CROSS // len(boundaries)

for axis, bval in boundaries:
    for _ in range(per_bdry):
        side_sign = random.choice([-1, 1])
        if axis == "x":
            x = bval + side_sign * random.uniform(0.01, STRIP_WIDTH)
            y = random.uniform(RADIUS, side - RADIUS)
            vx = -side_sign * random.uniform(CROSS_VEL * 0.5, CROSS_VEL)
            vy = random.uniform(-5.0, 5.0)
        else:  # y
            y = bval + side_sign * random.uniform(0.01, STRIP_WIDTH)
            x = random.uniform(RADIUS, side - RADIUS)
            vx = random.uniform(-5.0, 5.0)
            vy = -side_sign * random.uniform(CROSS_VEL * 0.5, CROSS_VEL)
        z = random.uniform(RADIUS, side - RADIUS)
        vz = random.uniform(-5.0, 5.0)
        add_particle(x, y, z, vx, vy, vz)

# ── Fill particles ──────────────────────────────────────────────────────
# Place on a cubic lattice with random perturbation (no overlap)
n_per_axis = int(math.ceil(N_FILL ** (1.0 / 3.0)))
spacing = side / n_per_axis

idx = 0
for ix in range(n_per_axis):
    for iy in range(n_per_axis):
        for iz in range(n_per_axis):
            if idx >= N_FILL:
                break
            x = (ix + 0.5) * spacing + random.uniform(-0.1 * spacing, 0.1 * spacing)
            y = (iy + 0.5) * spacing + random.uniform(-0.1 * spacing, 0.1 * spacing)
            z = (iz + 0.5) * spacing + random.uniform(-0.1 * spacing, 0.1 * spacing)
            x = max(RADIUS, min(side - RADIUS, x))
            y = max(RADIUS, min(side - RADIUS, y))
            z = max(RADIUS, min(side - RADIUS, z))
            vx = random.uniform(-5.0, 5.0)
            vy = random.uniform(-5.0, 5.0)
            vz = random.uniform(-5.0, 5.0)
            add_particle(x, y, z, vx, vy, vz)
            idx += 1

scene = {
    "_comment": (f"{len(particles)} particles ({N_CROSS} crossing + {len(particles)-N_CROSS} fill), "
                 f"domain={side}x{side}x{side}, boundary strips at X={BX:.1f} Y={BY:.1f}, "
                 f"cross_vel={CROSS_VEL} m/s, seed={SEED}"),
    "dt":        DT,
    "gravity":   GRAVITY,
    "domain":    {"min": [0.0, 0.0, 0.0], "max": DOMAIN},
    "cell_size": 2.0 * RADIUS,
    "particles": particles,
}

json.dump(scene, sys.stdout, indent=2)
