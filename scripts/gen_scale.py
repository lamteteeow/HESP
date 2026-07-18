#!/usr/bin/env python3
"""Generate a large random 3D particle scene for scaling benchmarks.

Non-overlapping placement via simple cubic lattice + random perturbation.
Way faster than rejection sampling for large N.

Usage:
    python3 scripts/gen_scale.py 10000 > scenes/scale10k.json
    python3 scripts/gen_scale.py 50000 > scenes/scale50k.json
"""

import json
import random
import sys
import math

N         = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
SEED      = 42
RADIUS    = 0.3
DT        = 5e-5
VEL_MAX   = 5.0
KN        = 5000.0
GAMMA_N   = 20.0
GAMMA_T   = 10.0
MU        = 0.3
MASS_MIN  = 0.5
MASS_MAX  = 1.5
GRAVITY   = [0.0, 0.0, 0.0]

# Size domain so packing fraction ~ 5% (plenty of room to move)
packing_frac = 0.05
vol_per_particle = (4.0 / 3.0) * math.pi * RADIUS**3
domain_vol = N * vol_per_particle / packing_frac
side = round(domain_vol ** (1.0 / 3.0), 1)
DOMAIN = [side, side, side]

# Place particles on a cubic lattice, one per cell
n_per_axis = int(math.ceil(N ** (1.0 / 3.0)))
spacing = side / n_per_axis
assert spacing > 2.0 * RADIUS, f"Spacing {spacing:.3f} too small for radius {RADIUS}"

random.seed(SEED)

particles = []
idx = 0
for ix in range(n_per_axis):
    for iy in range(n_per_axis):
        for iz in range(n_per_axis):
            if idx >= N:
                break
            # Base position at lattice point + small random perturbation
            x = (ix + 0.5) * spacing + random.uniform(-0.1 * spacing, 0.1 * spacing)
            y = (iy + 0.5) * spacing + random.uniform(-0.1 * spacing, 0.1 * spacing)
            z = (iz + 0.5) * spacing + random.uniform(-0.1 * spacing, 0.1 * spacing)
            # Clamp to domain
            x = max(RADIUS, min(side - RADIUS, x))
            y = max(RADIUS, min(side - RADIUS, y))
            z = max(RADIUS, min(side - RADIUS, z))
            particles.append({
                "position":  [round(x, 4), round(y, 4), round(z, 4)],
                "velocity":  [round(random.uniform(-VEL_MAX, VEL_MAX), 4),
                              round(random.uniform(-VEL_MAX, VEL_MAX), 4),
                              round(random.uniform(-VEL_MAX, VEL_MAX), 4)],
                "radius":    RADIUS,
                "mass":      round(random.uniform(MASS_MIN, MASS_MAX), 4),
                "kn":        KN,
                "gamma_n":   GAMMA_N,
                "gamma_t":   GAMMA_T,
                "mu":        MU,
            })
            idx += 1

scene = {
    "_comment": (f"{N} particles on perturbed cubic lattice, "
                 f"domain={side}x{side}x{side}, radius={RADIUS}, seed={SEED}"),
    "dt":        DT,
    "gravity":   GRAVITY,
    "domain":    {"min": [0.0, 0.0, 0.0], "max": DOMAIN},
    "cell_size": 2.0 * RADIUS,
    "particles": particles,
}

json.dump(scene, sys.stdout, indent=2)
