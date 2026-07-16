#!/usr/bin/env python3
"""Generate a multi-GPU stress scene: particles clustered at domain boundaries
with high velocity to force frequent migration.

For N GPUs in 3D mode, the split is determined by factorGrid3D(N):
  2 → 1×1×2 (Z-split at z=5)
  4 → 2×2×1 (X/Y-split)
Particles are placed near whichever boundaries exist for the given GPU count.

Usage:
    python3 scripts/gen_crossing.py [num_gpus] > scenes/crossing_freq.json
"""

import json
import random
import sys

# ── config ──────────────────────────────────────────────────────────────────
NUM_GPUS     = int(sys.argv[1]) if len(sys.argv) > 1 else 4
SEED         = 42
DOMAIN_X     = 20.0
DOMAIN_Y     = 10.0
DOMAIN_Z     = 10.0       # non-zero for 3D splits
PARTICLES    = 500
STRIP_WIDTH  = 0.3         # half-width of boundary strip
RADIUS       = 0.15
MASS         = 1.0
KN           = 5000.0
GAMMA_N      = 20.0
GAMMA_T      = 10.0
MU           = 0.3
DT           = 5e-4
VEL_MIN      = 20.0        # m/s in crossing direction
VEL_MAX      = 50.0
# ────────────────────────────────────────────────────────────────────────────

random.seed(SEED)

# Determine which axis the boundaries are on for this GPU count.
# 2 GPUs → Z-split (1×1×2), 4 GPUs → X/Y-split (2×2×1).
# For other counts, pick the axis with the largest divisor.
if NUM_GPUS == 2:
    # 1×1×2 → Z boundary at domain midpoint
    boundaries_axis = 'z'
    axis_size = DOMAIN_Z
    other1_size = DOMAIN_X
    other2_size = DOMAIN_Y
elif NUM_GPUS == 4:
    # 2×2×1 → X and Y boundaries
    boundaries_axis = 'xy'
else:
    # General case: find axes that are split
    # For now just use X (most common for >4)
    boundaries_axis = 'x'
    axis_size = DOMAIN_X
    other1_size = DOMAIN_Y
    other2_size = DOMAIN_Z

particles = []

if boundaries_axis == 'z':
    # Single Z boundary at domain midpoint
    bz = DOMAIN_Z / 2.0
    for _ in range(PARTICLES):
        side = random.choice([-1, 1])
        z = bz + side * random.uniform(0.01, STRIP_WIDTH)
        x = random.uniform(RADIUS, DOMAIN_X - RADIUS)
        y = random.uniform(RADIUS, DOMAIN_Y - RADIUS)
        # Velocity along Z to cross the boundary
        vz = side * random.uniform(VEL_MIN, VEL_MAX) * random.choice([-1, 1])
        vx = random.uniform(-2.0, 2.0)
        vy = random.uniform(-2.0, 2.0)
        particles.append({
            "position":  [round(x, 6), round(y, 6), round(z, 6)],
            "velocity":  [round(vx, 6), round(vy, 6), round(vz, 6)],
            "radius":    RADIUS, "mass": MASS,
            "kn": KN, "gamma_n": GAMMA_N, "gamma_t": GAMMA_T, "mu": MU,
        })

elif boundaries_axis == 'xy':
    # X and Y boundaries for 4-GPU 2×2×1 split
    bx = DOMAIN_X / 2.0
    by = DOMAIN_Y / 2.0
    boundaries = [('x', bx), ('y', by)]
    per_boundary = PARTICLES // len(boundaries)
    remainder = PARTICLES % len(boundaries)
    for bi, (axis, bval) in enumerate(boundaries):
        n = per_boundary + (1 if bi < remainder else 0)
        for _ in range(n):
            side = random.choice([-1, 1])
            if axis == 'x':
                x = bval + side * random.uniform(0.01, STRIP_WIDTH)
                y = random.uniform(RADIUS, DOMAIN_Y - RADIUS)
                vx = side * random.uniform(VEL_MIN, VEL_MAX) * random.choice([-1, 1])
                vy = random.uniform(-2.0, 2.0)
            else:
                x = random.uniform(RADIUS, DOMAIN_X - RADIUS)
                y = bval + side * random.uniform(0.01, STRIP_WIDTH)
                vx = random.uniform(-2.0, 2.0)
                vy = side * random.uniform(VEL_MIN, VEL_MAX) * random.choice([-1, 1])
            z = random.uniform(RADIUS, DOMAIN_Z - RADIUS)
            vz = random.uniform(-2.0, 2.0)
            particles.append({
                "position":  [round(x, 6), round(y, 6), round(z, 6)],
                "velocity":  [round(vx, 6), round(vy, 6), round(vz, 6)],
                "radius":    RADIUS, "mass": MASS,
                "kn": KN, "gamma_n": GAMMA_N, "gamma_t": GAMMA_T, "mu": MU,
            })

else:
    # X boundaries only
    dx = DOMAIN_X / NUM_GPUS
    boundaries = [i * dx for i in range(1, NUM_GPUS)]
    per_boundary = PARTICLES // len(boundaries)
    remainder = PARTICLES % len(boundaries)
    for bi, bx in enumerate(boundaries):
        n = per_boundary + (1 if bi < remainder else 0)
        for _ in range(n):
            side = random.choice([-1, 1])
            x = bx + side * random.uniform(0.01, STRIP_WIDTH)
            y = random.uniform(RADIUS, DOMAIN_Y - RADIUS)
            z = random.uniform(RADIUS, DOMAIN_Z - RADIUS)
            vx = side * random.uniform(VEL_MIN, VEL_MAX) * random.choice([-1, 1])
            vy = random.uniform(-2.0, 2.0)
            vz = random.uniform(-2.0, 2.0)
            particles.append({
                "position":  [round(x, 6), round(y, 6), round(z, 6)],
                "velocity":  [round(vx, 6), round(vy, 6), round(vz, 6)],
                "radius":    RADIUS, "mass": MASS,
                "kn": KN, "gamma_n": GAMMA_N, "gamma_t": GAMMA_T, "mu": MU,
            })

scene = {
    "_comment": (f"Crossing-frequency stress: {PARTICLES} particles, "
                 f"{NUM_GPUS}-GPU 3D split, axis={boundaries_axis}, "
                 f"seed={SEED}, v=[{VEL_MIN},{VEL_MAX}]"),
    "dt":        DT,
    "gravity":   [0.0, 0.0, 0.0],
    "domain":    {"min": [0.0, 0.0, 0.0],
                  "max": [DOMAIN_X, DOMAIN_Y, DOMAIN_Z]},
    "cell_size": 2.0 * RADIUS,
    "particles": particles,
}

json.dump(scene, sys.stdout, indent=2)
