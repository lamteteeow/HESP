#!/usr/bin/env python3
"""Generate a comet scene: dense cluster at one corner moving diagonally
across the domain with a trailing tail, crossing all GPU boundaries.

Usage:
    python3 scripts/gen_comet.py 100000 > scenes/comet100k.json
"""

import json
import math
import random
import sys

class V3:
    __slots__ = ('x', 'y', 'z')
    def __init__(self, x, y, z): self.x = x; self.y = y; self.z = z
    def __sub__(self, o): return V3(self.x - o.x, self.y - o.y, self.z - o.z)
    def __add__(self, o): return V3(self.x + o.x, self.y + o.y, self.z + o.z)
    def __mul__(self, s): return V3(self.x * s, self.y * s, self.z * s)
    def length(self): return math.sqrt(self.x**2 + self.y**2 + self.z**2)
    def norm(self):
        l = self.length()
        return V3(self.x / l, self.y / l, self.z / l) if l > 0 else V3(0, 0, 0)

N         = int(sys.argv[1]) if len(sys.argv) > 1 else 100000
SEED      = 42

RADIUS    = 0.3
DT        = 5e-5
KN        = 5000.0
GAMMA_N   = 20.0
GAMMA_T   = 10.0
MU        = 0.3
MASS_MIN  = 0.5
MASS_MAX  = 1.5
GRAVITY   = [0.0, 0.0, 0.0]

packing_frac = 0.05
vol_per_particle = (4.0 / 3.0) * math.pi * RADIUS**3
domain_vol = N * vol_per_particle / packing_frac
side = round(domain_vol ** (1.0 / 3.0), 1)
DOMAIN = [side, side, side]

random.seed(SEED)

HEAD_PCT    = 20
TAIL_PCT    = 30
TAIL_LENGTH = 10.0
VEL_HEAD    = 80.0
VEL_TAIL_FRONT = 40.0
VEL_TAIL_BACK  = 5.0

N_HEAD  = N * HEAD_PCT // 100
N_TAIL  = N * TAIL_PCT // 100
N_FILL  = N - N_HEAD - N_TAIL

corner_start = V3(RADIUS, RADIUS, RADIUS)
corner_end   = V3(side - RADIUS, side - RADIUS, side - RADIUS)
diag         = corner_end - corner_start
diag_norm    = diag.norm()

HEAD_RADIUS = 5.0

particles = []

def add(x, y, z, vx, vy, vz):
    x = max(RADIUS, min(side - RADIUS, x))
    y = max(RADIUS, min(side - RADIUS, y))
    z = max(RADIUS, min(side - RADIUS, z))
    particles.append({
        "position":  [round(x, 4), round(y, 4), round(z, 4)],
        "velocity":  [round(vx, 4), round(vy, 4), round(vz, 4)],
        "radius":    RADIUS,
        "mass":      round(random.uniform(MASS_MIN, MASS_MAX), 4),
        "kn":        KN, "gamma_n": GAMMA_N, "gamma_t": GAMMA_T, "mu": MU,
    })

# ── Head: dense sphere at the corner ──────────────────────────────────────────

head_center = corner_start + diag_norm * HEAD_RADIUS
for _ in range(N_HEAD):
    r = HEAD_RADIUS * random.random() ** (1.0 / 3.0)
    theta = random.uniform(0, 2 * math.pi)
    phi   = math.acos(2 * random.random() - 1)
    x = head_center.x + r * math.sin(phi) * math.cos(theta)
    y = head_center.y + r * math.sin(phi) * math.sin(theta)
    z = head_center.z + r * math.cos(phi)
    v_spread = 20.0
    vx = diag_norm.x * VEL_HEAD + random.uniform(-v_spread, v_spread)
    vy = diag_norm.y * VEL_HEAD + random.uniform(-v_spread, v_spread)
    vz = diag_norm.z * VEL_HEAD + random.uniform(-v_spread, v_spread)
    add(x, y, z, vx, vy, vz)

# ── Cross product helper for perpendicular vectors ────────────────────────────
def cross(a, b):
    return V3(a.y * b.z - a.z * b.y,
              a.z * b.x - a.x * b.z,
              a.x * b.y - a.y * b.x)

# ── Tail: trailing behind the head along the diagonal ─────────────────────────

# Pick two perpendicular directions in the plane normal to the diagonal
arbitrary = V3(0, 1, 0) if abs(diag_norm.x) < 0.9 else V3(1, 0, 0)
perp1 = cross(diag_norm, arbitrary).norm()
perp2 = cross(diag_norm, perp1).norm()

for i in range(N_TAIL):
    dist = TAIL_LENGTH * (i / max(N_TAIL, 1))
    t = dist / TAIL_LENGTH

    spread = 2.0 + t * 8.0
    base = head_center - diag_norm * dist

    r = spread * random.random() ** 0.5
    angle = random.uniform(0, 2 * math.pi)
    x = base.x + r * math.cos(angle) * perp1.x + r * math.sin(angle) * perp2.x
    y = base.y + r * math.cos(angle) * perp1.y + r * math.sin(angle) * perp2.y
    z = base.z + r * math.cos(angle) * perp1.z + r * math.sin(angle) * perp2.z

    v_mag = VEL_TAIL_BACK + (VEL_TAIL_FRONT - VEL_TAIL_BACK) * (1 - t)
    vx = diag_norm.x * v_mag + random.uniform(-10, 10)
    vy = diag_norm.y * v_mag + random.uniform(-10, 10)
    vz = diag_norm.z * v_mag + random.uniform(-10, 10)
    add(x, y, z, vx, vy, vz)

# ── Fill: cubic lattice ───────────────────────────────────────────────────────

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
            add(x, y, z, vx, vy, vz)
            idx += 1

scene = {
    "_comment": (f"{len(particles)} particles: {N_HEAD} head + {N_TAIL} tail + {len(particles)-N_HEAD-N_TAIL} fill, "
                 f"domain={side:.0f}x{side:.0f}x{side:.0f}, head_radius={HEAD_RADIUS:.0f}, "
                 f"vel_head={VEL_HEAD:.0f} m/s diagonal, tail_length={TAIL_LENGTH:.0f}"),
    "dt":        DT,
    "gravity":   GRAVITY,
    "domain":    {"min": [0.0, 0.0, 0.0], "max": DOMAIN},
    "cell_size": 2.0 * RADIUS,
    "particles": particles,
}

json.dump(scene, sys.stdout, indent=2)
