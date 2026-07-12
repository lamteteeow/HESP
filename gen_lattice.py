"""
gen_lattice.py — generate a 2D hexagonal lattice scene for md2d.

Usage:
    python gen_lattice.py [options] > scenes/lattice.json

Options (edit the CONFIG section below):
    NX, NY    — lattice size in x and y
    radius    — particle radius
    spacing   — center-to-center distance (>= 2*radius for no initial overlap)
    mass      — particle mass
    kn        — normal spring stiffness
    gamma_n   — normal damping coefficient
    gamma_t   — tangential damping coefficient
    mu        — friction coefficient
    T         — initial temperature (sets random velocity magnitude)
"""

import json
import math
import random

# ── CONFIG ─────────────────────────────────────────────────────────────────
NX      = 20        # particles in x
NY      = 20        # particles in y
radius  = 0.4
spacing = 0.9       # must be > 2*radius for no initial overlap
mass    = 1.0
kn      = 5000.0
gamma_n = 20.0
gamma_t = 10.0
mu      = 0.3
T       = 0.5       # velocity scale (thermal energy per particle ≈ 0.5*m*v^2)
dt      = 5e-5
gravity = [0.0, 0.0, 0.0]
# ───────────────────────────────────────────────────────────────────────────

random.seed(42)

particles = []
for iy in range(NY):
    for ix in range(NX):
        # Hexagonal offset every other row
        x = ix * spacing + (0.5 * spacing if iy % 2 else 0.0) + radius
        y = iy * spacing * math.sqrt(3) / 2 + radius
        z = 0.0

        # Random velocity drawn from Maxwell-Boltzmann-like distribution
        vx = random.gauss(0, math.sqrt(T / mass))
        vy = random.gauss(0, math.sqrt(T / mass))

        particles.append({
            "position":  [round(x, 6), round(y, 6), z],
            "velocity":  [round(vx, 6), round(vy, 6), 0.0],
            "radius":    radius,
            "mass":      mass,
            "kn":        kn,
            "gamma_n":   gamma_n,
            "gamma_t":   gamma_t,
            "mu":        mu,
        })

# Compute domain from particle positions
xs = [p["position"][0] for p in particles]
ys = [p["position"][1] for p in particles]
domain_min = [0.0, 0.0, 0.0]
domain_max = [round(max(xs) + spacing, 4), round(max(ys) + spacing, 4), 0.0]

scene = {
    "dt":       dt,
    "gravity":  gravity,
    "domain":   {"min": domain_min, "max": domain_max},
    "cell_size": spacing,
    "particles": particles,
}

print(json.dumps(scene, indent=4))
