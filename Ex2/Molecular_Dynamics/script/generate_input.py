#!/usr/bin/env python3
"""
Generate input.txt file for molecular dynamics simulation
Creates particles arranged in a 3D regular grid

Usage: python generate_input.py <nx> <ny> <nz> [--gaussian]
where nx, ny, nz are the number of particles in each dimension
--gaussian: Use Gaussian distribution for velocities (default: uniform)
"""

import sys
import random
import argparse


def generate_input(nx, ny, nz, filename="input.txt", use_gaussian=False):
    """
    Generate input file with particles in 3D regular grid arrangement

    Args:
        nx, ny, nz: Number of particles in x, y, z dimensions
        filename: Output filename
        use_gaussian: Use Gaussian distribution for velocities
    """
    total_particles = nx * ny * nz

    # Simulation parameters
    dt = 0.001
    sigma = 1.0
    epsilon = 0.2

    # Grid spacing (adjust as needed)
    spacing = 2.0

    # Velocity parameters
    if use_gaussian:
        # Gaussian distribution parameters
        vel_mean = 0.0      # Mean velocity
        vel_std = 0.005     # Standard deviation
    else:
        # Uniform distribution parameters
        vel_min = -0.01
        vel_max = 0.01

    with open(filename, "w") as f:
        # Write simulation parameters
        f.write("dt\n")
        f.write(f"{dt}\n")
        f.write("sigma\n")
        f.write(f"{sigma}\n")
        f.write("epsilon\n")
        f.write(f"{epsilon}\n")

        # Write positions in regular grid
        f.write("pos\n")
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    x = i * spacing
                    y = j * spacing
                    z = k * spacing
                    f.write(f"{x:.1f} {y:.1f} {z:.1f}\n")

        # Write velocities
        f.write("velos\n")
        for i in range(total_particles):
            if use_gaussian:
                # Gaussian distribution velocities
                vx = random.gauss(vel_mean, vel_std)
                vy = random.gauss(vel_mean, vel_std)
                vz = random.gauss(vel_mean, vel_std)
            else:
                # Uniform distribution velocities
                vx = random.uniform(vel_min, vel_max)
                vy = random.uniform(vel_min, vel_max)
                vz = random.uniform(vel_min, vel_max)
            
            f.write(f"{vx:.3f} {vy:.3f} {vz:.3f}\n")

        # Write masses (uniform mass with small variation)
        f.write("masses\n")
        base_mass = 32.0
        for i in range(total_particles):
            # Add small random variation to masses
            mass = base_mass + random.uniform(-2.0, 8.0)
            f.write(f"{mass:.1f}\n")

    print(f"Generated input file '{filename}' with {total_particles} particles")
    print(f"Grid dimensions: {nx} x {ny} x {nz}")
    print(f"Grid spacing: {spacing}")
    if use_gaussian:
        print(f"Velocity distribution: Gaussian (mean={vel_mean}, std={vel_std})")
    else:
        print(f"Velocity distribution: Uniform ({vel_min} to {vel_max})")


def main():
    parser = argparse.ArgumentParser(
        description="Generate input.txt file for molecular dynamics simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate_input.py 2 3 2
  python generate_input.py 3 3 3 --gaussian
  python generate_input.py 4 2 3 --output custom_input.txt
        """
    )
    
    parser.add_argument("nx", type=int, help="Number of particles in x dimension")
    parser.add_argument("ny", type=int, help="Number of particles in y dimension")
    parser.add_argument("nz", type=int, help="Number of particles in z dimension")
    parser.add_argument("--gaussian", action="store_true", 
                       help="Use Gaussian distribution for velocities (default: uniform)")
    parser.add_argument("--output", "-o", default="input.txt",
                       help="Output filename (default: input.txt)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducible results (default: 42)")

    args = parser.parse_args()

    try:
        if args.nx <= 0 or args.ny <= 0 or args.nz <= 0:
            raise ValueError("Grid dimensions must be positive integers and there must be at least one particle in each dimension")

        # Set random seed for reproducible results
        random.seed(args.seed)

        generate_input(args.nx, args.ny, args.nz, args.output, args.gaussian)

    except ValueError as e:
        print(f"Error: {e}")
        print("Please provide positive integers for grid dimensions")
        sys.exit(1)


if __name__ == "__main__":
    main()
