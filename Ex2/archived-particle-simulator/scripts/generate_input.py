import numpy as np

def generate_particle_data(num_particles, stable_distance, mass_range):
    positions = np.random.rand(num_particles, 3) * stable_distance
    velocities = np.random.rand(num_particles, 3) * 0.1  # Small random velocities
    masses = np.random.uniform(mass_range[0], mass_range[1], num_particles)
    
    return positions, velocities, masses

def write_to_file(filename, positions, velocities, masses):
    with open(filename, 'w') as f:
        for i in range(len(positions)):
            f.write(f"{masses[i]} {positions[i][0]} {positions[i][1]} {positions[i][2]} "
                     f"{velocities[i][0]} {velocities[i][1]} {velocities[i][2]}\n")

def main():
    num_particles = 10  # Example number of particles
    stable_distance = 10.0  # Example stable distance
    mass_range = (1.0, 5.0)  # Example mass range

    positions, velocities, masses = generate_particle_data(num_particles, stable_distance, mass_range)
    write_to_file('input/generated_particles.txt', positions, velocities, masses)

if __name__ == "__main__":
    main()