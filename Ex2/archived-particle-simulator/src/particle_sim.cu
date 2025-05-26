// This file contains the main logic for the particle simulation, including the setup of particles and the main simulation loop.

#include <iostream>
#include <vector>
#include "headers/particle.h"
#include "headers/simulation.h"
#include "utils/file_io.cpp"
#include "utils/vtk_writer.cpp"
#include "utils/config_parser.cpp"

int main(int argc, char** argv) {
    // Parse configuration file and command line arguments
    ConfigParser config("./input/config.txt"); // Provide default config file name

    // sigma = 1.0 epsilon = 1.0

    // Initialize particles from input file
    std::vector<Particle> particles = readParticleData(config.getString("particle_input_file"));

    // Create simulation object
    Simulation simulation(particles, config);
    simulation.run();

    return 0;
}