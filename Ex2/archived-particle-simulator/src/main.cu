#include <iostream>
#include <string>
#include "headers/simulation.h"

int main(int argc, char** argv) {
    // Check for the correct number of command line arguments
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <config_file>" << std::endl;
        return EXIT_FAILURE;
    }

    // Read configuration file
    std::string config_file = argv[1];
    Simulation simulation(config_file);

    // Initialize the simulation
    simulation.initialize();

    // Run the simulation loop
    simulation.run();

    // Finalize and clean up
    simulation.finalize();

    return EXIT_SUCCESS;
}