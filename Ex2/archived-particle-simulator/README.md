# Particle Simulator

## Overview
The Particle Simulator is a GPU-accelerated molecular dynamics simulation tool designed to model the interactions of particles in a 3D space using the Lennard-Jones potential. The simulator implements the Velocity Verlet algorithm for numerical integration and allows for the visualization of particle positions over time.

## Features
- Simulates particles in an infinite 3D space without boundary conditions.
- Initial conditions (positions, velocities, masses) can be specified via input files.
- Supports configuration through command line arguments or configuration files.
- Implements force calculations based on the Lennard-Jones potential.
- Utilizes the Velocity Verlet algorithm for particle integration.
- Outputs particle data in VTK format for visualization with third-party tools like Paraview.

## Project Structure
```
particle-simulator
├── src
│   ├── main.cu                # Entry point of the simulator
│   ├── particle_sim.cu        # Main simulation logic
│   ├── kernels
│   │   ├── force_calculation.cu # GPU kernel for force calculations
│   │   └── integration.cu      # GPU kernel for integration
│   ├── utils
│   │   ├── file_io.cpp         # File I/O operations
│   │   ├── vtk_writer.cpp      # VTK file writing
│   │   └── config_parser.cpp   # Configuration file parsing
│   └── headers
│       ├── particle.h          # Particle class definition
│       ├── simulation.h        # Simulation management
│       └── constants.h         # Constant definitions
├── input
│   ├── two_particle_stable.txt # Initial conditions for stable particles
│   ├── two_particle_attraction.txt # Initial conditions for attracting particles
│   ├── two_particle_repulsion.txt # Initial conditions for repelling particles
│   ├── collision_test.txt      # Initial conditions for collision test
│   └── config.txt              # Configuration parameters
├── output                       # Directory for output files
├── tests
│   ├── test_two_particles.cu    # Unit tests for two particle scenarios
│   └── test_performance.cu      # Performance tests
├── scripts
│   ├── generate_input.py        # Script to generate input files
│   └── visualize.py             # Script for visualizing output data
├── Makefile                     # Build instructions
└── CMakeLists.txt              # CMake configuration
```

## Installation
1. Clone the repository:
   ```
   git clone <repository-url>
   cd particle-simulator
   ```

2. Build the project using the provided Makefile or CMakeLists.txt:
   ```
   make
   ```
   or
   ```
   mkdir build
   cd build
   cmake ..
   make
   ```

## Usage
To run the simulation, execute the compiled binary with the appropriate input file and configuration parameters:
```
./particle_sim <input_file> <config_file>
```

## Visualization
After running the simulation, the output files can be visualized using Paraview or any other suitable visualization tool that supports VTK format.

## Testing
The project includes unit tests and performance tests. To run the tests, execute:
```
./test_two_particles
./test_performance
```

## Contributing
Contributions are welcome! Please submit a pull request or open an issue for any enhancements or bug fixes.

## License
This project is licensed under the MIT License. See the LICENSE file for details.