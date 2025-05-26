#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include "particle.h"

void write_vtk(const std::string& filename, const std::vector<Particle>& particles, int timestep) {
    std::ofstream vtk_file;
    vtk_file.open(filename);

    if (!vtk_file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }

    // Write VTK header
    vtk_file << "# vtk DataFile Version 3.0" << std::endl;
    vtk_file << "Particle simulation data" << std::endl;
    vtk_file << "ASCII" << std::endl;
    vtk_file << "DATASET UNSTRUCTURED_GRID" << std::endl;

    // Write points
    vtk_file << "POINTS " << particles.size() << " float" << std::endl;
    for (const auto& particle : particles) {
        vtk_file << particle.position.x << " " << particle.position.y << " " << particle.position.z << std::endl;
    }

    // Write cells (each particle as a vertex)
    vtk_file << "CELLS " << particles.size() << " " << particles.size() * 2 << std::endl;
    for (size_t i = 0; i < particles.size(); ++i) {
        vtk_file << "1 " << i << std::endl;
    }

    vtk_file << "CELL_TYPES " << particles.size() << std::endl;
    for (size_t i = 0; i < particles.size(); ++i) {
        vtk_file << "1" << std::endl; // VTK_VERTEX
    }

    // Write point data
    vtk_file << "POINT_DATA " << particles.size() << std::endl;
    vtk_file << "SCALARS mass float 1" << std::endl;
    vtk_file << "LOOKUP_TABLE default" << std::endl;
    for (const auto& particle : particles) {
        vtk_file << particle.mass << std::endl;
    }

    vtk_file.close();
}