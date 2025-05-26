#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include "../headers/particle.h"

std::vector<Particle> readParticleData(const std::string& filename) {
    std::vector<Particle> particles;
    
    // Open the file
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }

    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        Particle p;
        if (iss >> p.position.x >> p.position.y >> p.position.z >> p.velocity.x >> p.velocity.y >> p.velocity.z >> p.mass) {
            particles.push_back(p);
        } else {
            std::cerr << "Error reading particle data from line: " << line << std::endl;
        }
    }

    file.close();
    return particles;
}

void writeOutputData(const std::string& filename, const std::vector<Particle>& particles) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file for writing: " << filename << std::endl;
        return;
    }

    for (const auto& p : particles) {
        file << p.position.x << " " << p.position.y << " " << p.position.z << " "
             << p.velocity.x << " " << p.velocity.y << " " << p.velocity.z << " "
             << p.mass << std::endl;
    }

    file.close();
}