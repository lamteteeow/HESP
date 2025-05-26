#include <iostream>
#include <chrono>
#include <cuda_runtime.h>
#include "simulation.h"

__global__ void simulateParticles(Particle* particles, int numParticles, float timeStep, int numSteps) {
    for (int step = 0; step < numSteps; ++step) {
        // Calculate forces
        calculateForces(particles, numParticles);
        
        // Update positions and velocities using Velocity Verlet
        updateParticles(particles, numParticles, timeStep);
    }
}

void testPerformance(int numParticles, float timeStep, int numSteps) {
    Particle* d_particles;
    size_t size = numParticles * sizeof(Particle);
    
    cudaMalloc(&d_particles, size);
    
    // Initialize particles (this should be replaced with actual initialization logic)
    Particle* h_particles = new Particle[numParticles];
    // Fill h_particles with initial conditions...

    cudaMemcpy(d_particles, h_particles, size, cudaMemcpyHostToDevice);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    simulateParticles<<<1, 1>>>(d_particles, numParticles, timeStep, numSteps);
    cudaDeviceSynchronize();
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    
    std::cout << "Performance Test: " << numParticles << " particles, "
              << "Time taken: " << duration.count() << " seconds." << std::endl;
    
    cudaFree(d_particles);
    delete[] h_particles;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <numParticles> <timeStep> <numSteps>" << std::endl;
        return 1;
    }
    
    int numParticles = std::stoi(argv[1]);
    float timeStep = std::stof(argv[2]);
    int numSteps = std::stoi(argv[3]);
    
    testPerformance(numParticles, timeStep, numSteps);
    
    return 0;
}