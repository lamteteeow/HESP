// This file implements the GPU kernel for the Velocity Verlet algorithm to update particle positions and velocities.

#include "particle.h"
#include "constants.h"

__global__ void velocityVerletKernel(Particle* particles, int numParticles, float dt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < numParticles) {
        Particle& p = particles[idx];

        // Update positions
        p.position.x += p.velocity.x * dt + 0.5f * p.acceleration.x * dt * dt;
        p.position.y += p.velocity.y * dt + 0.5f * p.acceleration.y * dt * dt;
        p.position.z += p.velocity.z * dt + 0.5f * p.acceleration.z * dt * dt;

        // Update velocities
        p.velocity.x += 0.5f * p.acceleration.x * dt;
        p.velocity.y += 0.5f * p.acceleration.y * dt;
        p.velocity.z += 0.5f * p.acceleration.z * dt;
    }
}

__global__ void updateVelocitiesKernel(Particle* particles, int numParticles, float dt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < numParticles) {
        Particle& p = particles[idx];

        // Update velocities with the new acceleration
        p.velocity.x += p.acceleration.x * dt;
        p.velocity.y += p.acceleration.y * dt;
        p.velocity.z += p.acceleration.z * dt;
    }
}

void velocityVerlet(Particle* particles, int numParticles, float dt) {
    int blockSize = 256;
    int numBlocks = (numParticles + blockSize - 1) / blockSize;

    // Update positions and velocities
    velocityVerletKernel<<<numBlocks, blockSize>>>(particles, numParticles, dt);
    cudaDeviceSynchronize();

    // Update velocities with the new acceleration
    updateVelocitiesKernel<<<numBlocks, blockSize>>>(particles, numParticles, dt);
    cudaDeviceSynchronize();
}