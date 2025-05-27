#include "graphics.h"
#include "readInput.h"
#include <stdio.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

// Global variables for simulation parameters and particle data
int NUM_PARTICLES;        // Number of particles in the simulation
float epsilon, sigma, dt; // Lennard-Jones parameters (depth, size) and time step
float3* positions;
float3* velocities;
float3* forces;
float* masses;

/**
 * CUDA kernel to compute Lennard-Jones forces between all particle pairs
 * Uses the 12-6 potential: V(r) = 4ε[(σ/r)¹² - (σ/r)⁶]
 * Force is the negative gradient: F = -dV/dr
 */
__global__ void computeForces(float3 *positions,
                              float3 *forces,
                              float epsilon, // Energy depth parameter
                              float sigma,   // Size parameter
                              int num_particles)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_particles) return;

    // Initialize force accumulator for particle i
    float3 force = {0.0f, 0.0f, 0.0f};
    float3 pos_i = positions[i];
    const float sigma_sq = sigma * sigma;

    // Loop through all other particles to compute pairwise forces
    for (int j = 0; j < num_particles; ++j) {
        if (i == j) continue;

        // Calculate distance vector between particles i and j
        float3 pos_j = positions[j];
        float dx = pos_i.x - pos_j.x;
        float dy = pos_i.y - pos_j.y;
        float dz = pos_i.z - pos_j.z;

        const float dr_sq = dx * dx + dy * dy + dz * dz;

        // Optional cutoff distance for computational efficiency
        // const float cutoff = 2.5f * sigma;
        // if (dr_sq > cutoff*cutoff) continue;

        // Compute Lennard-Jones force components
        const float sigma_over_dr_sq = sigma_sq / dr_sq;
        const float sigma_over_dr6 = sigma_over_dr_sq * sigma_over_dr_sq * sigma_over_dr_sq;
        const float f = 24.0f * epsilon * (2.0f * sigma_over_dr6 * sigma_over_dr6 - sigma_over_dr6) / dr_sq;

        // Accumulate force components (F = f * unit_vector)
        force.x += f * dx;
        force.y += f * dy;
        force.z += f * dz;
    }
    // printf("f[%d]: %f/%f/%f\n", i, force.x, force.y, force.z );
    forces[i] = force;
}

/**
 * CUDA kernel to integrate particle positions using Verlet algorithm
 * Updates positions based on current velocity and acceleration
 * x(t+dt) = x(t) + v(t)*dt + 0.5*a(t)*dt²
 */
__global__ void integratePos(float3 *positions,
                             float3 *velocities,
                             float3 *forces,
                             float *masses,
                             float dt,
                             int num_particles)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_particles) return;

    // Update position using Verlet integration (F = ma, so a = F/m)
    positions[i].x += velocities[i].x * dt + 0.5f * forces[i].x * dt * dt / masses[i];
    positions[i].y += velocities[i].y * dt + 0.5f * forces[i].y * dt * dt / masses[i];
    positions[i].z += velocities[i].z * dt + 0.5f * forces[i].z * dt * dt / masses[i];
    //printf("p[%d]:, %f/%f/%f\n", i, positions[i].x, positions[i].y, positions[i].z);
}

/**
 * CUDA kernel to integrate particle velocities using Verlet algorithm
 * Updates velocities based on current acceleration
 * v(t+dt/2) = v(t) + 0.5*a(t)*dt
 */
__global__ void integrateVel(float3 *velocities,
                             float3 *forces,
                             float *masses,
                             float dt,
                             int num_particles)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_particles) return;

    // Update velocity using half-step integration (a = F/m)
    velocities[i].x += 0.5f * forces[i].x * dt / masses[i];
    velocities[i].y += 0.5f * forces[i].y * dt / masses[i];
    velocities[i].z += 0.5f * forces[i].z * dt / masses[i];
    // printf("v[%d]:, %f/%f/%f\n", i, velocities[i].x, velocities[i].y, velocities[i].z);
}

/**
 * Performs one simulation time step using the Velocity-Verlet algorithm:
 * 1. Compute forces F(t)
 * 2. Update positions: x(t+dt) = x(t) + v(t)*dt + 0.5*F(t)*dt²/m
 * 3. Update velocities (half step): v(t+dt/2) = v(t) + 0.5*F(t)*dt/m
 * 4. Compute new forces F(t+dt)
 * 5. Complete velocity update: v(t+dt) = v(t+dt/2) + 0.5*F(t+dt)*dt/m
 */
void simulateStep() {
    dim3 block(256);
    dim3 grid((NUM_PARTICLES + block.x - 1) / block.x);

    // Velocity-Verlet integration steps
    computeForces<<<grid, block>>>(positions, forces, epsilon, sigma, NUM_PARTICLES);
    integratePos<<<grid, block>>>(positions, velocities, forces, masses, dt, NUM_PARTICLES);
    integrateVel<<<grid, block>>>(velocities, forces, masses, dt, NUM_PARTICLES);
    computeForces<<<grid, block>>>(positions, forces, epsilon, sigma, NUM_PARTICLES);
    integrateVel<<<grid, block>>>(velocities, forces, masses, dt, NUM_PARTICLES);

    // Wait for all GPU operations to complete
    cudaDeviceSynchronize();
}

int main(int argc, char** argv) {
    if (argc < 2)
    {
        std::cerr << "No filename provided." << std::endl;
        return 1;
    }

    // Host arrays for initial data
    std::vector<float3> h_positions, h_velocities;
    std::vector<float> h_masses;

    // Read simulation parameters and initial conditions from file
    readInput(argv[1], epsilon, sigma, dt, h_positions, h_velocities, h_masses);

    NUM_PARTICLES = h_positions.size();

    // Allocate GPU memory for particle data
    cudaMalloc(&positions, NUM_PARTICLES*sizeof(float3));
    cudaMalloc(&velocities, NUM_PARTICLES*sizeof(float3));
    cudaMalloc(&forces, NUM_PARTICLES*sizeof(float3));
    cudaMalloc(&masses, NUM_PARTICLES*sizeof(float));

    // Copy initial data from host to device
    cudaMemcpy(positions, h_positions.data(), NUM_PARTICLES*sizeof(float3), cudaMemcpyHostToDevice);
    cudaMemcpy(velocities, h_velocities.data(), NUM_PARTICLES*sizeof(float3), cudaMemcpyHostToDevice);
    cudaMemcpy(masses, h_masses.data(), NUM_PARTICLES*sizeof(float), cudaMemcpyHostToDevice);

    // Initialize graphics system for visualization
    initGraphics();
    
    while(!windowShouldClose()) {
        for(int i=0; i<10; i++)
            simulateStep();

        // Copy current positions back to host for rendering
        std::vector<float3> curPos(NUM_PARTICLES);
        cudaMemcpy(curPos.data(), positions, NUM_PARTICLES * sizeof(float3), cudaMemcpyDeviceToHost);

        // Render current frame
        beginFrame();
        for(int i=0; i<NUM_PARTICLES; i++) {
            drawPoint(curPos[i].x / 10.0f, curPos[i].y / 10.0f);
        }
        endFrame();
    }

    cleanupGraphics();
    cudaFree(positions);
    cudaFree(velocities);
    cudaFree(forces);
    cudaFree(masses);
    return 0;
}
