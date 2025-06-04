#include "graphics.h"
#include "readInput.h"
#include <stdio.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <iomanip> // for std::setprecision

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

/**
 * Export current simulation state to VTK format for visualization in ParaView
 * Creates a VTK legacy format file with particle positions and velocities
 */
void exportToVTK(const std::vector<float3> &positions,
                 const std::vector<float3> &velocities,
                 const std::vector<float> &masses,
                 int timestep)
{
    // Create filename with timestep number
    std::string filename = "./output_vtk/output_" + std::to_string(timestep) + ".vtk";
    std::ofstream file(filename);

    if (!file.is_open())
    {
        std::cerr << "Error: Could not open " << filename << " for writing" << std::endl;
        return;
    }

    // Write VTK header
    file << "# vtk DataFile Version 3.0\n";
    file << "Molecular Dynamics Simulation - Timestep " << timestep << "\n";
    file << "ASCII\n";
    file << "DATASET UNSTRUCTURED_GRID\n\n";

    // Write points (particle positions)
    file << "POINTS " << NUM_PARTICLES << " float\n";
    for (int i = 0; i < NUM_PARTICLES; i++)
    {
        file << std::fixed << std::setprecision(6)
             << positions[i].x << " "
             << positions[i].y << " "
             << positions[i].z << "\n";
    }
    file << "\n";

    // Write cells (each particle is a vertex)
    file << "CELLS " << NUM_PARTICLES << " " << NUM_PARTICLES * 2 << "\n";
    for (int i = 0; i < NUM_PARTICLES; i++)
    {
        file << "1 " << i << "\n";
    }
    file << "\n";

    // Write cell types (1 = VTK_VERTEX)
    file << "CELL_TYPES " << NUM_PARTICLES << "\n";
    for (int i = 0; i < NUM_PARTICLES; i++)
    {
        file << "1\n";
    }
    file << "\n";

    // Write point data (velocities and masses)
    file << "POINT_DATA " << NUM_PARTICLES << "\n";

    // Velocity vectors
    file << "VECTORS velocity float\n";
    for (int i = 0; i < NUM_PARTICLES; i++)
    {
        file << std::fixed << std::setprecision(6)
             << velocities[i].x << " "
             << velocities[i].y << " "
             << velocities[i].z << "\n";
    }
    file << "\n";

    // Mass scalars
    file << "SCALARS mass float 1\n";
    file << "LOOKUP_TABLE default\n";
    for (int i = 0; i < NUM_PARTICLES; i++)
    {
        file << std::fixed << std::setprecision(6) << masses[i] << "\n";
    }
    file << "\n";

    // Velocity magnitude for coloring
    file << "SCALARS velocity_magnitude float 1\n";
    file << "LOOKUP_TABLE default\n";
    for (int i = 0; i < NUM_PARTICLES; i++)
    {
        float vel_mag = sqrt(velocities[i].x * velocities[i].x +
                             velocities[i].y * velocities[i].y +
                             velocities[i].z * velocities[i].z);
        file << std::fixed << std::setprecision(6) << vel_mag << "\n";
    }

    file.close();
    std::cout << "Exported timestep " << timestep << " to " << filename << std::endl;
}

/**
 * Calculate and check force conservation using Newton's 3rd law
 * The sum of all forces should be approximately zero due to action-reaction pairs
 * Also calculates energy conservation for comparison
 */
float calculateEnergy(const std::vector<float3> &positions,
                      const std::vector<float3> &velocities,
                      const std::vector<float> &masses,
                      int timestep)
{
    // Copy current forces from GPU to host for analysis
    std::vector<float3> h_forces(NUM_PARTICLES);
    cudaMemcpy(h_forces.data(), forces, NUM_PARTICLES * sizeof(float3), cudaMemcpyDeviceToHost);

    // Calculate total force (should be ~0 by Newton's 3rd law)
    float3 total_force = {0.0f, 0.0f, 0.0f};
    for (int i = 0; i < NUM_PARTICLES; i++)
    {
        total_force.x += h_forces[i].x;
        total_force.y += h_forces[i].y;
        total_force.z += h_forces[i].z;
    }

    // Calculate magnitude of total force
    float total_force_magnitude = sqrt(total_force.x * total_force.x +
                                       total_force.y * total_force.y +
                                       total_force.z * total_force.z);

    // // Calculate individual force magnitudes for context
    // float max_force_magnitude = 0.0f;
    // float avg_force_magnitude = 0.0f;
    // for (int i = 0; i < NUM_PARTICLES; i++)
    // {
    //     float force_mag = sqrt(h_forces[i].x * h_forces[i].x +
    //                            h_forces[i].y * h_forces[i].y +
    //                            h_forces[i].z * h_forces[i].z);
    //     avg_force_magnitude += force_mag;
    //     if (force_mag > max_force_magnitude)
    //     {
    //         max_force_magnitude = force_mag;
    //     }
    // }
    // avg_force_magnitude /= NUM_PARTICLES;

    // Calculate energy for additional conservation check
    float total_kinetic_energy = 0.0f;
    float total_potential_energy = 0.0f;

    // Calculate kinetic energy: KE = 0.5 * m * v^2
    for (int i = 0; i < NUM_PARTICLES; i++)
    {
        float vel_sq = velocities[i].x * velocities[i].x +
                       velocities[i].y * velocities[i].y +
                       velocities[i].z * velocities[i].z;
        total_kinetic_energy += 0.5f * masses[i] * vel_sq;
    }

    // Calculate potential energy using Lennard-Jones potential
    for (int i = 0; i < NUM_PARTICLES; i++)
    {
        for (int j = i + 1; j < NUM_PARTICLES; j++)
        {
            float dx = positions[i].x - positions[j].x;
            float dy = positions[i].y - positions[j].y;
            float dz = positions[i].z - positions[j].z;
            float r_sq = dx * dx + dy * dy + dz * dz;

            if (r_sq > 0.0f)
            { // Avoid division by zero
                float sigma_sq = sigma * sigma;
                float sigma_over_r_sq = sigma_sq / r_sq;
                float sigma_over_r6 = sigma_over_r_sq * sigma_over_r_sq * sigma_over_r_sq;
                float sigma_over_r12 = sigma_over_r6 * sigma_over_r6;

                // V(r) = 4ε[(σ/r)¹² - (σ/r)⁶]
                total_potential_energy += 4.0f * epsilon * (sigma_over_r12 - sigma_over_r6);
            }
        }
    }

    float total_energy = total_kinetic_energy + total_potential_energy;

    // Output conservation analysis
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "=== Conservation Analysis - Timestep " << timestep << " ===" << std::endl;
    // std::cout << "  Total Force = (" << total_force.x << ", " << total_force.y << ", " << total_force.z << ")" << std::endl;
    std::cout << "  Total Force Magnitude = " << total_force_magnitude << std::endl;
    // std::cout << "  Average Individual Force = " << avg_force_magnitude << std::endl;
    // std::cout << "  Maximum Individual Force = " << max_force_magnitude << std::endl;
    // std::cout << "  Force Conservation Ratio = " << (avg_force_magnitude > 0 ? total_force_magnitude / avg_force_magnitude : 0.0f) << std::endl;

    // std::cout << "  Kinetic Energy = " << total_kinetic_energy << std::endl;
    // std::cout << "  Potential Energy = " << total_potential_energy << std::endl;
    std::cout << "  Total Energy = " << total_energy << std::endl;

    return total_energy;
}

int main(int argc, char **argv)
{
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

    int timestep = 0;
    int file_cycle_interval = 1000;
    float total_energy = 0.0f;

    while(!windowShouldClose()) {
        for (int i = 0; i < 10; i++)
        {
            simulateStep();
            timestep++;

            // Export to VTK and check energy conservation at specified intervals
            if (timestep % file_cycle_interval == 0)
            {
                // Copy current data back to host for export
                std::vector<float3> curPos(NUM_PARTICLES);
                std::vector<float3> curVel(NUM_PARTICLES);

                cudaMemcpy(curPos.data(), positions, NUM_PARTICLES * sizeof(float3), cudaMemcpyDeviceToHost);
                cudaMemcpy(curVel.data(), velocities, NUM_PARTICLES * sizeof(float3), cudaMemcpyDeviceToHost);

                // TODO: Export current state to VTK needs to be checked
                // exportToVTK(curPos, curVel, h_masses, timestep);

                // Current total energy
                float new_total_energy = calculateEnergy(curPos, curVel, h_masses, timestep);

                // Energy difference from last interval
                std::cout << "  Total Energy difference from last interval" << std::setprecision(9) << std::fixed
                          << " (timestep " << timestep << "): "
                          << new_total_energy - total_energy << std::endl;
                std::cout << "==============================================" << std::endl
                          << std::endl;

                // Update total energy for next comparison
                total_energy = new_total_energy;
            }
        }

        // Copy current positions back to host for rendering
        std::vector<float3> curPos(NUM_PARTICLES);
        cudaMemcpy(curPos.data(), positions, NUM_PARTICLES * sizeof(float3), cudaMemcpyDeviceToHost);

        // Render current frame
        beginFrame();
        for(int i=0; i<NUM_PARTICLES; i++) {
            // drawPoint(curPos[i].x / 10.0f, curPos[i].y / 10.0f);
            // drawCircle(curPos[i].x / 10.0f, curPos[i].y / 10.0f, 0.01f, 100);
            drawFilledCircle(curPos[i].x / 10.0f, curPos[i].y / 10.0f, 0.01f, 100);
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
