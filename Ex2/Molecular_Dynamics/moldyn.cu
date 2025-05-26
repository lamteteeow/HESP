#include "graphics.h"
#include "readInput.h"
#include <stdio.h>
#include <cuda_runtime.h>
#include <iostream>

int NUM_PARTICLES;
float epsilon, sigma, dt;
float3* positions;
float3* velocities;
float3* forces;
float* masses;

__global__ void computeForces(	float3* positions, 
								float3* forces,
								float epsilon,
								float sigma,
								int num_particles) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_particles) return;

    float3 force = {0.0f, 0.0f, 0.0f};
    float3 pos_i = positions[i];
    const float sigma_sq = sigma * sigma;

    for (int j = 0; j < num_particles; ++j) {
        if (i == j) continue;

        float3 pos_j = positions[j];
        float dx = pos_i.x - pos_j.x;
        float dy = pos_i.y - pos_j.y;
        float dz = pos_i.z - pos_j.z;

        const float dr_sq = dx*dx + dy*dy + dz*dz;
		//const float cutoff = 2.5f * sigma;
        //if (dr_sq > cutoff*cutoff) continue;

        const float sigma_over_dr_sq = sigma_sq / dr_sq;
        const float sigma_over_dr6 = sigma_over_dr_sq * sigma_over_dr_sq * sigma_over_dr_sq;
        const float f = 24.0f * epsilon * (2.0f * sigma_over_dr6 * sigma_over_dr6 - sigma_over_dr6) / dr_sq;

        force.x += f * dx;
        force.y += f * dy;
        force.z += f * dz;
    }
	//printf("f[%d]: %f/%f/%f\n", i, force.x, force.y, force.z );
    forces[i] = force;
}

__global__ void integratePos(float3 *positions, 
							 float3 *velocities, 
							 float3 *forces,
							 float *masses, 
							 float dt,
							 int num_particles) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_particles) return;

    positions[i].x += velocities[i].x * dt + 0.5f * forces[i].x * dt * dt / masses[i];
    positions[i].y += velocities[i].y * dt + 0.5f * forces[i].y * dt * dt / masses[i];
    positions[i].z += velocities[i].z * dt + 0.5f * forces[i].z * dt * dt / masses[i];
    //printf("p[%d]:, %f/%f/%f\n", i, positions[i].x, positions[i].y, positions[i].z);
}
__global__ void integrateVel(float3 *velocities,
							 float3 *forces,
							 float *masses,
  							 float dt,
  							 int num_particles) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_particles) return;
    	
	velocities[i].x += 0.5f * forces[i].x * dt / masses[i];	
	velocities[i].y += 0.5f * forces[i].y * dt / masses[i];
	velocities[i].z += 0.5f * forces[i].z * dt / masses[i];
	//printf("v[%d]:, %f/%f/%f\n", i, velocities[i].x, velocities[i].y, velocities[i].z);
}

void simulateStep() {
	dim3 block(256);
	dim3 grid((NUM_PARTICLES + block.x - 1) / block.x);
    computeForces<<<grid, block>>>(positions, forces, epsilon, sigma, NUM_PARTICLES);
    integratePos<<<grid, block>>>(positions, velocities, forces, masses, dt, NUM_PARTICLES);
    integrateVel<<<grid, block>>>(velocities, forces, masses, dt, NUM_PARTICLES);
    computeForces<<<grid, block>>>(positions, forces, epsilon, sigma, NUM_PARTICLES);
    integrateVel<<<grid, block>>>(velocities, forces, masses, dt, NUM_PARTICLES);
    cudaDeviceSynchronize();
}

int main(int argc, char** argv) {
	if (argc < 2) {
		std::cerr << "No filename provided." << std::endl;
		return 1;
	}
	
	std::vector<float3> h_positions, h_velocities;
	std::vector<float> h_masses;
	readInput(argv[1], epsilon, sigma, dt, h_positions, h_velocities, h_masses);

	NUM_PARTICLES = h_positions.size();;
	
    cudaMalloc(&positions, NUM_PARTICLES*sizeof(float3));
    cudaMalloc(&velocities, NUM_PARTICLES*sizeof(float3));
    cudaMalloc(&forces, NUM_PARTICLES*sizeof(float3));
    cudaMalloc(&masses, NUM_PARTICLES*sizeof(float));
    
    cudaMemcpy(positions, h_positions.data(), NUM_PARTICLES*sizeof(float3), cudaMemcpyHostToDevice);
    cudaMemcpy(velocities, h_velocities.data(), NUM_PARTICLES*sizeof(float3), cudaMemcpyHostToDevice);
    cudaMemcpy(masses, h_masses.data(), NUM_PARTICLES*sizeof(float), cudaMemcpyHostToDevice);

    initGraphics();
    
    while(!windowShouldClose()) {
        for(int i=0; i<10; i++)
        	simulateStep();
        
        float3 curPos[NUM_PARTICLES];
        cudaMemcpy(curPos, positions, NUM_PARTICLES*sizeof(float3), cudaMemcpyDeviceToHost);
        
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
