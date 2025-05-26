// This file implements the GPU kernel for calculating forces between particles based on the Lennard-Jones potential.

#include "particle.h"
#include "constants.h"

__device__ void calculateLennardJonesForce(Particle* particles, int numParticles, float4* forces) {
    for (int i = 0; i < numParticles; i++) {
        float4 force = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        Particle p1 = particles[i];

        for (int j = 0; j < numParticles; j++) {
            if (i != j) {
                Particle p2 = particles[j];
                float4 r = make_float4(p2.position.x - p1.position.x,
                                        p2.position.y - p1.position.y,
                                        p2.position.z - p1.position.z,
                                        0.0f);
                float r2 = dot(r, r);
                float r6 = r2 * r2 * r2;
                float r12 = r6 * r6;

                float forceMagnitude = 24.0f * epsilon * (2.0f * sigma12 / r12 - sigma6 / r6) / r2;
                force.x += forceMagnitude * r.x;
                force.y += forceMagnitude * r.y;
                force.z += forceMagnitude * r.z;
            }
        }
        forces[i] = force;
    }
}

__global__ void computeForces(Particle* particles, int numParticles, float4* forces) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numParticles) {
        calculateLennardJonesForce(particles, numParticles, forces);
    }
}