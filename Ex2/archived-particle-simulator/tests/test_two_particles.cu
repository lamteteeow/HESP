// This file contains unit tests for validating the simulation with two particles under different conditions.

#include <iostream>
#include <cassert>
#include "particle.h"
#include "simulation.h"

void testStableDistance() {
    // Initialize simulation with two particles at a stable distance
    Simulation sim("input/two_particle_stable.txt");
    sim.run(100); // Run for 100 time steps

    // Check that particles have not moved
    assert(sim.getParticle(0).position == sim.getInitialPosition(0));
    assert(sim.getParticle(1).position == sim.getInitialPosition(1));
}

void testAttraction() {
    // Initialize simulation with two particles that will attract each other
    Simulation sim("input/two_particle_attraction.txt");
    sim.run(100); // Run for 100 time steps

    // Check that particles have moved closer
    assert(sim.getParticle(0).position.z < sim.getInitialPosition(0).z);
    assert(sim.getParticle(1).position.z < sim.getInitialPosition(1).z);
}

void testRepulsion() {
    // Initialize simulation with two particles that will repel each other
    Simulation sim("input/two_particle_repulsion.txt");
    sim.run(100); // Run for 100 time steps

    // Check that particles have moved further apart
    assert(sim.getParticle(0).position.z > sim.getInitialPosition(0).z);
    assert(sim.getParticle(1).position.z > sim.getInitialPosition(1).z);
}

int main() {
    testStableDistance();
    testAttraction();
    testRepulsion();

    std::cout << "All tests passed!" << std::endl;
    return 0;
}