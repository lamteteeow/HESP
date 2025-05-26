#ifndef SIMULATION_H
#define SIMULATION_H

#include "particle.h"
#include <vector>
#include <string>

class Simulation {
public:
    Simulation(const std::string& config);
    void initialize();
    void run(){
        // // Main simulation loop
        // for (float step = 0; step < config.getFloat("number_of_time_steps"); step = step + config.getFloat("time_step_length"))
        // {
        //     if (step % config.getInt("output_interval") == 0)
        //     {
        //         write_vtk(config.getString("output_file"), particles, step);
        //     }
        // }
    };
    void finalize();

private:
    std::vector<Particle> particles;
    float timeStep;
    int numTimeSteps;
    float sigma;
    float epsilon;

    void readInputFile(const std::string& filename);
    void writeOutputFile(const std::string& filename);
    void calculateForces();
    void integrate();
};

#endif // SIMULATION_H