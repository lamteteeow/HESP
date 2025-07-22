//
// Created by hans on 22.06.25.
//

#ifndef INPUT_FILE_H
#define INPUT_FILE_H
#include <filesystem>
#include <vector>
#include <sstream>
#include <fstream>
#include <unordered_set>

#include "Vec3.cuh"

inline void readInput(const std::string& filename,
                      float& dt,
                      std::vector<Vec3>& positions,
                      std::vector<Vec3>& velocities,
                      std::vector<float>& masses,
                      std::vector<float>& radii,
                      Vec3& L,
                      Vec3& offset,
                      float& K,
                      float& gam,
                      Vec3& grav)
{
    std::ifstream file(filename);
    if (!file.is_open())
        throw std::runtime_error("Failed to open input file");

    std::string line;
    std::string current_section;

    while (std::getline(file, line))
    {
        // Trim whitespace
        line.erase(line.find_last_not_of(" \t\n\r") + 1);

        if (line.empty()) continue;

        static const std::unordered_set<std::string> valid_sections = {
            "pos", "velos",
            "masses", "radii", "dt",
            "L", "offset", "K", "gam", "grav"
        };

        if (valid_sections.count(line) > 0)
        {
            current_section = line;
            continue;
        }

        std::istringstream iss(line);

        if (current_section == "dt")
        {
            if (!(iss >> dt))
            {
                throw std::runtime_error("Invalid dt value");
            }
        }
        else if (current_section == "K")
        {
            if (!(iss >> K))
            {
                throw std::runtime_error("Invalid K value");
            }
        }
        else if (current_section == "gam")
        {
            if (!(iss >> gam))
            {
                throw std::runtime_error("Invalid gamma value");
            }
        }
        else if (current_section == "grav")
        {
            if (!(iss >> grav.x >> grav.y >> grav.z))
                throw std::runtime_error("Invalid gravity values");
        }
        else if (current_section == "radii")
        {
            float radius;
            while (iss >> radius)
            {
                radii.push_back(radius);
            }
        }
        else if (current_section == "pos")
        {
            Vec3 pos;
            if (!(iss >> pos.x >> pos.y >> pos.z))
            {
                throw std::runtime_error("Invalid position format");
            }
            positions.push_back(pos);
        }
        else if (current_section == "velos")
        {
            Vec3 vel;
            if (!(iss >> vel.x >> vel.y >> vel.z))
            {
                throw std::runtime_error("Invalid velocity format");
            }
            velocities.push_back(vel);
        }
        else if (current_section == "masses")
        {
            float mass;
            while (iss >> mass)
            {
                masses.push_back(mass);
            }
        }
        else if (current_section == "L")
        {
            if (!(iss >> L.x >> L.y >> L.z))
                throw std::runtime_error("Invalid length values");
        }
        else if (current_section == "offset")
        {
            if (!(iss >> offset.x >> offset.y >> offset.z))
                throw std::runtime_error("Invalid offset values");
        }
    }

    // Validate input
    if (positions.size() != velocities.size() ||
        positions.size() != masses.size())
    {
        throw std::runtime_error("Particle count mismatch between sections");
    }
}

#endif //INPUT_FILE_H
