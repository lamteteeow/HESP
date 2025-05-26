#include <sstream>
#include <fstream>

void readInput(const std::string& filename,
               float& epsilon,
               float& sigma,
               float& dt,
               std::vector<float3>& positions,
               std::vector<float3>& velocities,
               std::vector<float>& masses) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open input file");
    }

    std::string line;
    std::string current_section;
    
    while (std::getline(file, line)) {
        // Trim whitespace
        line.erase(line.find_last_not_of(" \t\n\r") + 1);
        
        if (line.empty()) continue;
        
        if (line == "sigma" || line == "pos" || 
            line == "velos" || line == "masses" ||
            line == "epsilon" || line == "dt") {
            current_section = line;
            continue;
        }
        
        std::istringstream iss(line);
        
        if (current_section == "sigma") {
            if (!(iss >> sigma)) {
                throw std::runtime_error("Invalid sigma value");
            }
        }
        else if (current_section == "dt") {
            if (!(iss >> dt)) {
                throw std::runtime_error("Invalid dt value");
            }
        }
        else if (current_section == "epsilon") {
                    if (!(iss >> epsilon)) {
                        throw std::runtime_error("Invalid epsilon value");
                    }
                }
        else if (current_section == "pos") {
            float3 pos;
            if (!(iss >> pos.x >> pos.y >> pos.z)) {
                throw std::runtime_error("Invalid position format");
            }
            positions.push_back(pos);
        }
        else if (current_section == "velos") {
            float3 vel;
            if (!(iss >> vel.x >> vel.y >> vel.z)) {
                throw std::runtime_error("Invalid velocity format");
            }
            velocities.push_back(vel);
        }
        else if (current_section == "masses") {
            float mass;
            while (iss >> mass) {
                masses.push_back(mass);
            }
        }
    }

    // Validate input
    if (positions.size() != velocities.size() ||
        positions.size() != masses.size()) {
        throw std::runtime_error("Particle count mismatch between sections");
    }

}
