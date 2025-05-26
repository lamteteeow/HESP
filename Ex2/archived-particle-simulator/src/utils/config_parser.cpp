#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <map>

class ConfigParser {
public:
    ConfigParser(const std::string& filename) {
        parseConfigFile(filename);
    }

    double getDouble(const std::string& key) {
        return std::stod(configData[key]);
    }

    float getFloat(const std::string& key) {
        return std::stof(configData[key]);
    }

    int getInt(const std::string& key) {
        return std::stoi(configData[key]);
    }

    std::string getString(const std::string& key) {
        return configData[key];
    }

private:
    std::map<std::string, std::string> configData;

    void parseConfigFile(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error opening config file: " << filename << std::endl;
            return;
        }

        std::string line;
        while (std::getline(file, line)) {
            if (line.empty() || line[0] == '#') {
                continue; // Skip empty lines and comments
            }
            std::istringstream iss(line);
            std::string key, value;
            if (std::getline(iss, key, '=') && std::getline(iss, value)) {
                configData[key] = value;
            }
        }
        file.close();
    }
};