//
// Created by hans on 08.07.25.
//

#ifndef READ_INPUT_H
#define READ_INPUT_H
#include "json.hpp"
#include <fstream>
#include <string>
#include <iostream>
#include "SceneHost.h"

inline SceneHost loadSceneHost(const std::string& filename) {
    std::ifstream in{filename};
    if (!in) throw std::runtime_error("Cannot open " + filename);
    nlohmann::json j;
    in >> j;

    SceneHost scene;
    scene.load(j);
    return scene;
}
#endif //READ_INPUT_H
