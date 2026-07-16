#ifndef VTK_OUTPUT_H
#define VTK_OUTPUT_H

#include "domain.h"
#include "vec3.cuh"
#include <string>
#include <vector>

// Write one VTK frame for a set of particles.
// gpu_owner: which GPU owns the particle (color by this for domain view).
// border: 0 = interior, 1 = at halo edge (gradient at GPU boundaries).
// Output goes to output/<scenario>_<steps>/
void writeParticlesVTK(int frame, const std::vector<Vec3> &positions,
                       const std::vector<Vec3> &velocities,
                       const std::vector<float> &radii,
                       const std::vector<int> &gpu_owner,
                       const std::vector<float> &border,
                       const std::string &scenario, long steps);

// Write domain boundaries + halo regions as a wireframe + filled strips.
// Call once during the first frame.
// region_type: 0 = domain boundary, 1 = owned split, 2 = halo strip
void writeDomainBoundaryVTK(const Vec3 domain_min, const Vec3 domain_max,
                            const std::vector<Domain> &doms,
                            const std::string &scenario, long steps);

#endif // VTK_OUTPUT_H
