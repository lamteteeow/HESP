//
// Created by hans on 22.06.25.
//

#ifndef NEIGHBORHOODINIT_H
#define NEIGHBORHOODINIT_H
#include <vector>


inline bool axis_indices_out_of_boundary(const int3 numCellsPerAxis, const int3 n)
{
    return n.x < 0 || n.y < 0 || n.z < 0 || n.x >= numCellsPerAxis.x || n.y >= numCellsPerAxis.y || n.z >=
        numCellsPerAxis.z;
}

inline void clamp_axis_indices(const int3 numCellsPerAxis, int3& n)
{
    if (n.x < 0) n.x += numCellsPerAxis.x;
    else if (n.x >= numCellsPerAxis.x) n.x -= numCellsPerAxis.x;
    if (n.y < 0) n.y += numCellsPerAxis.y;
    else if (n.y >= numCellsPerAxis.y) n.y -= numCellsPerAxis.y;
    if (n.z < 0) n.z += numCellsPerAxis.z;
    else if (n.z >= numCellsPerAxis.z) n.z -= numCellsPerAxis.z;
}

inline int get_cell_index_for_periodic_boundary(const int3 numCellsPerAxis, int3& n)
{
    clamp_axis_indices(numCellsPerAxis, n);
    return n.x + n.y * numCellsPerAxis.x + n.z * numCellsPerAxis.x * numCellsPerAxis.y;
}

inline int get_cell_index_for_fixed_boundary(const int3 numCellsPerAxis, const int3& n)
{
    if (axis_indices_out_of_boundary(numCellsPerAxis, n))
        return -1;
    return n.x + n.y * numCellsPerAxis.x + n.z * numCellsPerAxis.x * numCellsPerAxis.y;
}

inline void initCellNeighborhood(const int3 numCellsPerAxis, std::vector<int>& neighborhood)
{
    for (int iz = 0; iz < numCellsPerAxis.z; ++iz)
    {
        for (int iy = 0; iy < numCellsPerAxis.y; ++iy)
        {
            for (int ix = 0; ix < numCellsPerAxis.x; ++ix)
            {
                const int c = ix + iy * numCellsPerAxis.x + iz * numCellsPerAxis.x * numCellsPerAxis.y;
                const int base = c * 27;
                int idx = 0;

                for (int dz = -1; dz <= 1; ++dz)
                {
                    for (int dy = -1; dy <= 1; ++dy)
                    {
                        for (int dx = -1; dx <= 1; ++dx)
                        {
                            int3 n = {ix + dx, iy + dy, iz + dz};

                            const int neighbor = get_cell_index_for_fixed_boundary(numCellsPerAxis, n);
                            neighborhood[base + idx] = neighbor;
                            ++idx;
                        }
                    }
                }
            }
        }
    }
}

inline float sum_of_largest_two(const std::vector<float>& vec)
{
    if (vec.size() == 0) return 0.0;
    if (vec.size() == 1) return vec[0] * 2;

    float max1 = std::numeric_limits<float>::lowest();
    float max2 = std::numeric_limits<float>::lowest();

    for (const float v : vec)
    {
        if (v > max1)
        {
            max2 = max1;
            max1 = v;
        }
        else if (v > max2)
        {
            max2 = v;
        }
    }

    return max1 + max2;
}


#endif //NEIGHBORHOODINIT_H
