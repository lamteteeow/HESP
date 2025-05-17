#pragma once

#include <iostream>

constexpr unsigned int streamNumWrites = 1;
// assume non-temporal stores
constexpr unsigned int streamNumReads = 1 + 0 * streamNumWrites;

void initStream(double *vec, size_t nx) {
    for (size_t i = 0; i < nx; ++i)
        vec[i] = (double) i;
}

void checkSolutionStream(const double *const vec, size_t nx, size_t nIt) {
    for (size_t i = 0; i < nx; ++i)
        if ((double) (i + nIt) != vec[i]) {
        	{
            	std::cerr << "Stream check failed for element " << i << "\n"
            	<< " Expected:  " << i + nIt << "\n"
            	<< " Actual:    " << vec[i] << "\n"
            	<< " Difference:" << (vec[i] - (i + nIt))
            	<< std::endl;
            }
            return;
        }

    std::cout << "  Passed result check" << std::endl;
}
