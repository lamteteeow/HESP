#include <chrono>
#include <cassert>

#include "../util.h"
#include "stream-util.h"

inline void stream(size_t nx,
				   const double *__restrict__ src, 
				   double *__restrict__ dest) {
# pragma omp parallel for schedule (static)
    for (int i = 0; i < nx; ++i)
        dest[i] = src[i] + 1;
}

void initOnCPU(double *src, size_t nx, size_t nIt) {
	for (size_t i = 0; i < nx; ++i)
		src[i] = static_cast<double>(i);
}

__global__ void copyOnGPU(double *src, double *dest, size_t nx) {
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < nx)
		dest[i] = src[i] + 1;
}

void checkOnCPU(double *dest, size_t nx, size_t nIt) {
	for (size_t i = 0; i < nx; ++i)
		assert(static_cast<double>(i) + 1. == dest[i]);
}

int main(int argc, char *argv[]) {
    size_t nx, nItWarmUp, nIt;
    parseCLA_1d(argc, argv, nx, nItWarmUp, nIt);
    size_t size = sizeof(double) * nx;

    // auto src = new double[nx];
    // auto dest = new double[nx];
    double *src, *dest;
    cudaMallocHost(&src, size);
    cudaMallocHost(&dest, size);

    double *d_src, *d_dest;
    cudaMalloc(&d_src, size);
    cudaMalloc(&d_dest, size);

    // init
    initOnCPU(src, nx, nIt);

    // checkOnCPU(dest, nx, nIt);

    cudaMemcpy(d_src, src, size, cudaMemcpyHostToDevice);

	auto numThreadsPerBlock = 256;
	auto numBlocks = (nx + numThreadsPerBlock - 1) / numThreadsPerBlock;

    // warm-up
    for (int i = 0; i < nItWarmUp; ++i) {
		copyOnGPU<<<numBlocks, numThreadsPerBlock>>>(d_src, d_dest, nx);
        std::swap(d_src, d_dest);
        cudaDeviceSynchronize();
    }

    // measurement
    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < nIt; ++i) {
		copyOnGPU<<<numBlocks, numThreadsPerBlock>>>(d_src, d_dest, nx);
        std::swap(d_src, d_dest);
        cudaDeviceSynchronize();
    }

    auto end = std::chrono::steady_clock::now();

    cudaDeviceSynchronize();

    cudaMemcpy(dest, d_src, size, cudaMemcpyDeviceToHost);

    printStats(end - start, nx, nIt, streamNumReads, streamNumWrites);

    // check solution
    checkSolutionStream(dest, nx, nIt + nItWarmUp);
    
    cudaFree(d_src);
    cudaFree(d_dest);

    cudaFreeHost(src);
    cudaFreeHost(dest);

    return 0;
}
