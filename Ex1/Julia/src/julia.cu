#include <chrono>
#include <cassert>
#include <fstream>

#include "util.h"
#include "stream-util.h"

__global__ void julia(float2 *pos, 
					  int *mark, 
					  size_t nx, 
					  size_t nIt, 
					  float esc, 
					  float2 c) {
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= nx)
		return;
	float2 z(pos[i]);
	float2 z_sq;
	for (int iter = 0; iter < nIt; iter++) {
		z_sq.x = z.x * z.x - z.y * z.y;
		z_sq.y = 2 * z.x * z.y;
		z.x = z_sq.x + c.x;
		z.y = z_sq.y + c.y;

		if (z.x * z.x + z.y * z.y > esc * esc){
			mark[i] = iter + 1;
			break;
		}
	}
}

int main(int argc, char *argv[]) {
    
    // default values
    size_t nIt = 128;
    size_t width = 1024;
    size_t height = 1024;
    float2 c = make_float2(-0.8f,0.2f);
    float esc = 10;

    // override with command line arguments
    int i = 1;
    if (argc > i) nIt = atoi(argv[i]);
    ++i;
    if (argc > i) width = atoi(argv[i]);
    ++i;
    if (argc > i) height = atoi(argv[i]);
    ++i;
    if (argc > i) c.x = atoi(argv[i]);
    ++i;
    if (argc > i) c.y = atoi(argv[i]);
    
    size_t nx = width * height;

    int *mark;
    size_t size_mark = sizeof(int) * nx;
    cudaMallocHost(&mark, size_mark);
    float2 *pos;
    size_t size_pos = sizeof(float2) * nx;
    cudaMallocHost(&pos, size_pos);

    int *d_mark;
    cudaMalloc(&d_mark, size_mark);
    float2 *d_pos;
    cudaMalloc(&d_pos, size_pos);

	for (size_t i = 0; i < nx; ++i)
		mark[i] = -1;

	float step_x = 4.0f / static_cast<float>(width - 1);
	float step_y = 4.0f / static_cast<float>(height - 1);
	for (size_t i = 0; i < width; ++i) {
		for (size_t j = 0; j < height; ++j) {
			pos[j * width + i].x = i * step_x - 2;
			pos[j * width + i].y = j * step_y - 2;
		}
	}	

    cudaMemcpy(d_mark, mark, size_mark, cudaMemcpyHostToDevice);
    cudaMemcpy(d_pos, pos, size_pos, cudaMemcpyHostToDevice);

	auto numThreadsPerBlock = 256;
	auto numBlocks = (nx + numThreadsPerBlock - 1) / numThreadsPerBlock;

	julia<<<numBlocks, numThreadsPerBlock>>>(d_pos, d_mark, nx, nIt, esc, c);

    cudaDeviceSynchronize();

    cudaMemcpy(mark, d_mark, size_mark, cudaMemcpyDeviceToHost);

	std::ofstream pgm("julia.pgm", std::ios::binary);
    pgm << "P5\n" << width << " " << height << "\n255\n";
    
    for (size_t i = 0; i < nx; ++i) {
        unsigned char pixel;
        if (mark[i] == 0) {
            pixel = 0;  // Never escaped: black
        } else {
            if (nIt <= 1) {
                pixel = 255;  // Edge case: single iteration = white
            } else {
                // Map [1..nIt] → [255..0] (white → black)
                float ratio = static_cast<float>(nIt - mark[i]) / (nIt - 1);
                pixel = static_cast<unsigned char>(255 * ratio);
            }
        }
        pgm.write(reinterpret_cast<const char*>(&pixel), 1);
    }
    pgm.close();
    
    cudaFree(d_pos);
    cudaFree(d_mark);

    cudaFreeHost(pos);
    cudaFreeHost(mark);

    return 0;
}
