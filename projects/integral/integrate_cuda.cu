#include <iostream>
#include <stdio.h>
#include <math.h>
#include <cmath>
#include <ctime>
#include <cuda.h>
#include <cuda_runtime.h>
#include <omp.h>
#include <iomanip>

__global__ void integrate(int n, float dx, float* integral)
{
	//Set the index
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	
	//Calculate the integral and store each result at the above index
	if (index < n)
	{
		float x = (index + 0.5) * dx;
		float product = 1.0 / sqrt(1.0-(x*x));
		integral[index] = 2.0 * product * dx;
	}
}


int main(void)
{
	//Initialize the clock and set number of iterations
	clock_t start = clock();
	float N = 10000000;
	float dx = 1.0 / N;
	
	//Set block size and initialize the output array
	int blockSize = 1024;
	int numBlocks = (N + blockSize - 1) / blockSize;
	float* integral;

	//Make tge output array available both locally and on the GPU
	cudaMallocManaged(&integral, N*sizeof(float));

	//Run the kernel
	integrate<<<numBlocks, blockSize>>> (N, dx, integral);

	//Synchronize the CPU and GPU
	cudaDeviceSynchronize();

	//Compute the sum of all the elements of the output array
	float sum = 0.0;
	for (int i = 0; i < N; i++) sum += integral[i];
	
	//Finish the clock
	clock_t end = clock();
	float el = float(end-start) / CLOCKS_PER_SEC;
	
	//Print the results
	std::cout << std::setprecision(9) << "Integral: " << sum << std::endl;
	std::cout << "Number of iterations: " << N << std::endl;
	std::cout << "Elapsed time: " << el << " seconds" << std::endl;

	//Free the memory
	cudaFree(integral);

	return 0;
}
