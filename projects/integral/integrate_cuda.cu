#include <iostream>
#include <stdio.h>
#include <math.h>
#include <cmath>
#include <ctime>
#include <cuda.h>
#include <cuda_runtime.h>
#include <omp.h>
#include <iomanip>

__global__ void integrate(int n, double dx, double* integral)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
  //int stride = blockDim.x * gridDim.x;
	//int chunk = n / blockIdx.x;
	//int start = threadIdx.x * chunk;
	//int end = (threadIdx.x + 1) * chunk;
	//printf("Hello");
	//float dx = 1.0 / float(n);
	//for (int i = index; i < n; i++)
	if (index < n)
	{
		double x = (index + 0.5) * dx;
		double product = 1.0 / sqrt(1.0-(x*x));
		//printf("x %f product %f", x, product);
		integral[index] = 2.0 * product * dx;
	}
	//integral = integral * 2;

	//if (threadIdx.x == 0)
    //    atomicAdd(integral + iter_num, block_ressult);
}


int main(void)
{
	clock_t start = clock();
	double N = 10000000;
	double dx = 1.0 / N;
	//cudaError_t errorcode = cudaSuccess;
	
	//int size = N*sizeof(float);
	int blockSize = 1024;
	int numBlocks = (N + blockSize - 1) / blockSize;
	double* integral;
	//int stride;

	cudaMallocManaged(&integral, N*sizeof(double));
	//cudaMallocManaged(&data, n * sizeof(int));

	integrate<<<numBlocks, blockSize>>> (N, dx, integral);

	
	
	cudaDeviceSynchronize();

	double sum = 0.0;
	for (int i = 0; i < N; i++) sum += integral[i];
	
	clock_t end = clock();
	float el = float(end-start) / CLOCKS_PER_SEC;
	std::cout << std::setprecision(9) << "Integral: " << sum << std::endl;
	std::cout << "Number of iterations: " << N << std::endl;
	std::cout << "Elapsed time: " << el << " seconds" << std::endl;

	cudaFree(integral);

	return 0;
}
