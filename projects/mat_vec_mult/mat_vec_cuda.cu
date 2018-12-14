#include <iostream>
#include <cstdio>
#include <ctime>
#include <math.h>

//CUDA kernel function to add the elements of two arrays
__global__
void matvec(float *a, float *x, float *y, int n)
{
	//Set index per block and sum variable
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	float sum = 0.0f;

	//Perform the matrix-vector multiplication
	for(unsigned int i = 0; i < n; i++)
	{
		if (row < n)
		{
			sum += x[i]*a[row*n+i];
		}
	}
	
	//Synchronize the various threads working
	__syncthreads();
	
	//Put the results in the output vector
	if(row < n)
	{
		y[row] = sum;
		__syncthreads();
	}
}

int main(void)
{
	//Initializae the clock timer
	clock_t start = clock();
	
	//Set matrix and vector dimensions
	int N = 10000;

	//Allocate unified memory -- accessible from cpu or gpu
	float *x, *y, *a;
	cudaMallocManaged(&x, N*sizeof(float));
	cudaMallocManaged(&y, N*sizeof(float));
	cudaMallocManaged(&a, N*N*sizeof(float));

	// initialize x and y arrays on the host
	for (unsigned int i = 0; i < N*N; ++i)
	a[i] = 0.0;
	for (unsigned int i = 0; i < N; i++) {
		a[i*N+i] = 4.0;
	if (i > 0) a[i*N+i-1] = -1.0;
	if (i < N-1) a[i*N+i+1] = -1.0;
	x[i] = 1.0;
	y[i] = 0.0;
	}

	//Set the block size and run the kernel
	int blockSize = 1024;
	int numBlocks = (N + blockSize - 1) / blockSize;
	matvec<<<numBlocks, blockSize>>>(a, x, y, N);

	//Wait for GPU to finish before accessing on host
	cudaDeviceSynchronize();

	//Finish the timer
	double el = double(clock() - start) / CLOCKS_PER_SEC;

	//Print the results
	printf("y[0]=%8.4e and y[1]=%8.4e\n", y[0], y[1]);
	printf("Number of elements in array %8.0f\n", float(N));
	printf("Elapsed time: %8.8f seconds\n", el);

	// Free memory
	cudaFree(x);
	cudaFree(y);
	cudaFree(a);

	return 0;
}
