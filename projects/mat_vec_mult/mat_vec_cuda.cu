#include <iostream>
#include <cstdio>
#include <ctime>
#include <math.h>

//CUDA kernel function to add the elements of two arrays
__global__
void matvec(float *a, float *x, float *y, int n)
{
  int row = blockIdx.x * blockDim.x + threadIdx.x;
	//int col = blockIdx.x * blockDim.x + threadIdx.x;
	//int tid = threadIdx.x + blockIdx.x * blockDim.x;
	float sum = 0.0f;
  /*for (int i = 0/; i < n; i++)
	{
		sum += a[row*n+i] * x[i*n + col];
	}*/
	
	for(unsigned int i = 0; i < n; i++)
	{
		if (row < n)
		{
			sum += x[i]*a[row*n+i];
		}
	}
	__syncthreads();
	//y[tid] = sum;
	//y[row*n+col] = sum;
	if(row < n)
	{
		y[row] = sum;
		__syncthreads();
	}
}

int main(void)
{
	clock_t start = clock();
  int N = 100; // 1M elements
  
  //Allocate unified memory -- accessible from cpu or gpu
  float *x, *y, *a;
	/*a = new float[N*N];
	x = new float[N];
	y = new float[N];*/
  cudaMallocManaged(&x, N*sizeof(float));
  cudaMallocManaged(&y, N*sizeof(float));
	cudaMallocManaged(&a, N*N*sizeof(float));

  // initialize x and y arrays on the host
	for (unsigned int i = 0; i < N*N; ++i)
    a[i] = 0.0;
  for (unsigned int i = 0; i < N; i++) {
    //x[i] = 1.0f;
    //y[i] = 2.0f;
		a[i*N+i] = 4.0;
    if (i > 0) a[i*N+i-1] = -1.0;
    if (i < N-1) a[i*N+i+1] = -1.0;
    x[i] = 1.0;
    y[i] = 0.0;
  }

  // Run kernel on 1M elements on the CPU
  int blockSize = 1024;
  int numBlocks = (N + blockSize - 1) / blockSize;
  matvec<<<numBlocks, blockSize>>>(a, x, y, N);

  //Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();
	
	double el = double(clock() - start) / CLOCKS_PER_SEC;

	printf("y[0]=%8.4e and y[1]=%8.4e\n", y[0], y[1]);
	printf("Number of elements in array %8.0f\n", float(N));
	printf("Elapsed time: %8.8f seconds\n", el);
  // Check for errors (all values should be 3.0f)
  /*float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = fmax(maxError, fabs(y[i]-3.0f));
  std::cout << "Max error: " << maxError << std::endl;*/

  // Free memory
  cudaFree(x);
  cudaFree(y);
	cudaFree(a);

  return 0;
}
