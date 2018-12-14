//nvcc mat_vec_cublas.cu -lcublas -o mat_vec_cublas
#include <iostream>
#include <cstdio>
#include <ctime>
#include <stdlib.h>
#include <math.h>
#include "cublas_v2.h"
#include <cuda_runtime.h>

//CUDA kernel function to add the elements of two arrays
/*__global__
void matvec(float *a, float *x, float *y, int n)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockDim.x * blockIdx.x + threadIdx.x
	//int col = blockIdx.x * blockDim.x + threadIdx.x;
	//int tid = threadIdx.x + blockIdx.x * blockDim.x;
	float sum = 0.0;
  //for (int i = 0/; i < n; i++)
	//{
		//sum += a[row*n+i] * x[i*n + col];
	//}
	
	for(int i = 0; i < n; i++)
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
}*/

int main(void)
{
    cudaError_t cudaStat;
    cublasStatus_t stat;
    cublasHandle_t handle;
    clock_t start = clock();

    int N = 10000; // 1M elements

    //Allocate unified memory -- accessible from cpu or gpu
    float *x, *y, *a;
    /*a = new float[N*N];
    x = new float[N];
    y = new float[N];
    cudaMallocManaged(&x, N*sizeof(float));
    cudaMallocManaged(&y, N*sizeof(float));
    cudaMallocManaged(&a, N*N*sizeof(float));*/
    x = (float*)malloc(N*sizeof(float));
    y = (float*)malloc(N*sizeof(float));
    a = (float*)malloc(N*N*sizeof(float));

    // initialize x and y arrays on the host
    for (int i = 0; i < N*N; ++i)
    {
        a[i] = 0.0;
    }
    for (int i = 0; i < N; i++) {
    //x[i] = 1.0f;
    //y[i] = 2.0f;
        a[i*N+i] = 4.0;
    if (i > 0) a[i*N+i-1] = -1.0;
    if (i < N-1) a[i*N+i+1] = -1.0;
    x[i] = 1.0;
    y[i] = 0.0;
    }

    float* deviceA;
    float* deviceX;
    float* deviceY;

    cudaStat = cudaMalloc((void**)&deviceA, N*N*sizeof(*a));
    cudaStat = cudaMalloc((void**)&deviceX, N*sizeof(*x));
    cudaStat = cudaMalloc((void**)&deviceY, N*sizeof(*y));

    stat = cublasCreate(&handle);
    stat = cublasSetMatrix(N, N, sizeof(*a), a, N, deviceA, N);
    stat = cublasSetVector(N, sizeof(*x), x, 1, deviceX, 1);
    stat = cublasSetVector(N, sizeof(*y), y, 1, deviceY, 1);
    float alpha = 1.0;
    float beta = 0.0;

    stat = cublasSgemv(handle,CUBLAS_OP_N,N,N,&alpha,deviceA,N,deviceX,1,&beta,deviceY,1);

    stat = cublasGetVector(N,sizeof(*y),deviceY,1,y,1);

    // Run kernel on 1M elements on the CPU
    //int blockSize = 1024;
    //int numBlocks = (N + blockSize - 1) / blockSize;
    //matvec<<<numBlocks, blockSize>>>(a, x, y, N);

    //Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();



    float el = float(clock() - start) / CLOCKS_PER_SEC;

    //cudaFree(deviceX);
    //cudaFree(deviceY);
    //cudaFree(deviceA);

    printf("y[0]=%8.4e and y[1]=%8.4e\n", y[0], y[1]);
    printf("Number of elements in array %8.0f\n", float(N));
    printf("Elapsed time: %8.8f seconds\n", el);
    // Check for errors (all values should be 3.0f)
    /*float maxError = 0.0f;
    for (int i = 0; i < N; i++)
    maxError = fmax(maxError, fabs(y[i]-3.0f));
    std::cout << "Max error: " << maxError << std::endl;*/

    // Free memory

    cublasDestroy(handle);
    //free(a);
    //free(x);
    //free(y);

    return EXIT_SUCCESS;
}
