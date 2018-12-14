//how to compile: nvcc mat_vec_cublas.cu -lcublas -o mat_vec_cublas
#include <iostream>
#include <cstdio>
#include <ctime>
#include <stdlib.h>
#include <math.h>
#include "cublas_v2.h"
#include <cuda_runtime.h>

int main(void)
{
	//Initialize the cuda and cublas variables
    cudaError_t cudaStat;
    cublasStatus_t stat;
    cublasHandle_t handle;
	
	//Initialize the clock to time the program
    clock_t start = clock();
	
	//Set 10000 elements in the N*N matrix and N*1 vector
    int N = 10000;

    //Initialize host variables
    float *x, *y, *a;
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

	//Initialize device variables
    float* deviceA;
    float* deviceX;
    float* deviceY;
    cudaStat = cudaMalloc((void**)&deviceA, N*N*sizeof(*a));
    cudaStat = cudaMalloc((void**)&deviceX, N*sizeof(*x));
    cudaStat = cudaMalloc((void**)&deviceY, N*sizeof(*y));
	
	//Create a cublas event and run cublas
    stat = cublasCreate(&handle);
    stat = cublasSetMatrix(N, N, sizeof(*a), a, N, deviceA, N);
    stat = cublasSetVector(N, sizeof(*x), x, 1, deviceX, 1);
    stat = cublasSetVector(N, sizeof(*y), y, 1, deviceY, 1);
    float alpha = 1.0;
    float beta = 0.0;

    stat = cublasSgemv(handle,CUBLAS_OP_N,N,N,&alpha,deviceA,N,deviceX,1,&beta,deviceY,1);

    stat = cublasGetVector(N,sizeof(*y),deviceY,1,y,1);

    //Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();
	
	//End the clock timer
    float el = float(clock() - start) / CLOCKS_PER_SEC;

	//Free device memory
    cudaFree(deviceX);
    cudaFree(deviceY);
    cudaFree(deviceA);

	//Print the results
    printf("y[0]=%8.4e and y[1]=%8.4e\n", y[0], y[1]);
    printf("Number of elements in array %8.0f\n", float(N));
    printf("Elapsed time: %8.8f seconds\n", el);

    // Free memory
    cublasDestroy(handle);
    free(a);
    free(x);
    free(y);

    return EXIT_SUCCESS;
}
