// Simple CUDA example by Ingemar Ragnemalm 2009. Simplest possible?
// Assigns every element in an array with its index.
// Update 2022: Changed to cudaDeviceSynchronize.

// nvcc simple.cu -L /usr/local/cuda/lib -lcudart -o simple

#include <stdio.h>
#include "milli.h"
const int N = 4096;
const int block_size = 16; 

void add_matrix_cpu(float *a, float *b, float *c, int N)
{
	int index;

	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++)
		{
			index = i + j * N;
			c[index] = a[index] + b[index];
		}
}

__global__ 
void add_matrix_gpu(float *a, float *b, float *c) 
{
	int rowIdx = blockIdx.y * blockDim.y + threadIdx.y;
    int colIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int elemIdx;
    if(rowIdx < N && colIdx < N)
    {
        elemIdx = colIdx + rowIdx * N; 
        c[elemIdx] = a[elemIdx] + b[elemIdx]; 
    }
}

__global__ 
void add_matrix_gpu_bad(float *a, float *b, float *c) 
{
	int rowIdx = blockIdx.y * blockDim.y + threadIdx.y;
    int colIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int elemIdx;
    if(rowIdx < N && colIdx < N)
    {
        elemIdx = colIdx * N + rowIdx ; 
        c[elemIdx] = a[elemIdx] + b[elemIdx]; 
    }
}



int main()
{
    float *a, *b, *c;
    float *ad, *bd, *cd;
    const int size = N * N * sizeof(float);

    // Allocate memory on the host
    a = new float[N * N];
    b = new float[N * N];
    c = new float[N * N];

    for (int i = 0; i < N; i++){
		for (int j = 0; j < N; j++)
		{
			a[i + j * N] = 10 + i;
			b[i + j * N] = (float)j / N;
		}
    }
	cudaMalloc( (void**)&ad, size );
    cudaMalloc( (void**)&bd, size );
    cudaMalloc( (void**)&cd, size );
    cudaMemcpy( ad, a, size, cudaMemcpyHostToDevice ); 
    cudaMemcpy( bd, b, size, cudaMemcpyHostToDevice );

	dim3 dimBlock( block_size, block_size );
    dim3 dimGrid(N/block_size, N/block_size);
    

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    //cudaThreadSynchronize();
    //ResetMilli();
	add_matrix_gpu<<<dimGrid, dimBlock>>>(ad, bd, cd);
    //cudaDeviceSynchronize();
    //float milliseconds_gpu = GetSeconds();
	cudaEventRecord(stop);
    cudaDeviceSynchronize();
    cudaEventSynchronize(stop);
    float milliseconds_gpu;
    cudaEventElapsedTime(&milliseconds_gpu, start, stop);
    printf("timing GPU coalescing: %f \n", milliseconds_gpu/1000);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("Error: %s\n", cudaGetErrorString(err));
    
	cudaMemcpy( c, cd, size, cudaMemcpyDeviceToHost ); 

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
	add_matrix_gpu<<<dimGrid, dimBlock>>>(ad, bd, cd);
	cudaEventRecord(stop);
    cudaDeviceSynchronize();
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds_gpu, start, stop);
    printf("timing GPU bad coalescing: %f \n", milliseconds_gpu/1000);
    err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("Error: %s\n", cudaGetErrorString(err));
	
	/*for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			printf("%0.2f ", c[i + j * N]);
		}
		printf("\n");
	}*/

    for (int i = 0; i < N; i++){
		for (int j = 0; j < N; j++)
		{
			a[i + j * N] = 10 + i;
			b[i + j * N] = (float)j / N;
		}
    }

    ResetMilli();
	add_matrix_cpu(a, b, c, N);
	float milliseconds_cpu = GetSeconds();
    printf("timing CPU: %f \n", milliseconds_cpu);

	delete[] c;
    delete[] b;
    

    delete[] a;
    cudaFree(ad);
    cudaFree(bd);
    cudaFree(cd);
    
	printf("done\n");
	return EXIT_SUCCESS;
}
