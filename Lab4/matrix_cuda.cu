// Simple CUDA example by Ingemar Ragnemalm 2009. Simplest possible?
// Assigns every element in an array with its index.
// Update 2022: Changed to cudaDeviceSynchronize.

// nvcc simple.cu -L /usr/local/cuda/lib -lcudart -o simple

#include <stdio.h>

const int N = 1024;
const int block_size = 32; 

__global__ 
void add_matrix(float *ad, float *bd, float *cd) 
{
	int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int i,j,idx;
    i = blockIdx.x + blockIdx.y * gridDim.x;
	int idx = blockId * (blockDim.x * blockDim.y)+ (threadIdx.y * blockDim.x) + threadIdx.x;

	cd[idx] = ad[idx] + bd[idx];
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
    
	//dim3 dimGrid(N/dimBlock.x, N/dimBlock.y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
	add_matrix<<<dimGrid, dimBlock>>>(ad, bd, cd);
	cudaEventRecord(stop);
    cudaDeviceSynchronize();
    cudaEventSynchronize(stop);
    float milliseconds;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaError_t err = cudaGetLastError();

    if (err != cudaSuccess)
        printf("Error: %s\n", cudaGetErrorString(err));
    
	cudaMemcpy( c, cd, size, cudaMemcpyDeviceToHost ); 
	cudaFree( cd );

    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, 0);  // Assuming device 0

    printf("Max grid dimensions: (%d, %d, %d)\n", props.maxGridSize[0], props.maxGridSize[1], props.maxGridSize[2]);
     printf("Maximum threads per block: %d\n", props.maxThreadsPerBlock);

    
	
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			printf("%0.2f ", c[i + j * N]);
		}
		printf("\n");
	}
	delete[] c;
    delete[] b;
    delete[] a;
    printf("timing %f \n", milliseconds);
	printf("done\n");
	return EXIT_SUCCESS;
}
