// Simple CUDA example by Ingemar Ragnemalm 2009. Simplest possible?
// Assigns every element in an array with its index.
// Update 2022: Changed to cudaDeviceSynchronize.

// nvcc simple.cu -L /usr/local/cuda/lib -lcudart -o simple

#include <stdio.h>

const int N = 16; 
const int blocksize = 16; 

__global__ 
void simple(float *c) 
{
	c[threadIdx.x] = sqrt((float)threadIdx.x);
}

int main()
{

	int nDevices;

	cudaGetDeviceCount(&nDevices);
	for (int i = 0; i < nDevices; i++) {
	  cudaDeviceProp prop;
	  cudaGetDeviceProperties(&prop, i);
	  printf("Device Number: %d\n", i);
	  printf("  Device name: %s\n", prop.name);
	  printf("  Cores per SM: %d\n",
			 prop.multiProcessorCount);
	  printf("  Nr of SM: %d\n",
			 prop.multiProcessorCount);
	}

	float *c = new float[N];	
	float *cd;
	const int size = N*sizeof(float);
	
	cudaMalloc( (void**)&cd, size );
	dim3 dimBlock( blocksize, 1 );
	dim3 dimGrid( 1, 1 );
	simple<<<dimGrid, dimBlock>>>(cd);
	cudaDeviceSynchronize();
	cudaMemcpy( c, cd, size, cudaMemcpyDeviceToHost ); 
	cudaFree( cd );
	
	for (int i = 0; i < N; i++)
	{
		printf("CUDA: %.17f ", c[i]);
		printf("CPU: %.17f\n", sqrt((double)i)); 	
	}
	printf("\n");
	delete[] c;
	printf("done\n");
	return EXIT_SUCCESS;
}
