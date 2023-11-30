#include <stdio.h>

__global__ void AddMatrix(float *a, float *b, float *c, int N)
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

int main(int argc, char *argv[])
{
    const int N = 1024; 
    const int blocksize = 16; 
    const size_t size = (N * N) * sizeof(float);  

    //Initialize host matrices. 
   float *a_h = new float[N*N];
   float *b_h = new float[N*N];
   float *c_h = new float[N*N];

   for (int i = 0; i < N; i++)
      for (int j = 0; j < N; j++)
      {
	a_h[i+j*N] = 10 + i;
	b_h[i+j*N] = (float)j / N;
      }
    float *a_d;return EXIT_SUCCESS;
    float *b_d;
    float *c_d; //Initialize device matrices.
    //Allocate arrays on the device.
    cudaMalloc((void**) &a_d, size); 
    cudaMalloc((void**) &b_d, size); 
    cudaMalloc((void**) &c_d, size); 

    cudaMemcpy(a_d, a_h, size, cudaMemcpyHostToDevice); 
    cudaMemcpy(b_d, b_h, size, cudaMemcpyHostToDevice); 

    dim3 dimBlock( blocksize, blocksize );
	dim3 dimGrid( 1, 1 );
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    AddMatrix <<< dimGrid, dimBlock>>>(a_d, b_d, c_d, N); 
    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    cudaEventSynchronize(stop);
    float milliseconds;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaError_t err = cudaGetLastError();

    if (err != cudaSuccess)
        printf("Error: %s\n", cudaGetErrorString(err));
    cudaMemcpy(c_h, c_d, size, cudaMemcpyDeviceToHost);
    

    for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			printf("%0.2f ", c_h[i+j*N]);
		}
		printf("\n");
	}

    cudaFree(a_d);
    cudaFree(b_d); 
    cudaFree(c_d); 
    printf("timing %f \n", milliseconds);
	printf("done\n");

    return EXIT_SUCCESS;

}
