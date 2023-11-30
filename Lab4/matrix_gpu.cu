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
    const int N = 16;
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
    float *a_d;
    float *b_d;
    float *c_d; //Initialize device matrices.
    //Allocate arrays on the device.
    cudaMalloc((void**) &a_d, size); 
    cudaMalloc((void**) &b_d, size); 
    cudaMalloc((void**) &c_d, size); 

    cudaMemcpy(a_d, a_h, size, cudaMemcpyHostToDevice); 
    cudaMemcpy(b_d, b_h, size, cudaMemcpyHostToDevice); 

    int threadsPerBlock = 1;
    dim3 threadsPerGrid(N/threadsPerBlock, N/threadsPerBlock); 

    AddMatrix <<< threadsPerGrid, threadsPerBlock>>>(a_d, b_d, c_d, N); 
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

}
