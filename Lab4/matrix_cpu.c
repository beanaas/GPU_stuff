// Matrix addition, CPU version
// gcc matrix_cpu.c milli.c -o matrix_cpu -std=c99

#include <stdio.h>
#include "milli.h"
void add_matrix(float *a, float *b, float *c, int N)
{
	int index;

	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++)
		{
			index = i + j * N;
			c[index] = a[index] + b[index];
		}
}

int main()
{
	const int N = 512;

	float a[N * N];
	float b[N * N];
	float c[N * N];

	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++)
		{
			a[i + j * N] = 10 + i;
			b[i + j * N] = (float)j / N;
		}
	ResetMilli();
	add_matrix(a, b, c, N);
	float milliseconds = GetSeconds();
	
	/*for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			printf("%0.2f ", c[i + j * N]);
		}
		printf("\n");
	}*/
	printf("timing %f \n", milliseconds);
}
