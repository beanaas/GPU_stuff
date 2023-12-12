// Lab 5, image filters with CUDA.

// Compile with a command-line similar to Lab 4:
// nvcc filter.cu -c -arch=sm_30 -o filter.o
// g++ filter.o milli.c readppm.c -lGL -lm -lcuda -lcudart -L/usr/local/cuda/lib -lglut -o filter
// or (multicore lab)
// nvcc filter.cu -c -arch=sm_20 -o filter.o
// g++ filter.o milli.c readppm.c -lGL -lm -lcuda -L/usr/local/cuda/lib64 -lcudart -lglut -o filter

// 2017-11-27: Early pre-release, dubbed "beta".
// 2017-12-03: First official version! Brand new lab 5 based on the old lab 6.
// Better variable names, better prepared for some lab tasks. More changes may come
// but I call this version 1.0b2.
// 2017-12-04: Two fixes: Added command-lines (above), fixed a bug in computeImages
// that allocated too much memory. b3
// 2017-12-04: More fixes: Tightened up the kernel with edge clamping.
// Less code, nicer result (no borders). Cleaned up some messed up X and Y. b4
// 2022-12-07: A correction for a deprecated function.
//for running limited filter
//nvcc -D SINGLE filter.cu -c -arch=sm_30 -o filter.o && g++ -o filter filter.o milli.c readppm.c -lGL -lm -lGLU -lglut -lcuda -lcudart -L/usr/local/cuda/lib && ./filter
//runnning regular
//nvcc filter.cu -c -arch=sm_30 -o filter.o && g++ -o filter filter.o milli.c readppm.c -lGL -lm -lGLU -lglut -lcuda -lcudart -L/usr/local/cuda/lib && ./filter
//nvcc -D BOX_FILTER filter.cu -c -arch=sm_30 -o filter.o && g++ -o filter filter.o milli.c readppm.c -lGL -lm -lGLU -lglut -lcuda -lcudart -L/usr/local/cuda/lib && ./filter
//nvcc -D SEPARABLE filter.cu -c -arch=sm_30 -o filter.o && g++ -o filter filter.o milli.c readppm.c -lGL -lm -lGLU -lglut -lcuda -lcudart -L/usr/local/cuda/lib && ./filter
//nvcc -D GAUSS filter.cu -c -arch=sm_30 -o filter.o && g++ -o filter filter.o milli.c readppm.c -lGL -lm -lGLU -lglut -lcuda -lcudart -L/usr/local/cuda/lib && ./filter
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#ifdef __APPLE__
  #include <GLUT/glut.h>
  #include <OpenGL/gl.h>
#else
  #include <GL/glut.h>
#endif
#include "readppm.h"
#include "milli.h"

// maxkernel is Maximum Shared Memory Per Block: 49152 = [BLOCK_SIZE+2*maxKernelSizeX][(BLOCK_SIZE+2*maxKernelSizeY)*3]
#define maxKernelSizeX 48
#define maxKernelSizeY 48
#define BLOCK_SIZE 32

__global__ void filter(unsigned char *image, unsigned char *out, const unsigned int imagesizex, const unsigned int imagesizey, const int kernelsizex, const int kernelsizey)
{ 
  // map from blockIdx to pixel position
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

  int dy, dx;
  unsigned int sumx, sumy, sumz;

  int divby = (2*kernelsizex+1)*(2*kernelsizey+1); // Works for box filters only!
	
	if (x < imagesizex && y < imagesizey) // If inside image
	{
// Filter kernel (simple box filter)
	sumx=0;sumy=0;sumz=0;
	for(dy=-kernelsizey;dy<=kernelsizey;dy++)
		for(dx=-kernelsizex;dx<=kernelsizex;dx++)	
		{
			// Use max and min to avoid branching!
			int yy = min(max(y+dy, 0), imagesizey-1);
			int xx = min(max(x+dx, 0), imagesizex-1);
			
			sumx += image[((yy)*imagesizex+(xx))*3+0];
			sumy += image[((yy)*imagesizex+(xx))*3+1];
			sumz += image[((yy)*imagesizex+(xx))*3+2];
		}
	out[(y*imagesizex+x)*3+0] = sumx/divby;
	out[(y*imagesizex+x)*3+1] = sumy/divby;
	out[(y*imagesizex+x)*3+2] = sumz/divby;
	}
}

__global__ void box_filter(unsigned char *image, unsigned char *out, const unsigned int imagesizex, const unsigned int imagesizey, const int kernelsizex, const int kernelsizey)
{ 	int size = BLOCK_SIZE+2*kernelsizey;
	
	__shared__ unsigned char shared_data[BLOCK_SIZE+2*maxKernelSizeX][(BLOCK_SIZE+2*maxKernelSizeY)*3]; // multiplied by three because pixel

	int dy, dx;
	unsigned int sumx, sumy, sumz;

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	//coalesced and non coalesced, true = coalesced faster when many reads from global

	for(int yi =threadIdx.y; yi<BLOCK_SIZE+2*kernelsizey; yi += blockDim.y){
		for(int xi = threadIdx.x; xi<BLOCK_SIZE+2*kernelsizex; xi += blockDim.x){
			//global indexes
			int clmp_xi = min(max(blockIdx.x * blockDim.x + xi - kernelsizex, 0), imagesizex-1);
			int clmp_yi = min(max(blockIdx.y * blockDim.y + yi - kernelsizey, 0), imagesizey-1);
		
			int img_idx = clmp_yi * imagesizex + clmp_xi;
			
			shared_data[yi][xi*3+0] = image[img_idx*3+0];
			shared_data[yi][xi*3+1] = image[img_idx*3+1];
			shared_data[yi][xi*3+2] = image[img_idx*3+2];
		}
	}
	

	__syncthreads();

	int divby = (2*kernelsizex+1)*(2*kernelsizey+1);
	//only threads inside "blockimage" should have output
	if (x < imagesizex && y < imagesizey) // If inside image
	{
		sumx=0;sumy=0;sumz=0;

		for(dx=0;dx<=kernelsizex*2;dx++){
			for(dy=0;dy<=kernelsizey*2;dy++)	
			{
				int yy = dy + threadIdx.y;
				int xx = dx + threadIdx.x;

				sumx += shared_data[yy][xx*3 + 0];
				sumy += shared_data[yy][xx*3 + 1];
				sumz += shared_data[yy][xx*3 + 2];
			}
		}
		out[(y * imagesizex + x)*3+0] = sumx/divby;
		out[(y * imagesizex + x)*3+1] =sumy/divby;
		out[(y * imagesizex + x)*3+2] =sumz/divby;
		}
		
	}

__global__ void gauss_filter(unsigned char *image, unsigned char *out, const unsigned int imagesizex, const unsigned int imagesizey, const int kernelsizex, const int kernelsizey)
{ 	
	__shared__ unsigned char shared_data[BLOCK_SIZE+2*maxKernelSizeX][(BLOCK_SIZE+2*maxKernelSizeY)*3]; // multiplied by three because pixels

	int dy, dx;
	unsigned int sumx, sumy, sumz;

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	//coalesced and non coalesced, true = coalesced faster when many reads from global
	for(int yi =threadIdx.y; yi<BLOCK_SIZE+2*kernelsizey; yi += blockDim.y){
		for(int xi = threadIdx.x; xi<BLOCK_SIZE+2*kernelsizex; xi += blockDim.x){
			//global indexes
			int clmp_xi = min(max(blockIdx.x * blockDim.x + xi - kernelsizex, 0), imagesizex-1);
			int clmp_yi = min(max(blockIdx.y * blockDim.y + yi - kernelsizey, 0), imagesizey-1);
		
			int img_idx = clmp_yi * imagesizex + clmp_xi;
			
			shared_data[yi][xi*3+0] = image[img_idx*3+0];
			shared_data[yi][xi*3+1] = image[img_idx*3+1];
			shared_data[yi][xi*3+2] = image[img_idx*3+2];
		}
	}
	
	__syncthreads();

	int weights[5] = { 1, 4, 6, 4, 1};
	int divby = 16;
	//only threads inside "blockimage" should have output
	if (x < imagesizex && y < imagesizey) // If inside image
	{
		sumx=0;sumy=0;sumz=0;

		for(dx=0;dx<=kernelsizex*2;dx++){
			for(dy=0;dy<=kernelsizey*2;dy++)	
			{
				int yy = dy + threadIdx.y;
				int xx = dx + threadIdx.x;

				int i = (kernelsizex>kernelsizey) ? dx:dy;
				sumx += shared_data[yy][xx*3+0] * weights[i];
				sumy += shared_data[yy][xx*3+1] * weights[i];
				sumz += shared_data[yy][xx*3+2] * weights[i];
			}
				
		}
		out[(y * imagesizex + x)*3+0] = sumx/divby;
		out[(y * imagesizex + x)*3+1] =sumy/divby;
		out[(y * imagesizex + x)*3+2] =sumz/divby;
		}
		
}

__global__ void median_filter2(unsigned char *image, unsigned char *out, const unsigned int imagesizex, const unsigned int imagesizey, const int kernelsizex, const int kernelsizey)
{   
    __shared__ unsigned char shared_data[BLOCK_SIZE+2*maxKernelSizeX][(BLOCK_SIZE+2*maxKernelSizeY)*3]; // multiplied by three because pixels

    int dy, dx;
    unsigned int sumx, sumy, sumz;

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // coalesced and non-coalesced, true = coalesced faster when many reads from global
    for(int yi = threadIdx.y; yi < BLOCK_SIZE+2*kernelsizey; yi += blockDim.y){
        for(int xi = threadIdx.x; xi < BLOCK_SIZE+2*kernelsizex; xi += blockDim.x){
            // global indexes
            int clmp_xi = min(max(blockIdx.x * blockDim.x + xi - kernelsizex, 0), imagesizex-1);
            int clmp_yi = min(max(blockIdx.y * blockDim.y + yi - kernelsizey, 0), imagesizey-1);
            
            int img_idx = clmp_yi * imagesizex + clmp_xi;
            
            shared_data[yi][xi*3+0] = image[img_idx*3+0];
            shared_data[yi][xi*3+1] = image[img_idx*3+1];
            shared_data[yi][xi*3+2] = image[img_idx*3+2];
        }
    }
    
    __syncthreads();

    // only threads inside "blockimage" should have output
    if (x < imagesizex && y < imagesizey) // If inside image
    {
        sumx=0; sumy=0; sumz=0;
        unsigned char red_values[(2*maxKernelSizeX + 1) * (2*maxKernelSizeY + 1)];
        unsigned char green_values[(2*maxKernelSizeX + 1) * (2*maxKernelSizeY + 1)];
        unsigned char blue_values[(2*maxKernelSizeX + 1) * (2*maxKernelSizeY + 1)];

        int index = 0;

        // Collect pixel values in the neighborhood
        for(dx = 0; dx <= kernelsizex*2; dx++){
            for(dy = 0; dy <= kernelsizey*2; dy++) {
                int yy = dy + threadIdx.y;
                int xx = dx + threadIdx.x;
                red_values[index] = shared_data[yy][xx*3+0];
                green_values[index] = shared_data[yy][xx*3+1];
                blue_values[index] = shared_data[yy][xx*3+2];
                index++;
            }
        }

        // Bubble Sort the pixel values
        for (int i = 0; i < index - 1; i++) {
            for (int j = 0; j < index - i - 1; j++) {
                if (red_values[j] > red_values[j+1]) {
                    // Swap values
                    unsigned char temp = red_values[j];
                    red_values[j] = red_values[j+1];
                    red_values[j+1] = temp;
                }
                if (green_values[j] > green_values[j+1]) {
                    // Swap values
                    unsigned char temp = green_values[j];
                    green_values[j] = green_values[j+1];
                    green_values[j+1] = temp;
                }
                if (blue_values[j] > blue_values[j+1]) {
                    // Swap values
                    unsigned char temp = blue_values[j];
                    blue_values[j] = blue_values[j+1];
                    blue_values[j+1] = temp;
                }
            }
        }

        // Find the median value for each channel
        out[(y * imagesizex + x)*3 + 0] = red_values[index/2];
        out[(y * imagesizex + x)*3 + 1] = green_values[index/2];
        out[(y * imagesizex + x)*3 + 2] = blue_values[index/2];
    }   
}

__global__ void median_filter(unsigned char *image, unsigned char *out, const unsigned int imagesizex, const unsigned int imagesizey, const int kernelsizex, const int kernelsizey)
{   
    __shared__ unsigned char shared_data[BLOCK_SIZE+2*maxKernelSizeX][(BLOCK_SIZE+2*maxKernelSizeY)*3]; // multiplied by three because pixels

    int dy, dx;
    unsigned int sumx, sumy, sumz;

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // coalesced and non-coalesced, true = coalesced faster when many reads from global
    for(int yi = threadIdx.y; yi < BLOCK_SIZE+2*kernelsizey; yi += blockDim.y){
        for(int xi = threadIdx.x; xi < BLOCK_SIZE+2*kernelsizex; xi += blockDim.x){
            // global indexes
            int clmp_xi = min(max(blockIdx.x * blockDim.x + xi - kernelsizex, 0), imagesizex-1);
            int clmp_yi = min(max(blockIdx.y * blockDim.y + yi - kernelsizey, 0), imagesizey-1);
            
            int img_idx = clmp_yi * imagesizex + clmp_xi;
            
            shared_data[yi][xi*3+0] = image[img_idx*3+0];
            shared_data[yi][xi*3+1] = image[img_idx*3+1];
            shared_data[yi][xi*3+2] = image[img_idx*3+2];
        }
    }
    
    __syncthreads();

    // only threads inside "blockimage" should have output
    if (x < imagesizex && y < imagesizey) // If inside image
    {
        sumx=0; sumy=0; sumz=0;
        unsigned int medianr[256] = {0};  // Histograms for each channel
        unsigned int mediang[256] = {0};
        unsigned int medianb[256] = {0};

        for(dx = 0; dx <= kernelsizex*2; dx++){
            for(dy = 0; dy <= kernelsizey*2; dy++) {
                int yy = dy + threadIdx.y;
                int xx = dx + threadIdx.x;
                medianr[shared_data[yy][xx*3+0]] += 1;
                mediang[shared_data[yy][xx*3+1]] += 1;
                medianb[shared_data[yy][xx*3+2]] += 1;
            }
        }

        int sum = 0;
        int i = 0;
		int len = ((2*kernelsizex)*(2*kernelsizey))/2;

        // Find the median value for each channel
        while(sum < len) {
            sum += medianr[i];
            i++;
        }
        out[(y * imagesizex + x)*3 + 0] = i;

        sum = 0;
        i = 0;
        while(sum < len) {
            sum += mediang[i];
            i++;
        }
        out[(y * imagesizex + x)*3 + 1] = i;

        sum = 0;
        i = 0;
        while(sum < len) {
            sum += medianb[i];
            i++;
        }
        out[(y * imagesizex + x)*3 + 2] = i;
    }   
}



#ifdef SINGLE
#define BLOCK_SIZE 32
//cant run when kernelsize exceds blocksize
__global__ void box_filter2(unsigned char *image, unsigned char *out, const unsigned int imagesizex, const unsigned int imagesizey, const int kernelsizex, const int kernelsizey)
{ 
	__shared__ unsigned char shared_data[BLOCK_SIZE][BLOCK_SIZE*3]; // multiplied by three because pixels

	int dy, dx;
	unsigned int sumx, sumy, sumz;

	int internal_block_sizex = blockDim.x - 2*kernelsizex;
	int internal_block_sizey = blockDim.y - 2*kernelsizey;

	int x = blockIdx.x*internal_block_sizex + threadIdx.x - kernelsizex;
	int y = blockIdx.y*internal_block_sizey + threadIdx.y - kernelsizey;

	x = min(max(x, 0), imagesizex-1);
	y = min(max(y, 0), imagesizey-1);

	int img_idx = y * imagesizex + x;

	shared_data[threadIdx.y][threadIdx.x*3+0] = image[img_idx*3+0];
	shared_data[threadIdx.y][threadIdx.x*3+1] = image[img_idx*3+1];
	shared_data[threadIdx.y][threadIdx.x*3+2] = image[img_idx*3+2];
	
	__syncthreads();

	int divby = (2*kernelsizex+1)*(2*kernelsizey+1);
	//only threads inside "blockimage" should have output
	if (threadIdx.x >= kernelsizex && threadIdx.x < (blockDim.x - kernelsizex) &&
        threadIdx.y >= kernelsizey && threadIdx.y < (blockDim.y - kernelsizey)) {
		sumx=0;sumy=0;sumz=0;

		for(dy=-kernelsizey;dy<=kernelsizey;dy++){
			for(dx=-kernelsizex;dx<=kernelsizex;dx++)	
			{
				int yy = dy + threadIdx.y;
				int xx = dx + threadIdx.x;

				sumx += shared_data[yy][xx*3 + 0];
				sumy += shared_data[yy][xx*3 + 1];
				sumz += shared_data[yy][xx*3 + 2];
			}
		}
		out[img_idx*3+0] =sumx/divby;
		out[img_idx*3+1] =sumy/divby;
		out[img_idx*3+2] =sumz/divby;
		}
		
	}

#endif
// Global variables for image data

unsigned char *image, *pixels, *dev_bitmap, *dev_input;
unsigned int imagesizey, imagesizex; // Image size

////////////////////////////////////////////////////////////////////////////////
// main computation function
////////////////////////////////////////////////////////////////////////////////
void computeImages(int kernelsizex, int kernelsizey)
{
	if (kernelsizex > maxKernelSizeX || kernelsizey > maxKernelSizeY)
	{
		printf("Kernel size out of bounds!\n");
		return;
	}
	int device;
    cudaGetDevice(&device);

    int maxSharedMemory;
    cudaDeviceGetAttribute(&maxSharedMemory, cudaDevAttrMaxSharedMemoryPerBlock, device);

	printf("Maximum Shared Memory Per Block: %d \n" ,maxSharedMemory);
	pixels = (unsigned char *) malloc(imagesizex*imagesizey*3);
	cudaMalloc( (void**)&dev_input, imagesizex*imagesizey*3);
	cudaMemcpy( dev_input, image, imagesizey*imagesizex*3, cudaMemcpyHostToDevice );
	cudaMalloc( (void**)&dev_bitmap, imagesizex*imagesizey*3);
	cudaEvent_t start, stop;
	//another implementation of box filter
	#ifdef SINGLE
	dim3 block(BLOCK_SIZE, BLOCK_SIZE);
	int tile_sizex = BLOCK_SIZE-2*kernelsizex;
	int tile_sizey = BLOCK_SIZE-2*kernelsizey;
	dim3 grid((imagesizex/tile_sizex)+1, (imagesizex/tile_sizey)+1);
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    box_filter2<<<grid,block>>>(dev_input, dev_bitmap, imagesizex, imagesizey, kernelsizex, kernelsizey);
	cudaEventRecord(stop);
    cudaDeviceSynchronize();
    cudaEventSynchronize(stop);
	#endif
	//box filter gpu
	#ifdef BOX_FILTER
	dim3 block(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid(imagesizex/(BLOCK_SIZE), imagesizex/(BLOCK_SIZE));
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    box_filter<<<grid,block>>>(dev_input, dev_bitmap, imagesizex, imagesizey, kernelsizex, kernelsizey);
	cudaEventRecord(stop);
    cudaDeviceSynchronize();
    cudaEventSynchronize(stop);
	#endif
	//ordinary gauss
	#ifdef GAUSS
	dim3 block(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid(imagesizex/(BLOCK_SIZE), imagesizex/(BLOCK_SIZE));
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
   	gauss_filter<<<grid,block>>>(dev_input, dev_bitmap, imagesizex, imagesizey, 0, kernelsizey);
	gauss_filter<<<grid,block>>>(dev_bitmap, dev_bitmap, imagesizex, imagesizey, kernelsizex, 0);
	cudaEventRecord(stop);
    cudaDeviceSynchronize();
    cudaEventSynchronize(stop);
	#endif
	//gauss separable
	#ifdef GAUSS_SEPARABLE
	dim3 block(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid(imagesizex/(BLOCK_SIZE), imagesizex/(BLOCK_SIZE));
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
	box_filter<<<grid,block>>>(dev_input, dev_bitmap, imagesizex, imagesizey, 0, 1);
	for(int i = 0; i<4; i++){
		box_filter<<<grid,block>>>(dev_bitmap, dev_bitmap, imagesizex, imagesizey, 0, 1);
	}
	cudaEventRecord(stop);
    cudaDeviceSynchronize();
    cudaEventSynchronize(stop);
	#endif
	//sep boxfilter
	#ifdef SEPARABLE
	dim3 block(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid(imagesizex/(BLOCK_SIZE)+1, imagesizex/(BLOCK_SIZE)+1);
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    box_filter<<<grid,block>>>(dev_input, dev_bitmap, imagesizex, imagesizey, 0, kernelsizey);
	box_filter<<<grid,block>>>(dev_bitmap, dev_bitmap, imagesizex, imagesizey, kernelsizex, 0);
	cudaEventRecord(stop);
    cudaDeviceSynchronize();
    cudaEventSynchronize(stop);
	#endif

	#ifdef MEDIAN
	dim3 block(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid(imagesizex/(BLOCK_SIZE), imagesizex/(BLOCK_SIZE));
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    median_filter<<<grid,block>>>(dev_input, dev_bitmap, imagesizex, imagesizey, kernelsizex, kernelsizey);
	cudaEventRecord(stop);
    cudaDeviceSynchronize();
    cudaEventSynchronize(stop);
	#endif

    float milliseconds_gpu;
    cudaEventElapsedTime(&milliseconds_gpu, start, stop);
    printf("timing: %f \n", milliseconds_gpu/1000);

//	Check for errors!
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("Error: %s\n", cudaGetErrorString(err));
	cudaMemcpy( pixels, dev_bitmap, imagesizey*imagesizex*3, cudaMemcpyDeviceToHost );
	cudaFree( dev_bitmap );
	cudaFree( dev_input );
}

// Display images
void Draw()
{
// Dump the whole picture onto the screen.	
	glClearColor( 0.0, 0.0, 0.0, 1.0 );
	glClear( GL_COLOR_BUFFER_BIT );

	if (imagesizey >= imagesizex)
	{ // Not wide - probably square. Original left, result right.
		glRasterPos2f(-1, -1);
		glDrawPixels( imagesizex, imagesizey, GL_RGB, GL_UNSIGNED_BYTE, image );
		glRasterPos2i(0, -1);
		glDrawPixels( imagesizex, imagesizey, GL_RGB, GL_UNSIGNED_BYTE,  pixels);
	}
	else
	{ // Wide image! Original on top, result below.
		glRasterPos2f(-1, -1);
		glDrawPixels( imagesizex, imagesizey, GL_RGB, GL_UNSIGNED_BYTE, pixels );
		glRasterPos2i(-1, 0);
		glDrawPixels( imagesizex, imagesizey, GL_RGB, GL_UNSIGNED_BYTE, image );
	}
	glFlush();
}

// Main program, inits
int main( int argc, char** argv) 
{
	glutInit(&argc, argv);
	glutInitDisplayMode( GLUT_SINGLE | GLUT_RGBA );

	if (argc > 1)
		image = readppm(argv[1], (int *)&imagesizex, (int *)&imagesizey);
	else
		image = readppm((char *)"maskros512.ppm", (int *)&imagesizex, (int *)&imagesizey);

	if (imagesizey >= imagesizex)
		glutInitWindowSize( imagesizex*2, imagesizey );
	else
		glutInitWindowSize( imagesizex, imagesizey*2 );
	glutCreateWindow("Lab 5");
	glutDisplayFunc(Draw);

	ResetMilli();

	computeImages(5, 5);

// You can save the result to a file like this:
//	writeppm("out.ppm", imagesizey, imagesizex, pixels);

	glutMainLoop();
	return 0;
}
