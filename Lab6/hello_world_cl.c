// Hello World for OpenCL - the real thing!
// Like my CUDA Hello World, it computes, in parallel, on the GPU,
// the string "World!" from "Hello " and an array of offsets.
// By Ingemar Ragnemalm, based on the hello.c demo (which was NOT Hello World in any way).

// gcc hello_world_cl.c -o hello_world_cl CLutilities.c -lOpenCL -I/usr/local/cuda/include/

#include <stdio.h>
#include <math.h>
#include "CLutilities.h"

#ifdef __APPLE__
  #include <OpenCL/opencl.h>
#else
  #include <CL/cl.h>
#endif

const char *KernelSource = "\n" \
"__kernel void hello(              \n" \
"   __global char* a,          \n" \
"   __global char* b,          \n" \
"   __global char* c,          \n" \
"   const unsigned int count)  \n" \
"{                             \n" \
"   int i = get_global_id(0);  \n" \
"   if(i < count)              \n" \
"       c[i] = a[i] + b[i];   \n" \
"}                             \n" \
"\n";

#define DATA_SIZE (16)

int main(int argc, char** argv)
{
	int err;							// error code returned from api calls
	cl_device_id device_id;			 // compute device id 
	cl_context context;				 // compute context
	cl_command_queue commands;		  // compute command queue
    cl_program program;                 // compute program
    cl_kernel kernel;                   // compute kernel
    cl_mem input;                       // device memory used for the input array
    cl_mem input2;                       // device memory used for the input array
    cl_mem output;                      // device memory used for the output array
    size_t global;                      // global domain size for our calculation
    size_t local;                       // local domain size for our calculation
    
	int i;
    unsigned int count = DATA_SIZE;

	// Input data
	char a[DATA_SIZE] = "Hello,\0\0\0\0\0\0";
	char b[DATA_SIZE] = {15, 10, 6, 0, -11, -11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
	// Output data
	char c[DATA_SIZE];
	
	// Print original data
	printf("%s ", a);


	cl_platform_id platform;
	unsigned int no_plat;
	//gets the platform used
	err =  clGetPlatformIDs(1,&platform,&no_plat);
	printCLError(err,0);

	// Where to run
	//from the platform get the GPU used
	err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
   	printCLError(err,1);

	//creating context for GPU
	context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
   	printCLError(err,2);

	//creates a queue for commands, api calls
	commands = clCreateCommandQueueWithProperties(context, device_id, 0, &err);
   	printCLError(err,4);
	
	// What to run
	//create OpenCL program object from the source code
	program = clCreateProgramWithSource(context, 1, (const char **) & KernelSource, NULL, &err);
   	printCLError(err,5);
	//builds the program
	err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
   	printCLError(err,12);
	//creates the kernel, "hello" is the kernel name
	kernel = clCreateKernel(program, "hello", &err);
   	printCLError(err,6);
	
	// Create space for data and copy a and b to device (note that we could also use clEnqueueWriteBuffer to upload)
	input = clCreateBuffer(context,  CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,  sizeof(char) * DATA_SIZE, a, NULL);
	input2 = clCreateBuffer(context,  CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,  sizeof(char) * DATA_SIZE, b, NULL);
	output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(char) * DATA_SIZE, NULL, NULL);
	printCLError(err,7);
	
	// Send data
	err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input);
	err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &input2);
	err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &output);
	err |= clSetKernelArg(kernel, 3, sizeof(unsigned int), &count);
	printCLError(err,8);
	
	local = DATA_SIZE;

	// Run kernel!
	global = DATA_SIZE; // count;
	err = clEnqueueNDRangeKernel(commands, kernel, 1, NULL, &global, &local, 0, NULL, NULL);
	printCLError(err,9);

	clFinish(commands);

	// Read result
	err = clEnqueueReadBuffer( commands, output, CL_TRUE, 0, sizeof(char) * count, c, 0, NULL, NULL );  
	printCLError(err,11);
	

//	Print result
		printf("%s\n", c);

	// Clean up
	clReleaseMemObject(input);
	clReleaseMemObject(output);
	clReleaseProgram(program);
	clReleaseKernel(kernel);
	clReleaseCommandQueue(commands);
	clReleaseContext(context);
    return 0;
}
