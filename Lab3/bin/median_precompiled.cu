#define SKEPU_PRECOMPILED
#define SKEPU_OPENMP
#define SKEPU_OPENCL
#define SKEPU_CUDA
/**************************
** TDDD56 Lab 3
***************************
** Author:
** August Ernstsson
**************************/

#include <stdio.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <time.h>
#include <iterator>

#include <skepu>

#include "support.h"


unsigned char median_kernel(skepu::Region2D<unsigned char> image, size_t elemPerPx)
{
	int size = ((image.oi*2+1)*((image.oj/elemPerPx)*2+1));
	unsigned char arr1d[20000];
	int idx = 0;
	for(int y = -image.oi; y <= image.oi; ++y){
		for (int x = -image.oj; x <= image.oj; x += elemPerPx){
			arr1d[idx] = image(y, x);
			idx++;
		}
	}
	unsigned char tmp;
    for (int i = 0; i < size - 1; i++) {
        for (int j = 0; j < size - i - 1; j++) {
            if (arr1d[j] > arr1d[j + 1]) {
				tmp = arr1d[j];
				arr1d[j] = arr1d[j+1];
				arr1d[j+1] = tmp;
            }
        }
    }

	return arr1d[size/2];
	
}



struct skepu_userfunction_skepu_skel_0calculateMedian_median_kernel
{
constexpr static size_t totalArity = 2;
constexpr static size_t outArity = 1;
constexpr static bool indexed = 0;
using IndexType = void;
using ElwiseArgs = std::tuple<>;
using ContainerArgs = std::tuple<>;
using UniformArgs = std::tuple<unsigned long>;
typedef std::tuple<> ProxyTags;
constexpr static skepu::AccessMode anyAccessMode[] = {
};

using Ret = unsigned char;

constexpr static bool prefersMatrix = 0;

#define SKEPU_USING_BACKEND_CUDA 1
#undef VARIANT_CPU
#undef VARIANT_OPENMP
#undef VARIANT_CUDA
#define VARIANT_CPU(block)
#define VARIANT_OPENMP(block)
#define VARIANT_CUDA(block) block
static inline SKEPU_ATTRIBUTE_FORCE_INLINE __device__ unsigned char CU(skepu::Region2D<unsigned char> image, unsigned long elemPerPx)
{
	int size = ((image.oi*2+1)*((image.oj/elemPerPx)*2+1));
	unsigned char arr1d[20000];
	int idx = 0;
	for(int y = -image.oi; y <= image.oi; ++y){
		for (int x = -image.oj; x <= image.oj; x += elemPerPx){
			arr1d[idx] = image(y, x);
			idx++;
		}
	}
	unsigned char tmp;
    for (int i = 0; i < size - 1; i++) {
        for (int j = 0; j < size - i - 1; j++) {
            if (arr1d[j] > arr1d[j + 1]) {
				tmp = arr1d[j];
				arr1d[j] = arr1d[j+1];
				arr1d[j+1] = tmp;
            }
        }
    }

	return arr1d[size/2];
	
}
#undef SKEPU_USING_BACKEND_CUDA

#define SKEPU_USING_BACKEND_OMP 1
#undef VARIANT_CPU
#undef VARIANT_OPENMP
#undef VARIANT_CUDA
#define VARIANT_CPU(block)
#define VARIANT_OPENMP(block) block
#define VARIANT_CUDA(block)
static inline SKEPU_ATTRIBUTE_FORCE_INLINE unsigned char OMP(skepu::Region2D<unsigned char> image, unsigned long elemPerPx)
{
	int size = ((image.oi*2+1)*((image.oj/elemPerPx)*2+1));
	unsigned char arr1d[20000];
	int idx = 0;
	for(int y = -image.oi; y <= image.oi; ++y){
		for (int x = -image.oj; x <= image.oj; x += elemPerPx){
			arr1d[idx] = image(y, x);
			idx++;
		}
	}
	unsigned char tmp;
    for (int i = 0; i < size - 1; i++) {
        for (int j = 0; j < size - i - 1; j++) {
            if (arr1d[j] > arr1d[j + 1]) {
				tmp = arr1d[j];
				arr1d[j] = arr1d[j+1];
				arr1d[j+1] = tmp;
            }
        }
    }

	return arr1d[size/2];
	
}
#undef SKEPU_USING_BACKEND_OMP

#define SKEPU_USING_BACKEND_CPU 1
#undef VARIANT_CPU
#undef VARIANT_OPENMP
#undef VARIANT_CUDA
#define VARIANT_CPU(block) block
#define VARIANT_OPENMP(block)
#define VARIANT_CUDA(block) block
static inline SKEPU_ATTRIBUTE_FORCE_INLINE unsigned char CPU(skepu::Region2D<unsigned char> image, unsigned long elemPerPx)
{
	int size = ((image.oi*2+1)*((image.oj/elemPerPx)*2+1));
	unsigned char arr1d[20000];
	int idx = 0;
	for(int y = -image.oi; y <= image.oi; ++y){
		for (int x = -image.oj; x <= image.oj; x += elemPerPx){
			arr1d[idx] = image(y, x);
			idx++;
		}
	}
	unsigned char tmp;
    for (int i = 0; i < size - 1; i++) {
        for (int j = 0; j < size - i - 1; j++) {
            if (arr1d[j] > arr1d[j + 1]) {
				tmp = arr1d[j];
				arr1d[j] = arr1d[j+1];
				arr1d[j+1] = tmp;
            }
        }
    }

	return arr1d[size/2];
	
}
#undef SKEPU_USING_BACKEND_CPU
};

#include "median_precompiled_Overlap2DKernel_median_kernel.cu"
#include "median_precompiled_Overlap2DKernel_median_kernel_cl_source.inl"
int main(int argc, char* argv[])
{
	LodePNGColorType colorType = LCT_RGB;
	
	if (argc < 5)
	{
		std::cout << "Usage: " << argv[0] << " input output radius [backend]\n";
		exit(1);
	}
	
	std::string inputFileName = argv[1];
	std::string outputFileName = argv[2];
	const int radius = atoi(argv[3]);
	auto spec = skepu::BackendSpec{argv[4]};
	skepu::setGlobalBackendSpec(spec);
	
	// Create the full path for writing the image.
	std::stringstream ss;
	ss << (2 * radius + 1) << "x" << (2 * radius + 1);
	std::string outputFileNamePad = outputFileName + ss.str() + "-median.png";
		
	// Read the padded image into a matrix. Create the output matrix without padding.
	ImageInfo imageInfo;
	skepu::Matrix<unsigned char> inputMatrix = ReadAndPadPngFileToMatrix(inputFileName, radius, colorType, imageInfo);
	skepu::Matrix<unsigned char> outputMatrix(imageInfo.height, imageInfo.width * imageInfo.elementsPerPixel, 120);
	std::cout << "Test1 \n";
	// Skeleton instance
	skepu::backend::MapOverlap2D<skepu_userfunction_skepu_skel_0calculateMedian_median_kernel, decltype(&median_precompiled_Overlap2DKernel_median_kernel_conv_cuda_2D_kernel), CLWrapperClass_median_precompiled_Overlap2DKernel_median_kernel> calculateMedian(median_precompiled_Overlap2DKernel_median_kernel_conv_cuda_2D_kernel);
	calculateMedian.setOverlap(radius, radius  * imageInfo.elementsPerPixel);
	std::cout << "Test2 \n";
	{
	auto timeTaken = skepu::benchmark::measureExecTime([&]
	{
		calculateMedian(outputMatrix, inputMatrix, imageInfo.elementsPerPixel);
	});

	std::cout << "Test3 \n";
	std::cout << "Time: " << (timeTaken.count() / 10E6) << "\n";
	WritePngFileMatrix(outputMatrix, outputFileNamePad, colorType, imageInfo);
	
	
	}
	
	return 0;
}


