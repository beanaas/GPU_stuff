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

unsigned char average_kernel(skepu::Region2D<unsigned char> m, size_t elemPerPx)
{
	float scaling = 1.0 / ((m.oj/elemPerPx*2+1)*(m.oi*2+1));
	float res = 0;
	for (int y = -m.oi; y <= m.oi; ++y)
		for (int x = -m.oj; x <= m.oj; x += elemPerPx)
			res += m(y, x);
	return res * scaling;
}

unsigned char average_kernel_1d(skepu::Region1D<unsigned char> m, size_t elemPerPx)
{
	float scaling = 1.0 / ((m.oi * 2 + 1) / elemPerPx);
	float res = 0;

	for (int y = -m.oi; y <= m.oi; y+=elemPerPx)
		res+=m(y);
	return res * scaling;
}



unsigned char gaussian_kernel(skepu::Region1D<unsigned char> m, const skepu::Vec<float> stencil, size_t elemPerPx)
{
	float res = 0;
	int i = 0;
	for (int y = -m.oi; y <= m.oi; y+=elemPerPx){
		res+=m(y) *stencil[i];
		i++;
	}
	return res;
}




int main(int argc, char* argv[])
{
	if (argc < 5)
	{
		std::cout << "Usage: " << argv[0] << " input output radius [backend]\n";
		exit(1);
	}
	
	LodePNGColorType colorType = LCT_RGB;
	std::string inputFileName = argv[1];
	std::string outputFileName = argv[2];
	const int radius = atoi(argv[3]);
	auto spec = skepu::BackendSpec{argv[4]};
	skepu::setGlobalBackendSpec(spec);
	
	// Create the full path for writing the image.
	std::stringstream ss;
	ss << (2 * radius + 1) << "x" << (2 * radius + 1);
	std::string outputFile = outputFileName + ss.str();
	
	// Read the padded image into a matrix. Create the output matrix without padding.
	// Padded version for 2D MapOverlap, non-padded for 1D MapOverlap
	ImageInfo imageInfo;
	skepu::Matrix<unsigned char> inputMatrixPad = ReadAndPadPngFileToMatrix(inputFileName, radius, colorType, imageInfo);
	skepu::Matrix<unsigned char> inputMatrix = ReadPngFileToMatrix(inputFileName, colorType, imageInfo);
	skepu::Matrix<unsigned char> outputMatrix(imageInfo.height, imageInfo.width * imageInfo.elementsPerPixel, 120);
	// more containers...?
	
	// Original version
	{
		auto conv = skepu::MapOverlap(average_kernel);
		conv.setOverlap(radius, radius  * imageInfo.elementsPerPixel);
	
		auto timeTaken = skepu::benchmark::measureExecTime([&]
		{
			conv(outputMatrix, inputMatrixPad, imageInfo.elementsPerPixel);
		});
	
		WritePngFileMatrix(outputMatrix, outputFile + "-average.png", colorType, imageInfo);
		std::cout << "Time for combined: " << (timeTaken.count() / 10E6) << "\n";
	}
	
	
	// Separable version
	// use conv.setOverlapMode(skepu::Overlap::[ColWise RowWise]);
	// and conv.setOverlap(<integer>)
	{
		auto conv = skepu::MapOverlap(average_kernel_1d);
		

		auto timeTaken = skepu::benchmark::measureExecTime([&]
		{
			conv.setOverlap(radius*imageInfo.elementsPerPixel);
			conv.setOverlapMode(skepu::Overlap::RowWise);
			conv(outputMatrix, inputMatrix, imageInfo.elementsPerPixel);

			conv.setOverlap(radius);
			conv.setOverlapMode(skepu::Overlap::ColWise);
			conv(outputMatrix, outputMatrix, 1);
		});
		
		WritePngFileMatrix(outputMatrix, outputFile + "-separable.png", colorType, imageInfo);
		std::cout << "Time for separable: " << (timeTaken.count() / 10E6) << "\n";
	}
	
	
	// Separable gaussian
	{
		skepu::Vector<float> stencil = sampleGaussian(radius);
		auto gaussian_conv = skepu::MapOverlap(gaussian_kernel);
			
		// skeleton instance, etc here (remember to set backend)
	
		auto timeTaken = skepu::benchmark::measureExecTime([&]
		{
			gaussian_conv.setOverlap(radius*imageInfo.elementsPerPixel);
			gaussian_conv.setOverlapMode(skepu::Overlap::RowWise);
			gaussian_conv(outputMatrix, inputMatrix, stencil, imageInfo.elementsPerPixel);

			gaussian_conv.setOverlap(radius);
			gaussian_conv.setOverlapMode(skepu::Overlap::ColWise);
			gaussian_conv(outputMatrix, outputMatrix, stencil, 1);
			// your code here
		});
	
		WritePngFileMatrix(outputMatrix, outputFile + "-gaussian.png", colorType, imageInfo);
		std::cout << "Time for gaussian: " << (timeTaken.count() / 10E6) << "\n";
	}
	
	
	
	return 0;
}


