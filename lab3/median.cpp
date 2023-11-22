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
	int counts[256];
	for (int i = 0; i < 256; ++i) {
		counts[i] = 0;
	}

	// No data dependency, this is trivially parallelizable!
	for (int y = -image.oi; y <= image.oi; ++y){
		for (int x = -image.oj; x <= image.oj; x += elemPerPx){
			counts[image(y, x)]++;
		}
	}

	// Number of r|b|g values to find the median in
	int n = (image.oi * 2 + 1) * (image.oj / elemPerPx * 2 + 1);

	// Data dependency here, can't parallelize!
	int acc = 0;
	for (unsigned i = 0; i < 256; ++i) {
		acc += counts[i];
		if (acc >= ((n-1)/2)) {
			return i;
		}
	}
	// This should never happen, and if so we can observe that the image is black
	return 0;
}

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
	// spec.setCPUThreads(16);
	skepu::setGlobalBackendSpec(spec);
	
	// Create the full path for writing the image.
	std::stringstream ss;
	ss << (2 * radius + 1) << "x" << (2 * radius + 1);
	std::string outputFileNamePad = outputFileName + ss.str() + "-median.png";
		
	// Read the padded image into a matrix. Create the output matrix without padding.
	ImageInfo imageInfo;
	skepu::Matrix<unsigned char> inputMatrix = ReadAndPadPngFileToMatrix(inputFileName, radius, colorType, imageInfo);
	skepu::Matrix<unsigned char> outputMatrix(imageInfo.height, imageInfo.width * imageInfo.elementsPerPixel, 120);
	
	// Skeleton instance
	auto calculateMedian = skepu::MapOverlap(median_kernel);
	calculateMedian.setOverlap(radius, radius  * imageInfo.elementsPerPixel);
	
	auto timeTaken = skepu::benchmark::measureExecTime([&]
	{
		calculateMedian(outputMatrix, inputMatrix, imageInfo.elementsPerPixel);
	});

	WritePngFileMatrix(outputMatrix, outputFileNamePad, colorType, imageInfo);
	
	std::cout << "Time: " << (timeTaken.count() / 10E6) << "\n";
	
	return 0;
}


