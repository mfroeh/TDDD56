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
	// For radius = 25 create a 1D array for sorting with values from the image
	unsigned char values[2601];
	int index = 0;
	for (int y = -image.oi; y <= image.oi; ++y){
		for (int x = -image.oj; x <= image.oj; x += elemPerPx){
			values[index] = image(y, x);
			index++;
		}
	}


	 int i, j, minimumIndex;
 
    // One by one move boundary of
    // unsorted subarray
    for (i = 0; i < index - 1; i++) {
 
        // Find the minimum element in
        // unsorted array
        minimumIndex = i;
        for (j = i + 1; j < index; j++) {
            if (values[j] < values[minimumIndex])
                minimumIndex = j;
        }
 
        // Swap the found minimum element
        // with the first element
        if (minimumIndex != i){
			int temp = values[minimumIndex];
			values[minimumIndex] = values[i];
			values[i] = temp;
		}
    }

	return values[(index-1)/2];
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
	spec.setCPUThreads(16);
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


