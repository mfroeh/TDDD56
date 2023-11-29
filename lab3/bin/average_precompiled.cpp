#define SKEPU_PRECOMPILED
#define SKEPU_OPENMP
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
	float scaling = 1.0 / (m.oi / elemPerPx * 2 + 1);
    float res = 0;
    for (int i = -m.oi; i < m.oi; i += elemPerPx)
        res += m(i);
    return res * scaling;
}



unsigned char gaussian_kernel(skepu::Region1D<unsigned char> m, const skepu::Vec<float> stencil, size_t elemPerPx)
{
	int j = 0;
	float res = 0;
	for (int i = -m.oi; i < m.oi; i += elemPerPx){
		res += m(i) * stencil(j);
		j++;
	}
        
    return res;
}





struct skepu_userfunction_skepu_skel_0convGausR_gaussian_kernel
{
constexpr static size_t totalArity = 3;
constexpr static size_t outArity = 1;
constexpr static bool indexed = 0;
using IndexType = void;
using ElwiseArgs = std::tuple<>;
using ContainerArgs = std::tuple<const skepu::Vec<float>>;
using UniformArgs = std::tuple<unsigned long>;
typedef std::tuple<skepu::ProxyTag::Default> ProxyTags;
constexpr static skepu::AccessMode anyAccessMode[] = {
skepu::AccessMode::Read, };

using Ret = unsigned char;

constexpr static bool prefersMatrix = 0;

#define SKEPU_USING_BACKEND_OMP 1
#undef VARIANT_CPU
#undef VARIANT_OPENMP
#undef VARIANT_CUDA
#define VARIANT_CPU(block)
#define VARIANT_OPENMP(block) block
#define VARIANT_CUDA(block)
static inline SKEPU_ATTRIBUTE_FORCE_INLINE unsigned char OMP(skepu::Region1D<unsigned char> m, const skepu::Vec<float> stencil, unsigned long elemPerPx)
{
	int j = 0;
	float res = 0;
	for (int i = -m.oi; i < m.oi; i += elemPerPx){
		res += m(i) * stencil(j);
		j++;
	}
        
    return res;
}
#undef SKEPU_USING_BACKEND_OMP

#define SKEPU_USING_BACKEND_CPU 1
#undef VARIANT_CPU
#undef VARIANT_OPENMP
#undef VARIANT_CUDA
#define VARIANT_CPU(block) block
#define VARIANT_OPENMP(block)
#define VARIANT_CUDA(block) block
static inline SKEPU_ATTRIBUTE_FORCE_INLINE unsigned char CPU(skepu::Region1D<unsigned char> m, const skepu::Vec<float> stencil, unsigned long elemPerPx)
{
	int j = 0;
	float res = 0;
	for (int i = -m.oi; i < m.oi; i += elemPerPx){
		res += m(i) * stencil(j);
		j++;
	}
        
    return res;
}
#undef SKEPU_USING_BACKEND_CPU
};


struct skepu_userfunction_skepu_skel_1convC_average_kernel_1d
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

#define SKEPU_USING_BACKEND_OMP 1
#undef VARIANT_CPU
#undef VARIANT_OPENMP
#undef VARIANT_CUDA
#define VARIANT_CPU(block)
#define VARIANT_OPENMP(block) block
#define VARIANT_CUDA(block)
static inline SKEPU_ATTRIBUTE_FORCE_INLINE unsigned char OMP(skepu::Region1D<unsigned char> m, unsigned long elemPerPx)
{
	float scaling = 1.0 / (m.oi / elemPerPx * 2 + 1);
    float res = 0;
    for (int i = -m.oi; i < m.oi; i += elemPerPx)
        res += m(i);
    return res * scaling;
}
#undef SKEPU_USING_BACKEND_OMP

#define SKEPU_USING_BACKEND_CPU 1
#undef VARIANT_CPU
#undef VARIANT_OPENMP
#undef VARIANT_CUDA
#define VARIANT_CPU(block) block
#define VARIANT_OPENMP(block)
#define VARIANT_CUDA(block) block
static inline SKEPU_ATTRIBUTE_FORCE_INLINE unsigned char CPU(skepu::Region1D<unsigned char> m, unsigned long elemPerPx)
{
	float scaling = 1.0 / (m.oi / elemPerPx * 2 + 1);
    float res = 0;
    for (int i = -m.oi; i < m.oi; i += elemPerPx)
        res += m(i);
    return res * scaling;
}
#undef SKEPU_USING_BACKEND_CPU
};


struct skepu_userfunction_skepu_skel_2convGausC_gaussian_kernel
{
constexpr static size_t totalArity = 3;
constexpr static size_t outArity = 1;
constexpr static bool indexed = 0;
using IndexType = void;
using ElwiseArgs = std::tuple<>;
using ContainerArgs = std::tuple<const skepu::Vec<float>>;
using UniformArgs = std::tuple<unsigned long>;
typedef std::tuple<skepu::ProxyTag::Default> ProxyTags;
constexpr static skepu::AccessMode anyAccessMode[] = {
skepu::AccessMode::Read, };

using Ret = unsigned char;

constexpr static bool prefersMatrix = 0;

#define SKEPU_USING_BACKEND_OMP 1
#undef VARIANT_CPU
#undef VARIANT_OPENMP
#undef VARIANT_CUDA
#define VARIANT_CPU(block)
#define VARIANT_OPENMP(block) block
#define VARIANT_CUDA(block)
static inline SKEPU_ATTRIBUTE_FORCE_INLINE unsigned char OMP(skepu::Region1D<unsigned char> m, const skepu::Vec<float> stencil, unsigned long elemPerPx)
{
	int j = 0;
	float res = 0;
	for (int i = -m.oi; i < m.oi; i += elemPerPx){
		res += m(i) * stencil(j);
		j++;
	}
        
    return res;
}
#undef SKEPU_USING_BACKEND_OMP

#define SKEPU_USING_BACKEND_CPU 1
#undef VARIANT_CPU
#undef VARIANT_OPENMP
#undef VARIANT_CUDA
#define VARIANT_CPU(block) block
#define VARIANT_OPENMP(block)
#define VARIANT_CUDA(block) block
static inline SKEPU_ATTRIBUTE_FORCE_INLINE unsigned char CPU(skepu::Region1D<unsigned char> m, const skepu::Vec<float> stencil, unsigned long elemPerPx)
{
	int j = 0;
	float res = 0;
	for (int i = -m.oi; i < m.oi; i += elemPerPx){
		res += m(i) * stencil(j);
		j++;
	}
        
    return res;
}
#undef SKEPU_USING_BACKEND_CPU
};


struct skepu_userfunction_skepu_skel_3convR_average_kernel_1d
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

#define SKEPU_USING_BACKEND_OMP 1
#undef VARIANT_CPU
#undef VARIANT_OPENMP
#undef VARIANT_CUDA
#define VARIANT_CPU(block)
#define VARIANT_OPENMP(block) block
#define VARIANT_CUDA(block)
static inline SKEPU_ATTRIBUTE_FORCE_INLINE unsigned char OMP(skepu::Region1D<unsigned char> m, unsigned long elemPerPx)
{
	float scaling = 1.0 / (m.oi / elemPerPx * 2 + 1);
    float res = 0;
    for (int i = -m.oi; i < m.oi; i += elemPerPx)
        res += m(i);
    return res * scaling;
}
#undef SKEPU_USING_BACKEND_OMP

#define SKEPU_USING_BACKEND_CPU 1
#undef VARIANT_CPU
#undef VARIANT_OPENMP
#undef VARIANT_CUDA
#define VARIANT_CPU(block) block
#define VARIANT_OPENMP(block)
#define VARIANT_CUDA(block) block
static inline SKEPU_ATTRIBUTE_FORCE_INLINE unsigned char CPU(skepu::Region1D<unsigned char> m, unsigned long elemPerPx)
{
	float scaling = 1.0 / (m.oi / elemPerPx * 2 + 1);
    float res = 0;
    for (int i = -m.oi; i < m.oi; i += elemPerPx)
        res += m(i);
    return res * scaling;
}
#undef SKEPU_USING_BACKEND_CPU
};


struct skepu_userfunction_skepu_skel_4conv_average_kernel
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

#define SKEPU_USING_BACKEND_OMP 1
#undef VARIANT_CPU
#undef VARIANT_OPENMP
#undef VARIANT_CUDA
#define VARIANT_CPU(block)
#define VARIANT_OPENMP(block) block
#define VARIANT_CUDA(block)
static inline SKEPU_ATTRIBUTE_FORCE_INLINE unsigned char OMP(skepu::Region2D<unsigned char> m, unsigned long elemPerPx)
{
	float scaling = 1.0 / ((m.oj/elemPerPx*2+1)*(m.oi*2+1));
	float res = 0;
	for (int y = -m.oi; y <= m.oi; ++y)
		for (int x = -m.oj; x <= m.oj; x += elemPerPx)
			res += m(y, x);
	return res * scaling;
}
#undef SKEPU_USING_BACKEND_OMP

#define SKEPU_USING_BACKEND_CPU 1
#undef VARIANT_CPU
#undef VARIANT_OPENMP
#undef VARIANT_CUDA
#define VARIANT_CPU(block) block
#define VARIANT_OPENMP(block)
#define VARIANT_CUDA(block) block
static inline SKEPU_ATTRIBUTE_FORCE_INLINE unsigned char CPU(skepu::Region2D<unsigned char> m, unsigned long elemPerPx)
{
	float scaling = 1.0 / ((m.oj/elemPerPx*2+1)*(m.oi*2+1));
	float res = 0;
	for (int y = -m.oi; y <= m.oi; ++y)
		for (int x = -m.oj; x <= m.oj; x += elemPerPx)
			res += m(y, x);
	return res * scaling;
}
#undef SKEPU_USING_BACKEND_CPU
};

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
	skepu::Matrix<unsigned char> outputMatrixAverage(imageInfo.height, imageInfo.width * imageInfo.elementsPerPixel, 120);
	skepu::Matrix<unsigned char> intermediaryMatrix(imageInfo.height, imageInfo.width * imageInfo.elementsPerPixel, 120);
	skepu::Matrix<unsigned char> outputMatrixAverage1D(imageInfo.height, imageInfo.width * imageInfo.elementsPerPixel, 120);
	skepu::Matrix<unsigned char> outputMatrixGausian(imageInfo.height, imageInfo.width * imageInfo.elementsPerPixel, 120);
	
	// more containers...?
	
	// Original version
	{
		skepu::backend::MapOverlap2D<skepu_userfunction_skepu_skel_4conv_average_kernel, bool, void> conv(false);
		conv.setOverlap(radius, radius  * imageInfo.elementsPerPixel);
	
		auto timeTaken = skepu::benchmark::measureExecTime([&]
		{
			conv(outputMatrixAverage, inputMatrixPad, imageInfo.elementsPerPixel);
		});
	
		WritePngFileMatrix(outputMatrixAverage, outputFile + "-average.png", colorType, imageInfo);
		std::cout << "Time for combined: " << (timeTaken.count() / 10E6) << "\n";
	}
	
	
	// Separable version
	// use conv.setOverlapMode(skepu::Overlap::[ColWise RowWise]);
	// and conv.setOverlap(<integer>)
	{
		skepu::backend::MapOverlap1D<skepu_userfunction_skepu_skel_3convR_average_kernel_1d, bool, bool, bool, bool, void> convR(false, false, false, false);
		convR.setOverlapMode(skepu::Overlap::RowWise);
		convR.setOverlap(radius * imageInfo.elementsPerPixel);

		skepu::backend::MapOverlap1D<skepu_userfunction_skepu_skel_1convC_average_kernel_1d, bool, bool, bool, bool, void> convC(false, false, false, false);
		convC.setOverlapMode(skepu::Overlap::ColWise);
		convC.setOverlap(radius);
		
		auto timeTaken = skepu::benchmark::measureExecTime([&]
		{
			// Apply the 1D average filter row-wise
			convR(intermediaryMatrix, inputMatrix, imageInfo.elementsPerPixel);

        	// Apply the 1D average filter column-wise
        	convC(outputMatrixAverage1D, intermediaryMatrix, 1);
	
		});
		
		WritePngFileMatrix(outputMatrixAverage1D, outputFile + "-separable.png", colorType, imageInfo);
		std::cout << "Time for separable: " << (timeTaken.count() / 10E6) << "\n";
	}	
	
	// Separable gaussian
	{
		skepu::Vector<float> stencil = sampleGaussian(radius);
		skepu::backend::MapOverlap1D<skepu_userfunction_skepu_skel_0convGausR_gaussian_kernel, bool, bool, bool, bool, void> convGausR(false, false, false, false);
		convGausR.setOverlapMode(skepu::Overlap::RowWise);
		convGausR.setOverlap(radius * imageInfo.elementsPerPixel);

		skepu::backend::MapOverlap1D<skepu_userfunction_skepu_skel_2convGausC_gaussian_kernel, bool, bool, bool, bool, void> convGausC(false, false, false, false);
		convGausC.setOverlapMode(skepu::Overlap::ColWise);
		convGausC.setOverlap(radius);
			
		// skeleton instance, etc here (remember to set backend)
	
		auto timeTaken = skepu::benchmark::measureExecTime([&]
		{
			// Apply the 1D average filter row-wise
			convGausR(intermediaryMatrix, inputMatrix, stencil, imageInfo.elementsPerPixel);

        	// Apply the 1D average filter column-wise
        	convGausC(outputMatrixGausian, intermediaryMatrix, stencil, 1);
		});
	
		WritePngFileMatrix(outputMatrixGausian, outputFile + "-gaussian.png", colorType, imageInfo);
		std::cout << "Time for gaussian: " << (timeTaken.count() / 10E6) << "\n";
	}
	
	
	
	return 0;
}


