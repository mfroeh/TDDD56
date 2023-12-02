// Lab 5, image filters with CUDA.

// Compile with a command-line similar to Lab 4:
// nvcc filter.cu -c -arch=sm_30 -o filter.o
// g++ filter.o milli.c readppm.c -lGL -lm -lcuda -lcudart -L/usr/local/cuda/lib
// -lglut -o filter or (multicore lab) nvcc filter.cu -c -arch=sm_20 -o filter.o
// g++ filter.o milli.c readppm.c -lGL -lm -lcuda -L/usr/local/cuda/lib64
// -lcudart -lglut -o filter

// 2017-11-27: Early pre-release, dubbed "beta".
// 2017-12-03: First official version! Brand new lab 5 based on the old lab 6.
// Better variable names, better prepared for some lab tasks. More changes may
// come but I call this version 1.0b2. 2017-12-04: Two fixes: Added
// command-lines (above), fixed a bug in computeImages that allocated too much
// memory. b3 2017-12-04: More fixes: Tightened up the kernel with edge
// clamping. Less code, nicer result (no borders). Cleaned up some messed up X
// and Y. b4 2022-12-07: A correction for a deprecated function.

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
#include "milli.h"
#include "readppm.h"

struct pixel {
  unsigned char r{}, g{}, b{};
};

// Use these for setting shared memory size.
#define maxKernelSizeX 10
#define maxKernelSizeY 10

#define BLOCKX 32
#define BLOCKY 32

#define BLOCK_HEIGHT (maxKernelSizeX * 2 + 1)
#define BLOCK_WIDTH (maxKernelSizeY * 2 + 1)

__global__ void filter(pixel *image, pixel *out, const unsigned int imagesizex,
                       const unsigned int imagesizey, const int kernelsizex,
                       const int kernelsizey) {
  // map from blockIdx to pixel position
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  __shared__ pixel
      block[BLOCKY + 2 * maxKernelSizeY][BLOCKX + 2 * maxKernelSizeX];

  // For now every thread just copies all their pixel values
  for (int dy{-kernelsizey}, i{}; dy <= kernelsizey; ++dy, ++i) {
    for (int dx{-kernelsizex}, j{}; dx <= kernelsizex; ++dx, ++j) {
      int yy{min(max(y + dy, 0), static_cast<int>(imagesizey) - 1)};
      int xx{min(max(x + dx, 0), static_cast<int>(imagesizex) - 1)};
      block[i + threadIdx.y][j + threadIdx.x] = image[yy * imagesizex + xx];
    }
  }

  // Synch all threads in block
  __syncthreads();

  // If inside image
  if (x < imagesizex && y < imagesizey) {
    int kernel_height{2 * kernelsizey + 1};
    int kernel_width{2 * kernelsizex + 1};

    // Filter kernel (simple box filter)
    unsigned sumx{}, sumy{}, sumz{};
    for (size_t i{}; i < kernel_height; ++i) {
      for (size_t j{}; j < kernel_width; ++j) {
        sumx += block[i + threadIdx.y][j + threadIdx.x].r;
        sumy += block[i + threadIdx.y][j + threadIdx.x].g;
        sumz += block[i + threadIdx.y][j + threadIdx.x].b;
      }
    }

    // Works for box filters only!
    int divby{(2 * kernelsizex + 1) * (2 * kernelsizey + 1)};
    out[y * imagesizex + x].r = sumx / divby;
    out[y * imagesizex + x].g = sumy / divby;
    out[y * imagesizex + x].b = sumz / divby;
  } else {
    printf("%d,%d is outside imagesize!", x, y);
  }
}

// Global variables for image data
pixel *dev_input, *dev_bitmap;
unsigned char *image, *pixels;
unsigned int imagesizey, imagesizex;  // Image size

////////////////////////////////////////////////////////////////////////////////
// main computation function
////////////////////////////////////////////////////////////////////////////////
void computeImages(int kernelsizex, int kernelsizey) {
  if (kernelsizex > maxKernelSizeX || kernelsizey > maxKernelSizeY) {
    printf("Kernel size out of bounds!\n");
    return;
  }

  pixels = (unsigned char *)malloc(imagesizex * imagesizey * 3);
  cudaMalloc((void **)&dev_input, imagesizex * imagesizey * 3);
  cudaMemcpy(dev_input, image, imagesizey * imagesizex * 3,
             cudaMemcpyHostToDevice);
  cudaMalloc((void **)&dev_bitmap, imagesizex * imagesizey * 3);

  dim3 block_dim{BLOCKX, BLOCKY};
  dim3 grid_dim{imagesizex / block_dim.x, imagesizey / block_dim.y};
  filter<<<grid_dim, block_dim>>>(dev_input, dev_bitmap, imagesizex, imagesizey,
                                  kernelsizex,
                                  kernelsizey);  // Awful load balance
  cudaDeviceSynchronize();
  //	Check for errors!
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(err));
  cudaMemcpy(pixels, dev_bitmap, imagesizey * imagesizex * 3,
             cudaMemcpyDeviceToHost);
  cudaFree(dev_bitmap);
  cudaFree(dev_input);
}

// Display images
void Draw() {
  // Dump the whole picture onto the screen.
  glClearColor(0.0, 0.0, 0.0, 1.0);
  glClear(GL_COLOR_BUFFER_BIT);

  if (imagesizey >= imagesizex) {  // Not wide - probably square. Original
                                   // left, result right.
    glRasterPos2f(-1, -1);
    glDrawPixels(imagesizex, imagesizey, GL_RGB, GL_UNSIGNED_BYTE, image);
    glRasterPos2i(0, -1);
    glDrawPixels(imagesizex, imagesizey, GL_RGB, GL_UNSIGNED_BYTE, pixels);
  } else {  // Wide image! Original on top, result below.
    glRasterPos2f(-1, -1);
    glDrawPixels(imagesizex, imagesizey, GL_RGB, GL_UNSIGNED_BYTE, pixels);
    glRasterPos2i(-1, 0);
    glDrawPixels(imagesizex, imagesizey, GL_RGB, GL_UNSIGNED_BYTE, image);
  }
  glFlush();
}

// Main program, inits
int main(int argc, char **argv) {
  glutInit(&argc, argv);
  glutInitDisplayMode(GLUT_SINGLE | GLUT_RGBA);

  if (argc > 1)
    image = readppm(argv[1], (int *)&imagesizex, (int *)&imagesizey);
  else
    image = readppm((char *)"maskros512.ppm", (int *)&imagesizex,
                    (int *)&imagesizey);

  if (imagesizey >= imagesizex)
    glutInitWindowSize(imagesizex * 2, imagesizey);
  else
    glutInitWindowSize(imagesizex, imagesizey * 2);
  glutCreateWindow("Lab 5");
  glutDisplayFunc(Draw);

  ResetMilli();

  computeImages(10, 10);

  // You can save the result to a file like this:
  //	writeppm("out.ppm", imagesizey, imagesizex, pixels);

  glutMainLoop();
  return 0;
}
