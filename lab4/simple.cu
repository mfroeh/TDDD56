// Simple CUDA example by Ingemar Ragnemalm 2009. Simplest possible?
// Assigns every element in an array with its index.
// Update 2022: Changed to cudaDeviceSynchronize.

// nvcc simple.cu -L /usr/local/cuda/lib -lcudart -o simple

#include <stdio.h>

#include <array>
#include <iostream>

const int N = 16;
const int blocksize = 16;

__global__ void simple(float *c) { c[threadIdx.x] = threadIdx.x; }

__global__ void sqrt(float *arr) { arr[threadIdx.x] = sqrt(arr[threadIdx.x]); }

int main() {
  float *c{new float[N]};
  float *cd{};
  cudaMalloc(&cd, sizeof(float) * N);

  dim3 dim_block(blocksize, 1);
  dim3 dim_grid(1, 1);
  simple<<<dim_grid, dim_block>>>(cd);
  cudaDeviceSynchronize();
  cudaMemcpy(c, cd, sizeof(float) * N, cudaMemcpyDeviceToHost);
  cudaFree(cd);
  for (int i = 0; i < N; i++) printf("%f ", c[i]);
  printf("\n");
  delete[] c;

  // Square root
  float *arr_host{new float[N]};
  std::array<float, N> roots{};
  for (size_t i{}; i < N; ++i) {
    arr_host[i] = i * 0.5f;
    roots[i] = sqrt(arr_host[i]);
  }

  float *arr_dev{};
  cudaMalloc(&arr_dev, sizeof(float) * N);
  cudaMemcpy(arr_dev, arr_host, sizeof(float) * N, cudaMemcpyHostToDevice);

  sqrt<<<dim_grid, dim_block>>>(arr_dev);
  cudaDeviceSynchronize();
  cudaMemcpy(arr_host, arr_dev, sizeof(float) * N, cudaMemcpyDeviceToHost);
  cudaFree(arr_dev);

  for (size_t i{}; i < N; ++i) {
    std::cout << "cpu = " << roots[i] << " == " << arr_host[i] << " = gpu ? "
              << (roots[i] == arr_host[i]) << std::endl;
  }
}
