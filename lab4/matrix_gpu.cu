#include <iomanip>
#include <iostream>
#include <vector>
#include <cassert>
#include "query.hh"

const unsigned N{1024 * 16};

void add_matrix(float *a, float *b, float *c, int N)
{
	int index;
	
	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++)
		{
			index = i + j*N;
			c[index] = a[index] + b[index];
		}
}

__global__ void matrix_add(float *a, float *b, float *c) {
  unsigned x{blockIdx.x * blockDim.x + threadIdx.x};
  unsigned y{blockIdx.y * blockDim.y + threadIdx.y};
  c[y * N + x] = a[y * N + x] + b[y * N + x];
}

int main(int argc, char *argv[]) {
  print_device_info();

  const dim3 block_dim{32, 32};
  const dim3 grid_dim{N / block_dim.x, N / block_dim.y};

  std::vector<float> a(N * N);
  std::vector<float> b(N * N);
  std::vector<float> c(N * N);

  for (size_t i{}; i < N; ++i) {
    for (size_t j{}; j < N; ++j) {
      a[i + j * N] = 10 + i;
      b[i + j * N] = static_cast<float>(j) / N;
    }
  }

  float *a_dev{};
  float *b_dev{};
  float *c_dev{};

  cudaMalloc(&a_dev, N * N * sizeof(float));
  cudaMalloc(&b_dev, N * N * sizeof(float));
  cudaMalloc(&c_dev, N * N * sizeof(float));
  cudaMemcpy(a_dev, a.data(), N * N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(b_dev, b.data(), N * N * sizeof(float), cudaMemcpyHostToDevice);

  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  cudaEventRecord(start, 0);
  matrix_add<<<grid_dim, block_dim>>>(a_dev, b_dev, c_dev);
  cudaDeviceSynchronize();
  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("Error: %s\n", cudaGetErrorString(err));
  }


  float elapsed;
  cudaEventElapsedTime(&elapsed, start, end);
  std::cout << "Elapsed GPU: " << elapsed << std::endl;

  cudaMemcpy(c.data(), c_dev, N * N * sizeof(float), cudaMemcpyDeviceToHost);

  // Assert that result is correct
  std::vector<float> c_cpu(N * N);
	// add_matrix(a.data(), b.data(), c_cpu.data(), N);
  // for (size_t i{}; i < N; ++i) {
  //   for (size_t j{}; j < N; ++j) {
  //     assert(c_cpu[i * N + j] == c[i * N + j]);
  //   }
  // }
}
