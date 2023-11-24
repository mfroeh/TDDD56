#include <iomanip>
#include <iostream>
#include <vector>

const size_t N{512};
const dim3 grid_dim{16, 16};
const dim3 block_dim{N / 16, N / 16};

__global__ void matrix_add(float *a, float *b, float *c) {
  unsigned x{blockIdx.x * blockDim.x + threadIdx.x};
  unsigned y{blockIdx.y * blockDim.y + threadIdx.y};
  c[y * N + x] = a[y * N + x] + b[y * N + x];
}

int main(int argc, char *argv[]) {
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

  matrix_add<<<grid_dim, block_dim>>>(a_dev, b_dev, c_dev);
  cudaDeviceSynchronize();

  cudaMemcpy(c.data(), c_dev, N * N * sizeof(float), cudaMemcpyDeviceToHost);

  std::cout << std::fixed << std::setprecision(2);
  for (size_t i{}; i < N; ++i) {
    for (size_t j{}; j < N; ++j) {
      std::cout << c[i + j * N] << " ";
    }
    std::cout << '\n';
  }
  std::cout.flush();
}
