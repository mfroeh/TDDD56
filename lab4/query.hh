#include <iostream>

void print_device_info() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0) {
        std::cout << "No CUDA devices found" << std::endl;
    }

    std::cout << "Number of CUDA devices: " << deviceCount << std::endl;

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout << "\nCUDA Device #" << dev << std::endl;
        std::cout << "Device name: " << deviceProp.name << std::endl;
        std::cout << "Compute capability: " << deviceProp.major << "." << deviceProp.minor << std::endl;
        std::cout << "Total global memory: " << deviceProp.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
        std::cout << "Clock rate: " << deviceProp.clockRate / 1000 << " MHz" << std::endl;
        std::cout << "Memory clock rate: " << deviceProp.memoryClockRate / 1000 << " MHz" << std::endl;
        std::cout << "Memory bus width: " << deviceProp.memoryBusWidth << " bits" << std::endl;
        std::cout << "L2 cache size: " << deviceProp.l2CacheSize / 1024 << " KB" << std::endl;
        std::cout << "Max threads per block: " << deviceProp.maxThreadsPerBlock << std::endl;
        std::cout << "Max threads dimensions: (" << deviceProp.maxThreadsDim[0] << ", " << deviceProp.maxThreadsDim[1] << ", " << deviceProp.maxThreadsDim[2] << ")" << std::endl;
        std::cout << "Max grid size: (" << deviceProp.maxGridSize[0] << ", " << deviceProp.maxGridSize[1] << ", " << deviceProp.maxGridSize[2] << ")" << std::endl;
        std::cout << "Number of SMs: " << deviceProp.multiProcessorCount << std::endl;
    }
}
