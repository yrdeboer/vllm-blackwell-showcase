#include <iostream>
#include <cuda_runtime.h>

__global__ void matrixDiagnostics() {
    // Empty kernel purely to test compilation
}

int main() {
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);

    if (error != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(error) << std::endl;
        return 1;
    }

    std::cout << "System architecture validation OK." << std::endl;
    std::cout << "Number of devices detected by CUDA: " << deviceCount << std::endl;
    return 0;
}
