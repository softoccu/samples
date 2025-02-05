#include <iostream>
#include <cuda_runtime.h>

// Kernel function to compute the area of rectangles
__global__ void integrate_kernel(float *result, float a, float b, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    float h = (b - a) / n;
    
    if (idx < n) {
        float x = a + idx * h + h / 2.0f; // Midpoint of the rectangle
        float fx = x * x; // Function to integrate, here f(x) = x^2
        result[idx] = fx * h; // Area of the rectangle
    }
}

// Function to compute the integral using CUDA
float integrate(float a, float b, int n) {
    float *d_result, *h_result;
    size_t size = n * sizeof(float);
    
    // Allocate host memory
    h_result = (float*)malloc(size);
    
    // Allocate device memory
    cudaMalloc((void**)&d_result, size);
    
    // Define block and grid sizes
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    
    // Launch kernel
    integrate_kernel<<<gridSize, blockSize>>>(d_result, a, b, n);
    
    // Copy result back to host
    cudaMemcpy(h_result, d_result, size, cudaMemcpyDeviceToHost);
    
    // Sum up the areas of all rectangles
    float integral = 0.0f;
    for (int i = 0; i < n; ++i) {
        integral += h_result[i];
    }
    
    // Free memory
    cudaFree(d_result);
    free(h_result);
    
    return integral;
}

int main() {
    float a = 0.0f; // Lower limit of integration
    float b = 1.0f; // Upper limit of integration
    int n = 1000000; // Number of rectangles
    
    float result = integrate(a, b, n);
    std::cout << "The integral of f(x) = x^2 from " << a << " to " << b << " is " << result << std::endl;
    
    return 0;
}