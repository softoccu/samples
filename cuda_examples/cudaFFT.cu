#include <cuda_runtime.h>
#include <cufft.h>
#include <iostream>

// Macro for error checking CUDA calls
#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// Macro for error checking cuFFT calls
#define CHECK_CUFFT(call) \
    do { \
        cufftResult err = call; \
        if (err != CUFFT_SUCCESS) { \
            std::cerr << "CUFFT error: " << err << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// FFT example
int main() {
    const int NX = 256;  // Size of input data
    const int BATCH = 1; // Batch size

    // Allocate host memory
    cufftComplex *h_data = (cufftComplex *)malloc(sizeof(cufftComplex) * NX * BATCH);

    // Initialize input data
    for (int i = 0; i < NX; ++i) {
        h_data[i].x = static_cast<float>(i);
        h_data[i].y = 0.0f;
    }

    // Allocate device memory
    cufftComplex *d_data;
    CHECK_CUDA(cudaMalloc((void **)&d_data, sizeof(cufftComplex) * NX * BATCH));

    // Copy data from host to device
    CHECK_CUDA(cudaMemcpy(d_data, h_data, sizeof(cufftComplex) * NX * BATCH, cudaMemcpyHostToDevice));

    // Create cuFFT plan
    cufftHandle plan;
    CHECK_CUFFT(cufftPlan1d(&plan, NX, CUFFT_C2C, BATCH));

    // Execute FFT
    CHECK_CUFFT(cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD));

    // Copy result from device back to host
    CHECK_CUDA(cudaMemcpy(h_data, d_data, sizeof(cufftComplex) * NX * BATCH, cudaMemcpyDeviceToHost));

    // Output result
    for (int i = 0; i < NX; ++i) {
        std::cout << "Result[" << i << "] = (" << h_data[i].x << ", " << h_data[i].y << ")" << std::endl;
    }

    // Clean up resources
    CHECK_CUFFT(cufftDestroy(plan));
    CHECK_CUDA(cudaFree(d_data));
    free(h_data);

    return 0;
}