#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>

// CUDA kernel function to perform matrix multiplication
__global__ void matrixMultiply(const float* A, const float* B, float* C, int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float value = 0.0f;
        for (int k = 0; k < K; ++k) {
            value += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = value;
    }
}

// Function to print a matrix
void printMatrix(const float* matrix, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << matrix[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }
}

// Host function to verify the result of matrix multiplication
void verifyResult(const float* A, const float* B, const float* C, int M, int K, int N) {
    bool valid = true;
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float value = 0.0f;
            for (int k = 0; k < K; ++k) {
                value += A[i * K + k] * B[k * N + j];
            }
            if (fabs(value - C[i * N + j]) > 1e-5) {
                std::cout << "Mismatch at (" << i << ", " << j << "): GPU = " << C[i * N + j] << ", CPU = " << value << std::endl;
                valid = false;
            }
        }
    }
    if (valid) {
        std::cout << "Results are correct!" << std::endl;
    } else {
        std::cout << "Results are incorrect!" << std::endl;
    }
}

int main() {
    // Define matrix dimensions (between 10 and 20)
    int M = 12;  // Number of rows in A and C
    int K = 15;  // Number of columns in A and rows in B
    int N = 18;  // Number of columns in B and C

    // Allocate host memory
    float *h_A = (float*)malloc(M * K * sizeof(float));
    float *h_B = (float*)malloc(K * N * sizeof(float));
    float *h_C = (float*)malloc(M * N * sizeof(float));

    // Initialize matrices A and B with random values
    for (int i = 0; i < M * K; ++i) {
        h_A[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    for (int i = 0; i < K * N; ++i) {
        h_B[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice);

    // Define the size of blocks and grids
    dim3 dimBlock(16, 16);
    dim3 dimGrid((N + dimBlock.x - 1) / dimBlock.x, (M + dimBlock.y - 1) / dimBlock.y);

    // Launch the CUDA kernel
    matrixMultiply<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, M, K, N);

    // Wait for the CUDA kernel to finish
    cudaDeviceSynchronize();

    // Copy the result from device to host
    cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Verify the result on the host
    verifyResult(h_A, h_B, h_C, M, K, N);

    // Print the matrices
    std::cout << "Matrix A:" << std::endl;
    printMatrix(h_A, M, K);
    std::cout << "Matrix B:" << std::endl;
    printMatrix(h_B, K, N);
    std::cout << "Matrix C (Result):" << std::endl;
    printMatrix(h_C, M, N);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}