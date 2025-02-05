#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <algorithm>

#define N 1024  // Number of items
#define W 50    // Maximum weight of the knapsack


#define CHECK_CUDA_ERROR(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// CUDA kernel function to solve the knapsack problem using multiple dp arrays
__global__ void knapsackKernel(int* weights, int* values, int* dp_old, int* dp_new) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid <= W) {
        for (int i = 0; i < N; ++i) {
            if (weights[i] <= tid) {
                dp_new[tid] = max(dp_old[tid], dp_old[tid - weights[i]] + values[i]);
            } else {
                dp_new[tid] = dp_old[tid];
            }
            __syncthreads(); // Ensure all threads have updated dp_new before next iteration
            // Swap dp arrays (dp_new becomes dp_old)
            int* temp = dp_old;
            dp_old = dp_new;
            dp_new = temp;
        }
    }
}

int knapsackHost(const std::vector<int>& weights, const std::vector<int>& values) {
    std::vector<int> dp(W + 1, 0);

    for (int i = 0; i < N; ++i) {
        for (int w = W; w >= weights[i]; --w) {
            dp[w] = std::max(dp[w], dp[w - weights[i]] + values[i]);
        }
    }

    return dp[W];
}

int main() {
    // Define weights and values of items
    std::vector<int> weights(N);
    std::vector<int> values(N);

    // Initialize weights and values randomly
    for (int i = 0; i < N; ++i) {
        weights[i] = rand() % 10 + 1;  // Random weight between 1 and 10
        values[i] = rand() % 100 + 1;  // Random value between 1 and 100
    }

    // Print weights and values of items
    std::cout << "Items (weight, value):" << std::endl;
    for (int i = 0; i < N; ++i) {
        std::cout << "(" << weights[i] << ", " << values[i] << ") ";
        if ((i + 1) % 10 == 0) std::cout << std::endl;
    }
    std::cout << std::endl;

    // Print knapsack capacity
    std::cout << "Knapsack capacity: " << W << std::endl;

    // Allocate memory on the device
    int* d_weights;
    int* d_values;
    int* d_dp_old;
    int* d_dp_new;

    CHECK_CUDA_ERROR(cudaMalloc(&d_weights, N * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_values, N * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_dp_old, (W + 1) * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_dp_new, (W + 1) * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMemset(d_dp_old, 0, (W + 1) * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMemset(d_dp_new, 0, (W + 1) * sizeof(int)));

    // Copy data to the device
    CHECK_CUDA_ERROR(cudaMemcpy(d_weights, weights.data(), N * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_values, values.data(), N * sizeof(int), cudaMemcpyHostToDevice));

    // Define block size and grid size
    int blockSize = 256;
    int gridSize = (W + blockSize - 1) / blockSize;

    // Launch the kernel
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    knapsackKernel<<<gridSize, blockSize>>>(d_weights, d_values, d_dp_old, d_dp_new);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // Copy the result back to the host
    std::vector<int> dp(W + 1);
    CHECK_CUDA_ERROR(cudaMemcpy(dp.data(), d_dp_old, (W + 1) * sizeof(int), cudaMemcpyDeviceToHost));

    // Find the maximum value that can be carried in the knapsack using CUDA
    int maxValueCUDA = dp[W];

    std::cout << "Maximum value in knapsack (CUDA): " << maxValueCUDA << std::endl;

    // Solve the knapsack problem on the host

    int maxValueHost = knapsackHost(weights, values);

    std::cout << "Maximum value in knapsack (Host): " << maxValueHost << std::endl;


    // Compare results
    if (maxValueCUDA == maxValueHost) {
        std::cout << "CUDA and Host results match!" << std::endl;
    } else {
        std::cout << "CUDA and Host results do not match!" << std::endl;
    }

    // Free device memory
    cudaFree(d_weights);
    cudaFree(d_values);
    cudaFree(d_dp_old);
    cudaFree(d_dp_new);

    return 0;
}
