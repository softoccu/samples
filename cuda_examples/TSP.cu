#include <iostream>
#include <vector>
#include <cmath>
#include <curand_kernel.h>
#include <limits>

#define N 128            // Number of cities
#define ALPHA 1.0f       // Pheromone importance factor
#define BETA 5.0f        // Heuristic factor
#define RHO 0.5f         // Pheromone evaporation coefficient
#define Q 100.0f         // Pheromone constant
#define MAX_ITER 1000    // Maximum number of iterations

// Structure to represent a city with x and y coordinates
struct City {
    float x, y;
};

// Function to calculate the distance between two cities
__device__ float distance(const City& a, const City& b) {
    return sqrtf((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y));
}

// Kernel function to initialize the paths of ants
__global__ void initializeAnts(int* paths, curandState* states, int n, int numAnts) {
    int antId = blockIdx.x * blockDim.x + threadIdx.x;
    if (antId < numAnts) {
        curand_init(1234, antId, 0, &states[antId]);
        for (int i = 0; i < n; ++i) {
            paths[antId * n + i] = i;
        }
        // Shuffle the initial path
        for (int i = 0; i < n; ++i) {
            int j = curand(&states[antId]) % n;
            int temp = paths[antId * n + i];
            paths[antId * n + i] = paths[antId * n + j];
            paths[antId * n + j] = temp;
        }
    }
}

// Kernel function for the ant colony optimization process
__global__ void antColonyOptimization(City* cities, float* pheromones, int* paths, float* lengths, curandState* states, int n, int numAnts) {
    int antId = blockIdx.x * blockDim.x + threadIdx.x;
    if (antId < numAnts) {
        // Each ant starts from a random city
        int* path = &paths[antId * n];
        curandState localState = states[antId];
        for (int i = 0; i < n - 1; ++i) {
            int currentCity = path[i];
            float totalPheromone = 0.0f;
            float probabilities[N];
            for (int j = i + 1; j < n; ++j) {
                int nextCity = path[j];
                probabilities[j] = powf(pheromones[currentCity * n + nextCity], ALPHA) *
                                   powf(1.0f / distance(cities[currentCity], cities[nextCity]), BETA);
                totalPheromone += probabilities[j];
            }
            float r = curand_uniform(&localState) * totalPheromone;
            float cumulative = 0.0f;
            for (int j = i + 1; j < n; ++j) {
                cumulative += probabilities[j];
                if (cumulative >= r) {
                    int temp = path[i + 1];
                    path[i + 1] = path[j];
                    path[j] = temp;
                    break;
                }
            }
        }
        // Calculate the path length
        float length = 0.0f;
        for (int i = 0; i < n - 1; ++i) {
            length += distance(cities[path[i]], cities[path[i + 1]]);
        }
        length += distance(cities[path[n - 1]], cities[path[0]]);
        lengths[antId] = length;
    }
}

// Kernel function to update the pheromones on paths
__global__ void updatePheromones(float* pheromones, int* paths, float* lengths, int n, int numAnts) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n * n) {
        int i = id / n;
        int j = id % n;
        pheromones[i * n + j] *= (1.0f - RHO);
    }
    __syncthreads();
    if (id < numAnts) {
        int* path = &paths[id * n];
        float length = lengths[id];
        for (int i = 0; i < n - 1; ++i) {
            int from = path[i];
            int to = path[i + 1];
            atomicAdd(&pheromones[from * n + to], Q / length);
            atomicAdd(&pheromones[to * n + from], Q / length);
        }
        int from = path[n - 1];
        int to = path[0];
        atomicAdd(&pheromones[from * n + to], Q / length);
        atomicAdd(&pheromones[to * n + from], Q / length);
    }
}

// Function to check CUDA errors
void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << msg << " (" << cudaGetErrorString(err) << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main() {
    City cities[N];        // Array of cities
    float pheromones[N * N]; // Pheromone matrix
    int* paths;            // Path for each ant
    float* lengths;        // Length of each path
    curandState* states;   // Random states for each ant

    // Allocate host memory
    paths = (int*)malloc(N * N * sizeof(int));
    lengths = (float*)malloc(N * sizeof(float));
    if (paths == nullptr || lengths == nullptr) {
        std::cerr << "Failed to allocate host memory" << std::endl;
        return -1;
    }

    // Initialize cities with random coordinates
    for (int i = 0; i < N; ++i) {
        cities[i].x = 100*(static_cast<float>(rand()) / RAND_MAX);
        cities[i].y = 100*(static_cast<float>(rand()) / RAND_MAX);
    }

    // Print city coordinates
    std::cout << "City coordinates:" << std::endl;
    for (int i = 0; i < N; ++i) {
        std::cout << "City " << i << ": (" << cities[i].x << ", " << cities[i].y << ")" << std::endl;
    }

    // Initialize pheromones with initial value
    for (int i = 0; i < N * N; ++i) {
        pheromones[i] = 1.0f;
    }

    // Allocate device memory
    City* d_cities;
    float* d_pheromones;
    int* d_paths;
    float* d_lengths;
    checkCudaError(cudaMalloc(&d_cities, N * sizeof(City)), "Failed to allocate device memory for cities");
    checkCudaError(cudaMalloc(&d_pheromones, N * N * sizeof(float)), "Failed to allocate device memory for pheromones");
    checkCudaError(cudaMalloc(&d_paths, N * N * sizeof(int)), "Failed to allocate device memory for paths");
    checkCudaError(cudaMalloc(&d_lengths, N * sizeof(float)), "Failed to allocate device memory for lengths");
    checkCudaError(cudaMalloc(&states, N * sizeof(curandState)), "Failed to allocate device memory for states");

    // Copy data to device
    checkCudaError(cudaMemcpy(d_cities, cities, N * sizeof(City), cudaMemcpyHostToDevice), "Failed to copy cities to device");
    checkCudaError(cudaMemcpy(d_pheromones, pheromones, N * N * sizeof(float), cudaMemcpyHostToDevice), "Failed to copy pheromones to device");

    // Define block size and grid size
    int blockSize = 128;
    int gridSize = (N + blockSize - 1) / blockSize;

    // Initialize ants
    initializeAnts<<<gridSize, blockSize>>>(d_paths, states, N, N);
    checkCudaError(cudaGetLastError(), "Kernel launch failed for initializeAnts");
    checkCudaError(cudaDeviceSynchronize(), "Kernel execution failed for initializeAnts");

    // Variables to store the best path and its length
    float bestLength = std::numeric_limits<float>::max();
    std::vector<int> bestPath(N);

    // Main loop for ACO iterations
    for (int iter = 0; iter < MAX_ITER; ++iter) {
        // Perform ACO by calculating paths
        antColonyOptimization<<<gridSize, blockSize>>>(d_cities, d_pheromones, d_paths, d_lengths, states, N, N);
        checkCudaError(cudaGetLastError(), "Kernel launch failed for antColonyOptimization");
        checkCudaError(cudaDeviceSynchronize(), "Kernel execution failed for antColonyOptimization");

        // Update pheromones based on paths found
        updatePheromones<<<gridSize, blockSize>>>(d_pheromones, d_paths, d_lengths, N, N);
        checkCudaError(cudaGetLastError(), "Kernel launch failed for updatePheromones");
        checkCudaError(cudaDeviceSynchronize(), "Kernel execution failed for updatePheromones");

        // Copy lengths back to host to find the best path
        checkCudaError(cudaMemcpy(lengths, d_lengths, N * sizeof(float), cudaMemcpyDeviceToHost), "Failed to copy lengths to host");
        checkCudaError(cudaMemcpy(paths, d_paths, N * N * sizeof(int), cudaMemcpyDeviceToHost), "Failed to copy paths to host");

        // Find the best path in this iteration
        for (int i = 0; i < N; ++i) {
            if (lengths[i] < bestLength) {
                bestLength = lengths[i];
                for (int j = 0; j < N; ++j) {
                    bestPath[j] = paths[i * N + j];
                }
            }
        }
    }

    // Print the best path and its length
    std::cout << "Best path length: " << bestLength << std::endl;
    std::cout << "Best path: ";
    for (int i = 0; i < N; ++i) {
        std::cout << bestPath[i] << " ";
    }
    std::cout << std::endl;

    // Free device memory
    cudaFree(d_cities);
    cudaFree(d_pheromones);
    cudaFree(d_paths);
    cudaFree(d_lengths);
    cudaFree(states);

    // Free host memory
    free(paths);
    free(lengths);

    return 0;
}