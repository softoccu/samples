#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <random>


using namespace std;
// Function to generate a random string of uppercase letters of given length
std::string generate_random_string(size_t length) {
    const std::string CHARACTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
    std::random_device rd;  // Seed for the random number engine
    std::mt19937 generator(rd());  // Standard mersenne_twister_engine seeded with rd()
    std::uniform_int_distribution<> distribution(0, CHARACTERS.size() - 1);

    std::string random_string;
    for (size_t i = 0; i < length; ++i) {
        random_string += CHARACTERS[distribution(generator)];
    }
    return random_string;
}

// Kernel to compute the LCS matrix
// using diagonal-traverse updade DP matrix, for dp[i][j] depend on dp[i-1][j] and dp[i][j-1]
__global__ void lcs_kernel(char* X, char* Y, int* L, int m, int n, int offset) {
    int i, j;
    if(offset < n){
        i = threadIdx.x + 1;
        j = offset - threadIdx.x + 1;
    } else {
        i = offset - (n - 1) + threadIdx.x + 1;
        j = (n - 1) - threadIdx.x + 1;
    }

    if (i >= 1 && i <= m && j >= 1 && j <= n) {
        if (X[i - 1] == Y[j - 1])
            L[i * (n + 1) + j] = L[(i - 1) * (n + 1) + (j - 1)] + 1;
        else
            L[i * (n + 1) + j] = max(L[(i - 1) * (n + 1) + j], L[i * (n + 1) + (j - 1)]);
    }
}

// Host function to compute LCS using CUDA
std::pair<int, std::vector<int>> lcs_cuda(const std::string& X, const std::string& Y) {
    int m = X.length();
    int n = Y.length();

    // Allocate host memory
    int* h_L = (int*)malloc((m + 1) * (n + 1) * sizeof(int));
    char* h_X = (char*)X.c_str();
    char* h_Y = (char*)Y.c_str();

    // Allocate device memory
    int* d_L;
    char* d_X;
    char* d_Y;
    cudaMalloc((void**)&d_L, (m + 1) * (n + 1) * sizeof(int));
    cudaMalloc((void**)&d_X, m * sizeof(char));
    cudaMalloc((void**)&d_Y, n * sizeof(char));

    // Copy data from host to device
    cudaMemcpy(d_X, h_X, m * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Y, h_Y, n * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemset(d_L, 0, (m + 1) * (n + 1) * sizeof(int));

    // Define block and grid sizes
    int blocksize = 256;
    dim3 blockSize(blocksize, 1, 1);
    dim3 gridSize((min(m,n) + (blocksize - 1))/blocksize , 1, 1);

    // Launch kernel
    for(int offset = 0; offset < (m+n); ++offset){
        lcs_kernel<<<gridSize, blockSize>>>(d_X, d_Y, d_L, m, n, offset);
        cudaDeviceSynchronize();
    }

    // Copy result back to host
    cudaMemcpy(h_L, d_L, (m + 1) * (n + 1) * sizeof(int), cudaMemcpyDeviceToHost);

    // Get LCS length
    int lcs_length = h_L[m * (n + 1) + n];

    // Copy the entire LCS matrix for verification
    std::vector<int> lcs_matrix((m + 1) * (n + 1));
    memcpy(lcs_matrix.data(), h_L, (m + 1) * (n + 1) * sizeof(int));

    // Free memory
    cudaFree(d_L);
    cudaFree(d_X);
    cudaFree(d_Y);
    free(h_L);

    return {lcs_length, lcs_matrix};
}

// Host function to compute LCS using dynamic programming
std::string lcs_host(const std::string& X, const std::string& Y) {
    int m = X.length();
    int n = Y.length();
    std::vector<std::vector<int>> L(m + 1, std::vector<int>(n + 1, 0));

    for (int i = 1; i <= m; ++i) {
        for (int j = 1; j <= n; ++j) {
            if (X[i - 1] == Y[j - 1])
                L[i][j] = L[i - 1][j - 1] + 1;
            else
                L[i][j] = std::max(L[i - 1][j], L[i][j - 1]);
        }
    }

    // Reconstruct LCS
    int index = L[m][n];
    std::string lcs(index, ' ');
    int i = m, j = n;
    while (i > 0 && j > 0) {
        if (X[i - 1] == Y[j - 1]) {
            lcs[--index] = X[i - 1];
            --i;
            --j;
        } else if (L[i - 1][j] > L[i][j - 1]) {
            --i;
        } else {
            --j;
        }
    }

    return lcs;
}

int main() {
    std::string X = generate_random_string(256);
    std::string Y = generate_random_string(256);
    std::cout << X << endl;
    std::cout << Y << endl;

    auto [lcs_length_cuda, lcs_matrix_cuda] = lcs_cuda(X, Y);
    std::string lcs_host_result = lcs_host(X, Y);

    std::cout << "Length of LCS (CUDA): " << lcs_length_cuda << std::endl;
    std::cout << "LCS (Host): " << lcs_host_result <<  ",length = " << lcs_host_result.size() << std::endl;

    return 0;
}