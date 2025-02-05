#include <iostream>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256  // Number of threads per block

// CUDA kernel to mark multiples of a given prime
__global__ void sieve_kernel(bool* primes, int start, int end, int prime, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Mark multiples of prime within the range [start, end]
    if (idx * prime + start <= end && (idx * prime + start) <= n) {
        primes[idx * prime + start] = false;
    }
}

// Function to find primes using CUDA
void find_primes(int n) {
    // Array to hold the prime status of numbers
    // you may use bitset to save memory, here to show how to use CUDA, just keep the code simple
    bool *primes = new bool[n + 1];  
    for (int i = 0; i <= n; ++i) {
        primes[i] = true;
    }
    primes[0] = primes[1] = false;  // 0 and 1 are not prime

    // Allocate memory on GPU
    bool* d_primes;
    cudaMalloc(&d_primes, (n + 1) * sizeof(bool));
    cudaMemcpy(d_primes, primes, (n + 1) * sizeof(bool), cudaMemcpyHostToDevice);

    int sqrt_n = sqrt(n);

    for (int prime = 2; prime <= sqrt_n; ++prime) {
        if (primes[prime]) {
            int start = prime * prime;
            int end = n;

            // Launch kernel to mark multiples of prime as false (non-prime)
            int num_threads = (end - start) / prime + 1;
            int num_blocks = (num_threads + BLOCK_SIZE - 1) / BLOCK_SIZE;

            sieve_kernel<<<num_blocks, BLOCK_SIZE>>>(d_primes, start, end, prime, n);
        }
    }

    // Copy result back to host
    cudaMemcpy(primes, d_primes, (n + 1) * sizeof(bool), cudaMemcpyDeviceToHost);

    // Print primes
    int cnt = 0;
    for (int i = 2; i <= n; ++i) {
        if (primes[i]) {
            ++cnt;
            //std::cout << i << " ";
        }
    }
    std::cout << std::endl;
    std::cout << cnt << " primes found by CUDA from 2 to " << n << std::endl;

    // Free allocated memory
    delete[] primes;
    cudaFree(d_primes);
}

void sieveHost(bool* h_prime, int n) {
    for (int i = 2; i <= sqrt(n); ++i) {
        if (h_prime[i]) {
            for (int j = i * i; j <= n; j += i) {
                h_prime[j] = false;
            }
        }
    }
}


int main() {
    int N = 1000000;  // Limit for prime search, you can adjust this number
    find_primes(N);    // Call the function to find primes

    // Allocate memory on host
    bool* h_prime = new bool[N + 1];
    std::fill_n(h_prime, N + 1, true);
    h_prime[0] = h_prime[1] = false;

    // Perform sieve on host
    sieveHost(h_prime, N);

    int cnt = 0;
    for (int i = 2; i <= N; ++i) {
        if (h_prime[i]) {
            ++cnt;
            //std::cout << i << " ";
        }
    }
    std::cout << cnt << " primes found by host from 2 to " << N << std::endl;

    return 0;
}

