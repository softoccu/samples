#include <iostream>
#include <cmath>
#include <curand_kernel.h>
#include <cuda_runtime.h>

const int NUM_THREADS = 256;
const int NUM_BLOCKS = 256;
const int NUM_SAMPLES = NUM_THREADS * NUM_BLOCKS ;

// Kernel to initialize the random number generator state
__global__ void setup_kernel(curandState *state, unsigned long seed) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, id, 0, &state[id]);
}

// Kernel to perform Monte Carlo simulation for European call option pricing
__global__ void monte_carlo_option_price(curandState *state, float *results, float S, float K, float r, float T, float sigma) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    curandState localState = state[id];
    float S_T; // Variable to store the simulated stock price at maturity
    float payoff; // Variable to store the payoff of the option
    
    // Loop to simulate multiple paths
    for (int i = 0; i < 1000; ++i) {
        // Generate a random number from the standard normal distribution
        float gauss_bm = curand_normal(&localState);
        // Calculate the simulated stock price at maturity using the geometric Brownian motion formula
        S_T = S * exp((r - 0.5f * sigma * sigma) * T + sigma * sqrt(T) * gauss_bm);
        // Calculate the payoff of the call option
        payoff = max(S_T - K, 0.0f);
        // Accumulate the discounted payoff
        results[id] += exp(-r * T) * payoff;
    }
    // Save the local random number generator state back to global memory
    state[id] = localState;
}

int main() {
    // Option parameters
    float S = 100.0f;   // Initial stock price
    float K = 100.0f;   // Strike price
    float r = 0.05f;    // Risk-free interest rate
    float T = 1.0f;     // Time to maturity (1 year)
    float sigma = 0.2f; // Volatility

    float *d_results;
    curandState *d_state;

    // Allocate memory on the device
    cudaMalloc(&d_results, NUM_THREADS * NUM_BLOCKS * sizeof(float));
    cudaMalloc(&d_state, NUM_THREADS * NUM_BLOCKS * sizeof(curandState));
    cudaMemset(d_results, 0, NUM_THREADS * NUM_BLOCKS * sizeof(float));

    // Initialize the random number generator states on the device
    setup_kernel<<<NUM_BLOCKS, NUM_THREADS>>>(d_state, time(0));
    cudaDeviceSynchronize();

    // Perform the Monte Carlo simulation on the device
    monte_carlo_option_price<<<NUM_BLOCKS, NUM_THREADS>>>(d_state, d_results, S, K, r, T, sigma);
    cudaDeviceSynchronize();

    float h_results[NUM_THREADS * NUM_BLOCKS];
    // Copy the results from the device to the host
    cudaMemcpy(h_results, d_results, NUM_THREADS * NUM_BLOCKS * sizeof(float), cudaMemcpyDeviceToHost);

    float option_price = 0.0f;
    // Calculate the average option price
    for (int i = 0; i < NUM_THREADS * NUM_BLOCKS; ++i) {
        option_price += h_results[i];
    }
    option_price /= (NUM_SAMPLES * 1000);

    // Print the estimated option price
    std::cout << "Estimated European Call Option Price: " << option_price << std::endl;

    // Free the allocated memory on the device
    cudaFree(d_results);
    cudaFree(d_state);

    return 0;
}
