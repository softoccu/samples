#include <iostream>
#include <vector>
#include <omp.h>
#include <cmath>

// Function to find prime numbers using Sieve of Eratosthenes with OpenMP
std::vector<bool> sieve_openmp(const int limit) {
    std::vector<bool> is_prime(limit + 1, true);
    is_prime[0] = is_prime[1] = false;
    int sqrt_limit = int(sqrt(limit));
    // Parallel region starts here
    //#pragma omp parallel
    {
        #pragma omp for schedule(dynamic)
        for (int i = 2; i <= sqrt_limit; ++i) {
            if (is_prime[i]) {
                // Mark all multiples of i as non-prime
                #pragma omp parallel for 
                for (int j = i*i ; j <= limit; j += i) {
                    is_prime[j] = false;
                }
            }
        }
    }

    return is_prime;
}

// Function to find prime numbers using Sieve of Eratosthenes without OpenMP
std::vector<bool> sieve(const int limit) {
    std::vector<bool> is_prime(limit + 1, true);
    is_prime[0] = is_prime[1] = false;

    for (int i = 2; i * i <= limit; ++i) {
        if (is_prime[i]) {
            // Mark all multiples of i as non-prime
            for (int j = i * i; j <= limit; j += i) {
                is_prime[j] = false;
            }
        }
    }

    return is_prime;
}

int main() {
    const int limit = 1000000;

    // Find primes using OpenMP
    std::vector<bool> is_prime_openmp = sieve_openmp(limit);

    // Find primes without OpenMP
    std::vector<bool> is_prime = sieve(limit);

    // Verify the results and print primes
    bool is_correct = true;
    int cnt = 0;
    for (int i = 2; i <= limit; ++i) {
        if (is_prime_openmp[i]) ++cnt;
    }
    std::cout <<  cnt << " primes found using OpenMP:\n";
    cnt = 0;
    for (int i = 2; i <= limit; ++i) {
        if (is_prime[i]) ++cnt;
    }
    std::cout <<  cnt << " primes found without OpenMP:\n";
    for (int i = 2; i <= limit; ++i) {
        if (is_prime[i]) {
            //std::cout << i << " ";
        }
        if (is_prime[i] != is_prime_openmp[i]) {
            is_correct = false;
            std::cout << " i = " << i << " host:" << is_prime[i] << " openmp: " << is_prime_openmp[i] << std::endl;
        }
    }

    if (is_correct) {
        std::cout << "\n\nResults are correct." << std::endl;
    } else {
        std::cout << "\n\nResults are incorrect." << std::endl;
    }

    return 0;
}