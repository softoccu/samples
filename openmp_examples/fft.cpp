#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <omp.h>

const double PI = 3.141592653589793238460;

typedef std::complex<double> Complex;
typedef std::vector<Complex> CArray;

// Cooley-Tukey FFT (recursive with OpenMP)
void fft_openmp(CArray& x) {
    const size_t N = x.size();
    if (N <= 1) return;

    // Divide
    CArray even = CArray(N / 2);
    CArray odd = CArray(N / 2);
    for (size_t i = 0; i < N / 2; ++i) {
        even[i] = x[i * 2];
        odd[i] = x[i * 2 + 1];
    }

    // Conquer
    #pragma omp parallel sections
    {
        #pragma omp section
        fft_openmp(even);
        #pragma omp section
        fft_openmp(odd);
    }

    // Combine
    for (size_t k = 0; k < N / 2; ++k) {
        Complex t = std::polar(1.0, -2 * PI * k / N) * odd[k];
        x[k] = even[k] + t;
        x[k + N / 2] = even[k] - t;
    }
}

// Cooley-Tukey FFT (recursive without OpenMP)
void fft(CArray& x) {
    const size_t N = x.size();
    if (N <= 1) return;

    // Divide
    CArray even = CArray(N / 2);
    CArray odd = CArray(N / 2);
    for (size_t i = 0; i < N / 2; ++i) {
        even[i] = x[i * 2];
        odd[i] = x[i * 2 + 1];
    }

    // Conquer
    fft(even);
    fft(odd);

    // Combine
    for (size_t k = 0; k < N / 2; ++k) {
        Complex t = std::polar(1.0, -2 * PI * k / N) * odd[k];
        x[k] = even[k] + t;
        x[k + N / 2] = even[k] - t;
    }
}

int main() {
    const int N = 16; // Must be a power of 2
    CArray data(N);
    CArray data_copy(N);

    // Initialize the input array with some data (example: a sine wave)
    std::cout << "Input data:\n";
    for (int i = 0; i < N; ++i) {
        data[i] = std::sin(2 * PI * i / N);
        data_copy[i] = data[i];
        std::cout << data[i] << "\n";
    }

    // Perform FFT with OpenMP
    fft_openmp(data);

    // Perform FFT without OpenMP
    fft(data_copy);

    // Verify the results
    bool is_correct = true;
    std::cout << "\nFFT results with OpenMP:\n";
    for (int i = 0; i < N; ++i) {
        std::cout << data[i] << "\n";
    }

    std::cout << "\nFFT results without OpenMP:\n";
    for (int i = 0; i < N; ++i) {
        std::cout << data_copy[i] << "\n";
        if (std::abs(data[i] - data_copy[i]) > 1e-6) {
            is_correct = false;
        }
    }

    if (is_correct) {
        std::cout << "\nFFT results are correct." << std::endl;
    } else {
        std::cout << "\nFFT results are incorrect." << std::endl;
    }

    return 0;
}