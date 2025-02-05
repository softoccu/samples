#include <iostream>
#include <omp.h>
#include <cmath>

// Function to compute the value of the function to integrate
double function(double x) {
    return std::sin(x); // Example function: sin(x)
}

// Function to compute the integral using the trapezoidal rule with OpenMP
double integrate_openmp(double (*func)(double), double a, double b, int n) {
    double h = (b - a) / n;
    double integral = 0.0;

    #pragma omp parallel for reduction(+:integral)
    for (int i = 1; i < n; ++i) {
        integral += func(a + i * h);
    }

    integral += (func(a) + func(b)) / 2.0;
    integral *= h;

    return integral;
}

// Function to compute the integral using the trapezoidal rule without OpenMP
double integrate(double (*func)(double), double a, double b, int n) {
    double h = (b - a) / n;
    double integral = 0.0;

    for (int i = 1; i < n; ++i) {
        integral += func(a + i * h);
    }

    integral += (func(a) + func(b)) / 2.0;
    integral *= h;

    return integral;
}

int main() {
    double a = 0.0; // Lower limit of integration
    double b = 10.0; // Upper limit of integration
    int n = 1000000; // Number of subintervals

    // Compute the integral using OpenMP
    double result_openmp = integrate_openmp(function, a, b, n);
    // Compute the integral without OpenMP
    double result = integrate(function, a, b, n);

    // Print the results
    std::cout << "Integral with OpenMP: " << result_openmp << std::endl;
    std::cout << "Integral without OpenMP: " << result << std::endl;

    // Verify the results
    if (std::abs(result_openmp - result) < 1e-6) {
        std::cout << "Results are correct." << std::endl;
    } else {
        std::cout << "Results are incorrect." << std::endl;
    }

    return 0;
}
