#include <iostream>
#include <random>
#include <omp.h>
#include <cmath>

// Function to calculate the price of a European call option using Monte Carlo method with OpenMP
double monte_carlo_openmp(int num_simulations, double S, double K, double T, double r, double sigma) {
    double sum_payoff = 0.0;
    double dt = T;
    std::random_device rd;
    std::default_random_engine generator(rd());
    std::normal_distribution<double> distribution(0.0, 1.0);

    #pragma omp parallel for reduction(+:sum_payoff)
    for (int i = 0; i < num_simulations; ++i) {
        double Z = distribution(generator);
        double ST = S * std::exp((r - 0.5 * sigma * sigma) * dt + sigma * std::sqrt(dt) * Z);
        double payoff = std::max(ST - K, 0.0);
        sum_payoff += payoff;
    }

    return std::exp(-r * T) * (sum_payoff / num_simulations);
}

// Function to calculate the price of a European call option using Monte Carlo method without OpenMP
double monte_carlo(int num_simulations, double S, double K, double T, double r, double sigma) {
    double sum_payoff = 0.0;
    double dt = T;
    std::random_device rd;
    std::default_random_engine generator(rd());
    std::normal_distribution<double> distribution(0.0, 1.0);

    for (int i = 0; i < num_simulations; ++i) {
        double Z = distribution(generator);
        double ST = S * std::exp((r - 0.5 * sigma * sigma) * dt + sigma * std::sqrt(dt) * Z);
        double payoff = std::max(ST - K, 0.0);
        sum_payoff += payoff;
    }

    return std::exp(-r * T) * (sum_payoff / num_simulations);
}

int main() {
    int num_simulations = 1000000; // Number of Monte Carlo simulations
    double S = 100.0; // Underlying asset price
    double K = 100.0; // Strike price
    double T = 1.0; // Time to maturity in years
    double r = 0.05; // Risk-free interest rate
    double sigma = 0.2; // Volatility

    std::cout << "As random number is used in calculating, result may different: "  << std::endl;
    // Calculate the option price using OpenMP
    double price_openmp = monte_carlo_openmp(num_simulations, S, K, T, r, sigma);
    std::cout << "Option price with OpenMP: " << price_openmp << std::endl;

    // Calculate the option price without OpenMP
    double price = monte_carlo(num_simulations, S, K, T, r, sigma);
    std::cout << "Option price without OpenMP: " << price << std::endl;


    return 0;
}
