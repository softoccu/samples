#include <iostream>
#include <vector>
#include <omp.h>
#include <random>
#include <iomanip>

// Function to solve the knapsack problem using dynamic programming with OpenMP
int knapsack_openmp(int W, const std::vector<int>& weights, const std::vector<int>& values) {
    int n = weights.size();
    std::vector<int> dp_old(W + 1, 0);
    std::vector<int> dp_new(W + 1, 0);

    for (int i = 0; i < n; ++i) {
        #pragma omp parallel for
        for (int w = 0; w <= W; ++w) {
            if (w >= weights[i]) {
                dp_new[w] = std::max(dp_old[w], dp_old[w - weights[i]] + values[i]);
            } else {
                dp_new[w] = dp_old[w];
            }
        }
        dp_old.swap(dp_new);
    }

    return dp_old[W];
}

// Function to solve the knapsack problem using dynamic programming without OpenMP
int knapsack(int W, const std::vector<int>& weights, const std::vector<int>& values) {
    int n = weights.size();
    std::vector<int> dp(W + 1, 0);

    for (int i = 0; i < n; ++i) {
        for (int w = W; w >= weights[i]; --w) {
            dp[w] = std::max(dp[w], dp[w - weights[i]] + values[i]);
        }
    }

    return dp[W];
}

// Function to generate random weights and values for the knapsack problem
void generate_knapsack_data(int n, int max_weight, int max_value, std::vector<int>& weights, std::vector<int>& values) {
    std::random_device rd;
    std::default_random_engine generator(rd());
    std::uniform_int_distribution<int> weight_distribution(1, max_weight);
    std::uniform_int_distribution<int> value_distribution(1, max_value);

    for (int i = 0; i < n; ++i) {
        weights.push_back(weight_distribution(generator));
        values.push_back(value_distribution(generator));
    }
}

int main() {
    int n = 50; // Number of items
    int W = 100; // Maximum weight of the knapsack
    int max_weight = 20; // Maximum weight of an item
    int max_value = 100; // Maximum value of an item

    std::vector<int> weights;
    std::vector<int> values;

    // Generate random weights and values
    generate_knapsack_data(n, max_weight, max_value, weights, values);

    // Print the generated weights and values
    std::cout << "Weights: ";
    for (int weight : weights) {
        std::cout << weight << " ";
    }
    std::cout << std::endl;

    std::cout << "Values: ";
    for (int value : values) {
        std::cout << value << " ";
    }
    std::cout << std::endl;

    // Solve the knapsack problem using OpenMP
    int max_value_openmp = knapsack_openmp(W, weights, values);
    std::cout << "Maximum value with OpenMP: " << max_value_openmp << std::endl;

    // Solve the knapsack problem without OpenMP
    int max_value_without_openmp = knapsack(W, weights, values);
    std::cout << "Maximum value without OpenMP: " << max_value_without_openmp << std::endl;

    // Verify the results
    if (max_value_openmp == max_value_without_openmp) {
        std::cout << "Results are correct." << std::endl;
    } else {
        std::cout << "Results are incorrect." << std::endl;
    }

    return 0;
}