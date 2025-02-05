#include <iostream>
#include <vector>
#include <random>
#include <omp.h>

// Function to multiply matrices using OpenMP
void matrix_multiply_openmp(const std::vector<std::vector<int>>& A, const std::vector<std::vector<int>>& B, std::vector<std::vector<int>>& C) {
    int n = A.size();
    int m = B.size();
    int p = B[0].size();

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < p; ++j) {
            C[i][j] = 0;
            for (int k = 0; k < m; ++k) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

// Function to multiply matrices without OpenMP
void matrix_multiply(const std::vector<std::vector<int>>& A, const std::vector<std::vector<int>>& B, std::vector<std::vector<int>>& C) {
    int n = A.size();
    int m = B.size();
    int p = B[0].size();

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < p; ++j) {
            C[i][j] = 0;
            for (int k = 0; k < m; ++k) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

// Function to generate a random matrix of given size
std::vector<std::vector<int>> generate_random_matrix(int rows, int cols, int max_value = 10) {
    std::vector<std::vector<int>> matrix(rows, std::vector<int>(cols));
    std::random_device rd;
    std::default_random_engine generator(rd());
    std::uniform_int_distribution<int> distribution(0, max_value);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            matrix[i][j] = distribution(generator);
        }
    }
    return matrix;
}

// Function to print a matrix
void print_matrix(const std::vector<std::vector<int>>& matrix) {
    for (const auto& row : matrix) {
        for (int val : row) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }
}

int main() {
    int n = 100; // Number of rows of A and C
    int m = 100; // Number of columns of A and rows of B
    int p = 100; // Number of columns of B and C

    // Generate random matrices A and B
    std::vector<std::vector<int>> A = generate_random_matrix(n, m);
    std::vector<std::vector<int>> B = generate_random_matrix(m, p);

    // Initialize result matrices
    std::vector<std::vector<int>> C_openmp(n, std::vector<int>(p, 0));
    std::vector<std::vector<int>> C(n, std::vector<int>(p, 0));

    // Multiply matrices using OpenMP
    matrix_multiply_openmp(A, B, C_openmp);

    // Multiply matrices without OpenMP
    matrix_multiply(A, B, C);

    // Print the results
    std::cout << "Matrix A:" << std::endl;
    print_matrix(A);
    std::cout << std::endl;

    std::cout << "Matrix B:" << std::endl;
    print_matrix(B);
    std::cout << std::endl;

    std::cout << "Matrix C (OpenMP):" << std::endl;
    print_matrix(C_openmp);
    std::cout << std::endl;

    std::cout << "Matrix C (without OpenMP):" << std::endl;
    print_matrix(C);
    std::cout << std::endl;

    // Verify the results
    bool correct = true;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < p; ++j) {
            if (C_openmp[i][j] != C[i][j]) {
                correct = false;
                break;
            }
        }
    }

    if (correct) {
        std::cout << "Results are correct." << std::endl;
    } else {
        std::cout << "Results are incorrect." << std::endl;
    }

    return 0;
}
