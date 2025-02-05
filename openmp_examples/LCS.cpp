#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <omp.h>
#include <random>

// Function to compute LCS using dynamic programming with OpenMP
// update DP matrix diagonally
int lcs_openmp(const std::string& X, const std::string& Y) {
    int m = X.length();
    int n = Y.length();
    std::vector<std::vector<int>> L(m + 1, std::vector<int>(n + 1, 0));
    int num_threads;

    for(int offset = 0; offset < (m + n); ++offset){
        #pragma omp parallel
        {
            //#pragma omp single
            //{
            //    num_threads = omp_get_num_threads();
            //    std::cout << "Number of threads used: " << num_threads << std::endl;
            //}

            #pragma omp for
            for(int idx = 0; idx < std::min(m,n); ++idx) {
                int i, j;
                if(offset < n){
                    i = idx + 1;
                    j = offset - idx + 1;
                } else {
                    i = offset - (n - 1) + idx + 1;
                    j = (n - 1) - idx + 1;
                }
                if (i >= 1 && i <= m && j >= 1 && j <= n) {
                    if (X[i - 1] == Y[j - 1])
                        L[i][j] = L[i - 1][j - 1] + 1;
                    else
                        L[i][j] = std::max(L[i - 1][j], L[i][j - 1]);
                }
            }
        }
    }

    return L[m][n];
}

// Function to compute LCS using dynamic programming without OpenMP
int lcs(const std::string& X, const std::string& Y) {
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

    return L[m][n];
}

// Function to generate a random string of a given length
std::string generate_random_string(int length) {
    std::string charset = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    std::random_device rd;
    std::default_random_engine generator(rd());
    std::uniform_int_distribution<int> distribution(0, charset.size() - 1);
    std::string random_string;
    for (int i = 0; i < length; ++i) {
        random_string += charset[distribution(generator)];
    }
    return random_string;
}

int main() {
    // Generate random strings X and Y of length 512
    std::string X = generate_random_string(512);
    std::string Y = generate_random_string(512);

    // Print the generated strings
    std::cout << "String X: " << X << std::endl;
    std::cout << "String Y: " << Y << std::endl;

    // Compute LCS using OpenMP
    int lcs_length_openmp = lcs_openmp(X, Y);
    std::cout << "LCS length (OpenMP): " << lcs_length_openmp << std::endl;
    
    // Compute LCS without OpenMP
    int lcs_length = lcs(X, Y);
    std::cout << "LCS length (without OpenMP): " << lcs_length << std::endl;

    // Verify the results
    if (lcs_length_openmp == lcs_length) {
        std::cout << "Results are correct." << std::endl;
    } else {
        std::cout << "Results are incorrect." << std::endl;
    }

    return 0;
}