These CUDA examples are showing how to solve problems using CUDA, take advantages from GPU hardware. Not only multiply matrixs which are heavily used in AI platforms, but also we can make other algorithm go to parallel so as to make it faster.

Test environment:

Ubuntu 24 with apt updated
CUDA 12 library/runtime/driver are correctly installed
RTX 4090




1. CUDA BFS 
2. CUDA FFT
3. CUDA LCS, things to care, DP matrix need update diagonally.
4. FLOYD WARSHALL, things to care, a for(for(for)) structure can't be simply go parallel as current computer depend on previous result. So, a practical way is to run a multi-dijstra in parallel way, which aren't depend on eath other.
5. Integral, cut to small pieces, add up.
6. KNAPSACK problem
7. Monte Carlo price estimation is frequently used in financial industry.
8. Multiply Mitrixs
9. Find primes using CUDA
10. TSP problem, (ACO) Ant Colony Optimization is used