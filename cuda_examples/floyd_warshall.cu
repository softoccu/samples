#include <iostream>
#include <vector>
#include <algorithm>
#include <queue>
#include <limits>
#include <cuda_runtime.h>

#define INF 1e9

// Parallel Floyd–Warshall algorithm can be complex thing see the link  https://en.wikipedia.org/wiki/Parallel_all-pairs_shortest_path_algorithm
// however a easy way to get same result, may be faster via run multi-dijstra for each V, use you cuda codes, run parallel
// finally checked the result by floydWarshallCPU


__global__ void dijkstraKernel(int* d_graph, int* d_dist, int V) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= V) return;

    bool* visited = new bool[V];
    for (int i = 0; i < V; ++i) {
        visited[i] = false;
        d_dist[tid * V + i] = INF;
    }
    d_dist[tid * V + tid] = 0;

    for (int i = 0; i < V - 1; ++i) {
        int u = -1;
        for (int j = 0; j < V; ++j) {
            if (!visited[j] && (u == -1 || d_dist[tid * V + j] < d_dist[tid * V + u])) {
                u = j;
            }
        }

        visited[u] = true;

        for (int v = 0; v < V; ++v) {
            if (!visited[v] && d_graph[u * V + v] && d_dist[tid * V + u] + d_graph[u * V + v] < d_dist[tid * V + v]) {
                d_dist[tid * V + v] = d_dist[tid * V + u] + d_graph[u * V + v];
            }
        }
    }

    delete[] visited;
}

void dijkstraHost(const std::vector<std::vector<int>>& graph, std::vector<std::vector<int>>& dist) {
    int V = graph.size();
    int* h_graph = new int[V * V];
    int* h_dist = new int[V * V];

    // Flatten the graph matrix
    for (int i = 0; i < V; ++i) {
        for (int j = 0; j < V; ++j) {
            h_graph[i * V + j] = graph[i][j];
        }
    }

    int* d_graph;
    int* d_dist;
    cudaMalloc(&d_graph, V * V * sizeof(int));
    cudaMalloc(&d_dist, V * V * sizeof(int));

    cudaMemcpy(d_graph, h_graph, V * V * sizeof(int), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (V + blockSize - 1) / blockSize;

    for (int src = 0; src < V; ++src) {
        dijkstraKernel<<<numBlocks, blockSize>>>(d_graph, d_dist, V);
        cudaDeviceSynchronize();
    }

    cudaMemcpy(h_dist, d_dist, V * V * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < V; ++i) {
        for (int j = 0; j < V; ++j) {
            dist[i][j] = h_dist[i * V + j];
        }
    }

    delete[] h_graph;
    delete[] h_dist;
    cudaFree(d_graph);
    cudaFree(d_dist);
}

void floydWarshallCPU(std::vector<std::vector<int>>& dist) {
    int n = dist.size();
    for (int k = 0; k < n; ++k) {
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                if (dist[i][k] < INF && dist[k][j] < INF && dist[i][k] + dist[k][j] < dist[i][j]) {
                    dist[i][j] = dist[i][k] + dist[k][j];
                }
            }
        }
    }
}

void verifyResults(const std::vector<std::vector<int>>& cudaDist, const std::vector<std::vector<int>>& floydDist) {
    int V = cudaDist.size();
    for (int i = 0; i < V; ++i) {
        for (int j = 0; j < V; ++j) {
            if (cudaDist[i][j] != floydDist[i][j]) {
                std::cout << "Mismatch at (" << i << ", " << j << "): CUDA = " << cudaDist[i][j] << ", Floyd-Warshall = " << floydDist[i][j] << std::endl;
                return;
            }
        }
    }
    std::cout << "All results are correct!" << std::endl;
}

int main() {
    int V = 128;  // Number of vertices

    // Initialize graph with INF for non-edges and random weights for edges
    std::vector<std::vector<int>> graph(V, std::vector<int>(V, INF));
    for (int i = 0; i < V; ++i) {
        graph[i][i] = 0;
    }

    srand(time(0));
    for (int i = 0; i < V; ++i) {
        for (int j = i + 1; j < V; ++j) {
            if (rand() % 2) {
                int weight = rand() % 10 + 1;
                graph[i][j] = weight;
                graph[j][i] = weight;
            }
        }
    }

    // Print original graph
    std::cout << "Original graph:" << std::endl;
    for (const auto& row : graph) {
        for (int val : row) {
            if (val == INF) {
                std::cout << "INF ";
            } else {
                std::cout << val << " ";
            }
        }
        std::cout << std::endl;
    }

    // Perform Dijkstra on each vertex using CUDA
    std::vector<std::vector<int>> cudaDist(V, std::vector<int>(V, INF));
    dijkstraHost(graph, cudaDist);

    // Print the CUDA distance matrix
    std::cout << "CUDA Distance matrix:" << std::endl;
    for (const auto& row : cudaDist) {
        for (int val : row) {
            if (val == INF) {
                std::cout << "INF ";
            } else {
                std::cout << val << " ";
            }
        }
        std::cout << std::endl;
    }

    // Perform Floyd-Warshall on the CPU
    std::vector<std::vector<int>> floydDist = graph;
    floydWarshallCPU(floydDist);

    // Print the Floyd-Warshall distance matrix
    std::cout << "Floyd-Warshall Distance matrix:" << std::endl;
    for (const auto& row : floydDist) {
        for (int val : row) {
            if (val == INF) {
                std::cout << "INF ";
            } else {
                std::cout << val << " ";
            }
        }
        std::cout << std::endl;
    }

    // Verify results
    verifyResults(cudaDist, floydDist);

    return 0;
}