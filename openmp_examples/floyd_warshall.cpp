#include <iostream>
#include <vector>
#include <queue>
#include <omp.h>
#include <limits>
#include <random>

// Function to perform Dijkstra's algorithm for a single source
void dijkstra(const std::vector<std::vector<int>>& graph, int src, std::vector<int>& dist) {
    int V = graph.size();
    dist.assign(V, std::numeric_limits<int>::max());
    dist[src] = 0;

    std::priority_queue<std::pair<int, int>, std::vector<std::pair<int, int>>, std::greater<>> pq;
    pq.push({0, src});

    while (!pq.empty()) {
        int u = pq.top().second;
        pq.pop();

        for (int v = 0; v < V; ++v) {
            if (graph[u][v] != std::numeric_limits<int>::max() && dist[u] != std::numeric_limits<int>::max() && dist[u] + graph[u][v] < dist[v]) {
                dist[v] = dist[u] + graph[u][v];
                pq.push({dist[v], v});
            }
        }
    }
}

// Function to compute shortest paths using multiple Dijkstra's algorithms with OpenMP
void floyd_warshall_openmp_dijkstra(const std::vector<std::vector<int>>& graph, std::vector<std::vector<int>>& dist) {
    int V = graph.size();
    dist.resize(V, std::vector<int>(V, std::numeric_limits<int>::max()));

    #pragma omp parallel for
    for (int i = 0; i < V; ++i) {
        dijkstra(graph, i, dist[i]);
    }
}

// Function to compute shortest paths using Floyd-Warshall algorithm without OpenMP
void floyd_warshall(std::vector<std::vector<int>>& dist) {
    int V = dist.size();

    for (int k = 0; k < V; ++k) {
        for (int i = 0; i < V; ++i) {
            for (int j = 0; j < V; ++j) {
                if (dist[i][k] != std::numeric_limits<int>::max() && dist[k][j] != std::numeric_limits<int>::max()) {
                    dist[i][j] = std::min(dist[i][j], dist[i][k] + dist[k][j]);
                }
            }
        }
    }
}

// Function to print the distance matrix
void print_matrix(const std::vector<std::vector<int>>& matrix) {
    int V = matrix.size();
    for (int i = 0; i < V; ++i) {
        for (int j = 0; j < V; ++j) {
            if (matrix[i][j] == std::numeric_limits<int>::max())
                std::cout << "INF ";
            else
                std::cout << matrix[i][j] << " ";
        }
        std::cout << std::endl;
    }
}

// Function to generate a random graph as an adjacency matrix
std::vector<std::vector<int>> generate_random_graph(int V, int max_weight = 10) {
    std::vector<std::vector<int>> graph(V, std::vector<int>(V, std::numeric_limits<int>::max()));
    std::random_device rd;
    std::default_random_engine generator(rd());
    std::uniform_int_distribution<int> distribution(1, max_weight);

    for (int i = 0; i < V; ++i) {
        for (int j = 0; j < V; ++j) {
            if (i != j) {
                graph[i][j] = distribution(generator);
            } else {
                graph[i][j] = 0;
            }
        }
    }
    return graph;
}

int main() {
    // Number of vertices in the graph
    int V = 128;

    // Generate a random graph
    std::vector<std::vector<int>> graph = generate_random_graph(V);

    // Create a copy of the graph for OpenMP and non-OpenMP versions
    std::vector<std::vector<int>> dist_openmp;
    std::vector<std::vector<int>> dist = graph;

    // Compute shortest paths using multiple Dijkstra's algorithms with OpenMP
    floyd_warshall_openmp_dijkstra(graph, dist_openmp);
    // Compute shortest paths without OpenMP using Floyd-Warshall algorithm
    floyd_warshall(dist);

    // Print the results
    std::cout << "Shortest path matrix with OpenMP (Dijkstra):" << std::endl;
    print_matrix(dist_openmp);
    std::cout << std::endl;

    std::cout << "Shortest path matrix without OpenMP (Floyd-Warshall):" << std::endl;
    print_matrix(dist);
    std::cout << std::endl;

    // Verify the results
    bool correct = true;
    for (int i = 0; i < V; ++i) {
        for (int j = 0; j < V; ++j) {
            if (dist_openmp[i][j] != dist[i][j]) {
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