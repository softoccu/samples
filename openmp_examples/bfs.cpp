#include <iostream>
#include <vector>
#include <queue>
#include <omp.h>
#include <random>

// Function to perform BFS using OpenMP
void bfs_openmp(const std::vector<std::vector<int>>& graph, int start) {
    int n = graph.size();
    std::vector<bool> visited(n, false);
    std::queue<int> q;
    q.push(start);
    visited[start] = true;

    #pragma omp parallel
    {
        std::queue<int> local_q;
        #pragma omp single
        {
            local_q.swap(q);
        }

        while (!local_q.empty()) {
            int node = local_q.front();
            local_q.pop();
            std::cout << node << " ";

            #pragma omp parallel for
            for (int i = 0; i < graph[node].size(); ++i) {
                int neighbor = graph[node][i];
                if (!visited[neighbor]) {
                    visited[neighbor] = true;
                    #pragma omp critical
                    local_q.push(neighbor);
                }
            }
        }
    }
}

// Function to perform BFS without OpenMP
void bfs(const std::vector<std::vector<int>>& graph, int start) {
    int n = graph.size();
    std::vector<bool> visited(n, false);
    std::queue<int> q;

    q.push(start);
    visited[start] = true;

    while (!q.empty()) {
        int node = q.front();
        q.pop();
        std::cout << node << " ";

        for (int i = 0; i < graph[node].size(); ++i) {
            int neighbor = graph[node][i];
            if (!visited[neighbor]) {
                visited[neighbor] = true;
                q.push(neighbor);
            }
        }
    }
}

int main() {
    const int num_nodes = 100;
    std::vector<std::vector<int>> graph(num_nodes);

    // Randomly generate a graph with 100 nodes
    std::default_random_engine generator;
    std::uniform_int_distribution<int> distribution(0, num_nodes - 1);

    for (int i = 0; i < num_nodes; ++i) {
        int num_edges = distribution(generator) % 10 + 1; // Each node has between 1 and 10 edges
        for (int j = 0; j < num_edges; ++j) {
            int neighbor = distribution(generator);
            if (neighbor != i) {
                graph[i].push_back(neighbor);
            }
        }
    }

    std::cout << "BFS with OpenMP:\n";
    bfs_openmp(graph, 0);

    std::cout << "\n\nBFS without OpenMP:\n";
    bfs(graph, 0);

    return 0;
}
