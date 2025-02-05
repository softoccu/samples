#include <iostream>
#include <vector>
#include <queue>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>

const int INF = 1e9;
const int NUM_NODES = 100;
const int MAX_EDGES_PER_NODE = 10;

__global__ void bfs_kernel(int* d_adjList, int* d_adjListIndices, int* d_dist, int* d_frontier, int frontierSize, int numNodes) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < frontierSize) {
        int node = d_frontier[tid];
        int start = d_adjListIndices[node];
        int end = d_adjListIndices[node + 1];
        for (int i = start; i < end; i++) {
            int neighbor = d_adjList[i];
            if (d_dist[neighbor] == INF) {
                d_dist[neighbor] = d_dist[node] + 1;
            }
        }
    }
}

void bfs_gpu(const std::vector<std::vector<int>>& adjList, int startNode, std::vector<int>& dist) {
    int numNodes = adjList.size();
    int numEdges = 0;
    for (const auto& neighbors : adjList) {
        numEdges += neighbors.size();
    }

    std::vector<int> adjListFlat;
    std::vector<int> adjListIndices(numNodes + 1, 0);
    for (int i = 0; i < numNodes; i++) {
        adjListIndices[i + 1] = adjListIndices[i] + adjList[i].size();
        adjListFlat.insert(adjListFlat.end(), adjList[i].begin(), adjList[i].end());
    }

    int* d_adjList;
    int* d_adjListIndices;
    int* d_dist;
    int* d_frontier;

    cudaMalloc((void**)&d_adjList, numEdges * sizeof(int));
    cudaMalloc((void**)&d_adjListIndices, (numNodes + 1) * sizeof(int));
    cudaMalloc((void**)&d_dist, numNodes * sizeof(int));
    cudaMalloc((void**)&d_frontier, numNodes * sizeof(int));

    cudaMemcpy(d_adjList, adjListFlat.data(), numEdges * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_adjListIndices, adjListIndices.data(), (numNodes + 1) * sizeof(int), cudaMemcpyHostToDevice);

    dist.assign(numNodes, INF);
    dist[startNode] = 0;
    cudaMemcpy(d_dist, dist.data(), numNodes * sizeof(int), cudaMemcpyHostToDevice);

    std::queue<int> frontier;
    frontier.push(startNode);
    while (!frontier.empty()) {
        int frontierSize = frontier.size();
        std::vector<int> h_frontier(frontierSize);
        for (int i = 0; i < frontierSize; i++) {
            h_frontier[i] = frontier.front();
            frontier.pop();
        }
        cudaMemcpy(d_frontier, h_frontier.data(), frontierSize * sizeof(int), cudaMemcpyHostToDevice);

        int blockSize = 256;
        int numBlocks = (frontierSize + blockSize - 1) / blockSize;
        bfs_kernel<<<numBlocks, blockSize>>>(d_adjList, d_adjListIndices, d_dist, d_frontier, frontierSize, numNodes);
        cudaDeviceSynchronize();

        cudaMemcpy(dist.data(), d_dist, numNodes * sizeof(int), cudaMemcpyDeviceToHost);
        for (int i = 0; i < frontierSize; i++) {
            int node = h_frontier[i];
            int start = adjListIndices[node];
            int end = adjListIndices[node + 1];
            for (int j = start; j < end; j++) {
                int neighbor = adjListFlat[j];
                if (dist[neighbor] == dist[node] + 1) {
                    frontier.push(neighbor);
                }
            }
        }
    }

    cudaFree(d_adjList);
    cudaFree(d_adjListIndices);
    cudaFree(d_dist);
    cudaFree(d_frontier);
}

int main() {
    srand(time(0));
    std::vector<std::vector<int>> adjList(NUM_NODES);

    // Generate a random graph
    for (int i = 0; i < NUM_NODES; i++) {
        int numEdges = rand() % MAX_EDGES_PER_NODE;
        for (int j = 0; j < numEdges; j++) {
            int neighbor = rand() % NUM_NODES;
            if (neighbor != i) {
                adjList[i].push_back(neighbor);
            }
        }
    }

    int startNode = 0;
    std::vector<int> dist;
    bfs_gpu(adjList, startNode, dist);

    for (int i = 0; i < dist.size(); i++) {
        std::cout << "Node " << i << " is at distance " << dist[i] << " from the start node." << std::endl;
    }

    return 0;
}