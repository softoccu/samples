#include <CL/cl.h>
#include <iostream>
#include <vector>
#include <queue>
#include <chrono>
#include <random>
#include <iomanip>

// OpenCL kernel to perform BFS
const char *kernelSource = R"(
__kernel void bfs_kernel(__global int* graph, __global int* frontier, __global int* next_frontier, __global int* distances, int num_nodes, int level) {
    int id = get_global_id(0);
    if (frontier[id] == 1) {
        frontier[id] = 0;
        for (int i = 0; i < num_nodes; i++) {
            if (graph[id * num_nodes + i] == 1 && distances[i] == -1) {
                distances[i] = level;
                next_frontier[i] = 1;
            }
        }
    }
}
)";

// Host function to perform BFS
void bfs_host(const std::vector<int>& graph, std::vector<int>& distances, int start_node, int num_nodes) {
    std::queue<int> q;
    q.push(start_node);
    distances[start_node] = 0;

    while (!q.empty()) {
        int node = q.front();
        q.pop();
        int level = distances[node] + 1;
        for (int i = 0; i < num_nodes; i++) {
            if (graph[node * num_nodes + i] == 1 && distances[i] == -1) {
                distances[i] = level;
                q.push(i);
            }
        }
    }
}

int main() {
    const int num_nodes = 1000;
    std::vector<int> graph(num_nodes * num_nodes, 0);
    std::vector<int> distances_host(num_nodes, -1);
    std::vector<int> distances_opencl(num_nodes, -1);
    std::vector<int> frontier(num_nodes, 0);
    std::vector<int> next_frontier(num_nodes, 0);

    // Generate a random sparse graph using std::random_device
    std::random_device rd;
    std::default_random_engine generator(rd());
    std::uniform_real_distribution<float> distribution(0.0, 1.0);
    float edge_probability = 0.05; // Low probability for a sparse graph
    for (int i = 0; i < num_nodes; i++) {
        for (int j = 0; j < num_nodes; j++) {
            if (i != j && distribution(generator) < edge_probability) {
                graph[i * num_nodes + j] = 1;
            }
        }
    }

    int start_node = 0;
    int end_node = num_nodes - 1;

    // Print the adjacency matrix of the graph
    std::cout << "Adjacency Matrix:" << std::endl;
    for (int i = 0; i < num_nodes; i++) {
        for (int j = 0; j < num_nodes; j++) {
            std::cout << std::setw(2) << graph[i * num_nodes + j] << " ";
        }
        std::cout << std::endl;
    }

    // Get platform and device information
    cl_platform_id platform_id = NULL;
    cl_device_id device_id = NULL;
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;
    cl_int ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, &ret_num_devices);

    if (ret != CL_SUCCESS) {
        std::cerr << "Failed to get platform or device ID." << std::endl;
        return -1;
    }

    // Create an OpenCL context
    cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);

    if (ret != CL_SUCCESS) {
        std::cerr << "Failed to create OpenCL context." << std::endl;
        return -1;
    }

    // Create a command queue with properties
    cl_command_queue_properties properties[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
    cl_command_queue command_queue = clCreateCommandQueueWithProperties(context, device_id, properties, &ret);

    if (ret != CL_SUCCESS) {
        std::cerr << "Failed to create command queue." << std::endl;
        return -1;
    }

    // Create memory buffers on the device
    cl_mem graph_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, num_nodes * num_nodes * sizeof(int), NULL, &ret);
    cl_mem frontier_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE, num_nodes * sizeof(int), NULL, &ret);
    cl_mem next_frontier_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE, num_nodes * sizeof(int), NULL, &ret);
    cl_mem distances_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE, num_nodes * sizeof(int), NULL, &ret);

    if (ret != CL_SUCCESS) {
        std::cerr << "Failed to create buffer objects." << std::endl;
        return -1;
    }

    // Copy the graph to the memory buffer
    ret = clEnqueueWriteBuffer(command_queue, graph_mem_obj, CL_TRUE, 0, num_nodes * num_nodes * sizeof(int), graph.data(), 0, NULL, NULL);
    if (ret != CL_SUCCESS) {
        std::cerr << "Failed to write graph buffer." << std::endl;
        return -1;
    }

    // Create a program from the kernel source
    cl_program program = clCreateProgramWithSource(context, 1, &kernelSource, NULL, &ret);

    if (ret != CL_SUCCESS) {
        std::cerr << "Failed to create program." << std::endl;
        return -1;
    }

    // Build the program
    ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);

    if (ret != CL_SUCCESS) {
        std::cerr << "Failed to build program." << std::endl;

        // Print build log in case of failure
        size_t log_size;
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        std::vector<char> log(log_size);
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, log_size, log.data(), NULL);
        std::cerr << log.data() << std::endl;

        return -1;
    }

    // Create the OpenCL kernel
    cl_kernel kernel = clCreateKernel(program, "bfs_kernel", &ret);

    if (ret != CL_SUCCESS) {
        std::cerr << "Failed to create kernel." << std::endl;
        return -1;
    }

    // Initialize distances and frontier
    distances_opencl[start_node] = 0;
    frontier[start_node] = 1;
    ret = clEnqueueWriteBuffer(command_queue, frontier_mem_obj, CL_TRUE, 0, num_nodes * sizeof(int), frontier.data(), 0, NULL, NULL);
    ret = clEnqueueWriteBuffer(command_queue, distances_mem_obj, CL_TRUE, 0, num_nodes * sizeof(int), distances_opencl.data(), 0, NULL, NULL);

    // Set kernel arguments
    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&graph_mem_obj);
    ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&frontier_mem_obj);
    ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&next_frontier_mem_obj);
    ret = clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&distances_mem_obj);
    ret = clSetKernelArg(kernel, 4, sizeof(int), (void *)&num_nodes);

    if (ret != CL_SUCCESS) {
        std::cerr << "Failed to set kernel arguments." << std::endl;
        return -1;
    }

    size_t global_item_size = num_nodes;
    size_t local_item_size = 64;

    // Execute the BFS on host
    auto start = std::chrono::high_resolution_clock::now();
    bfs_host(graph, distances_host, start_node, num_nodes);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Host function time: " << duration.count() << " seconds" << std::endl;

    // Execute the BFS on device
    start = std::chrono::high_resolution_clock::now();
    int level = 1;
    bool done = false;
    while (!done) {
        // Set level argument
        ret = clSetKernelArg(kernel, 5, sizeof(int), (void *)&level);

        // Adjust global_item_size to be a multiple of local_item_size
        size_t adjusted_global_item_size = ((global_item_size + local_item_size - 1) / local_item_size) * local_item_size;

        // Execute the kernel
        ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &adjusted_global_item_size, &local_item_size, 0, NULL, NULL);
        if (ret != CL_SUCCESS) {
            std::cerr << "Failed to enqueue NDRange kernel." << std::endl;
            return -1;
        }

        // Read next frontier
        ret = clEnqueueReadBuffer(command_queue, next_frontier_mem_obj, CL_TRUE, 0, num_nodes * sizeof(int), next_frontier.data(), 0, NULL, NULL);
        if (ret != CL_SUCCESS) {
            std::cerr << "Failed to read next frontier buffer." << std::endl;
            return -1;
        }

        // Check if done
        done = true;
        for (int i = 0; i < num_nodes; i++) {
            if (next_frontier[i] == 1) {
                done = false;
                break;
            }
        }

        // Swap frontiers
        std::copy(next_frontier.begin(), next_frontier.end(), frontier.begin());
        std::fill(next_frontier.begin(), next_frontier.end(), 0);

        // Write frontiers to device
        ret = clEnqueueWriteBuffer(command_queue, frontier_mem_obj, CL_TRUE, 0, num_nodes * sizeof(int), frontier.data(), 0, NULL, NULL);
        ret = clEnqueueWriteBuffer(command_queue, next_frontier_mem_obj, CL_TRUE, 0, num_nodes * sizeof(int), next_frontier.data(), 0, NULL, NULL);

        level++;
    }

    // Read the memory buffer distances_mem_obj on the device to the local variable distances_opencl
    ret = clEnqueueReadBuffer(command_queue, distances_mem_obj, CL_TRUE, 0, num_nodes * sizeof(int), distances_opencl.data(), 0, NULL, NULL);

    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    std::cout << "OpenCL function time: " << duration.count() << " seconds" << std::endl;

    // Verify results
    if (distances_opencl[end_node] == distances_host[end_node]) {
        std::cout << "Results match! Number of steps: " << distances_opencl[end_node] << std::endl;
    } else {
        std::cout << "Results do not match! OpenCL steps: " << distances_opencl[end_node] << ", Host steps: " << distances_host[end_node] << std::endl;
    }

    // Clean up OpenCL resources
    ret = clFlush(command_queue);
    ret = clFinish(command_queue);
    ret = clReleaseKernel(kernel);
    ret = clReleaseProgram(program);
    ret = clReleaseMemObject(graph_mem_obj);
    ret = clReleaseMemObject(frontier_mem_obj);
    ret = clReleaseMemObject(next_frontier_mem_obj);
    ret = clReleaseMemObject(distances_mem_obj);
    ret = clReleaseCommandQueue(command_queue);
    ret = clReleaseContext(context);

    return 0;
}