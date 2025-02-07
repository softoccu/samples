#include <CL/cl.h>
#include <iostream>
#include <vector>
#include <limits>
#include <chrono>
#include <random>

const int INF = std::numeric_limits<int>::max();

// OpenCL kernel to perform Floyd-Warshall algorithm using Dijkstra's algorithm in parallel
const char *kernelSource = R"(
__kernel void dijkstra_kernel(__global int* graph, __global int* dist, __global int* visited, int n, int src) {
    int id = get_global_id(0);

    if (id == src) {
        for (int i = 0; i < n; i++) {
            dist[src * n + i] = graph[src * n + i];
            visited[src * n + i] = (i == src) ? 1 : 0;
        }

        for (int count = 0; count < n - 1; count++) {
            int min_dist = INT_MAX;
            int min_index = -1;
            for (int v = 0; v < n; v++) {
                if (!visited[src * n + v] && dist[src * n + v] <= min_dist) {
                    min_dist = dist[src * n + v];
                    min_index = v;
                }
            }

            visited[src * n + min_index] = 1;

            for (int v = 0; v < n; v++) {
                if (!visited[src * n + v] && graph[min_index * n + v] && dist[src * n + min_index] != INT_MAX &&
                    dist[src * n + min_index] + graph[min_index * n + v] < dist[src * n + v]) {
                    dist[src * n + v] = dist[src * n + min_index] + graph[min_index * n + v];
                }
            }
        }
    }
}
)";

// Helper function to perform Floyd-Warshall algorithm on host
void floyd_warshall_host(std::vector<int>& graph, int n) {
    for (int k = 0; k < n; ++k) {
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                if (graph[i * n + k] != INF && graph[k * n + j] != INF && graph[i * n + k] + graph[k * n + j] < graph[i * n + j]) {
                    graph[i * n + j] = graph[i * n + k] + graph[k * n + j];
                }
            }
        }
    }
}

int main() {
    const int n = 128;
    std::vector<int> graph(n * n, INF);
    std::vector<int> graph_opencl(n * n, INF);
    std::vector<int> dist(n * n, INF);
    std::vector<int> visited(n * n, 0);

    // Generate random graph
    std::random_device rd;
    std::default_random_engine generator(rd());
    std::uniform_int_distribution<int> distribution(1, 10);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i != j) {
                graph[i * n + j] = distribution(generator);
                graph_opencl[i * n + j] = graph[i * n + j];
            } else {
                graph[i * n + j] = 0;
                graph_opencl[i * n + j] = 0;
            }
        }
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
    cl_mem graph_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, n * n * sizeof(int), graph_opencl.data(), &ret);
    cl_mem dist_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE, n * n * sizeof(int), NULL, &ret);
    cl_mem visited_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE, n * n * sizeof(int), NULL, &ret);

    if (ret != CL_SUCCESS) {
        std::cerr << "Failed to create buffer objects." << std::endl;
        return -1;
    }

    // Initialize distance and visited arrays
    ret = clEnqueueFillBuffer(command_queue, dist_mem_obj, &INF, sizeof(int), 0, n * n * sizeof(int), 0, NULL, NULL);
    ret = clEnqueueFillBuffer(command_queue, visited_mem_obj, &INF, sizeof(int), 0, n * n * sizeof(int), 0, NULL, NULL);

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
    cl_kernel kernel = clCreateKernel(program, "dijkstra_kernel", &ret);

    if (ret != CL_SUCCESS) {
        std::cerr << "Failed to create kernel." << std::endl;
        return -1;
    }

    // Set kernel arguments
    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&graph_mem_obj);
    ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&dist_mem_obj);
    ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&visited_mem_obj);
    ret = clSetKernelArg(kernel, 3, sizeof(int), (void *)&n);

    if (ret != CL_SUCCESS) {
        std::cerr << "Failed to set kernel arguments." << std::endl;
        return -1;
    }

    // Execute the Floyd-Warshall on host
    auto start = std::chrono::high_resolution_clock::now();
    floyd_warshall_host(graph, n);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Host function time: " << duration.count() << " seconds" << std::endl;

    // Execute the Floyd-Warshall on device using Dijkstra's algorithm in parallel
    start = std::chrono::high_resolution_clock::now();

    for (int src = 0; src < n; src++) {
        ret = clSetKernelArg(kernel, 4, sizeof(int), (void *)&src);

        size_t global_work_size = n;
        size_t local_work_size = 64;
        size_t adjusted_global_work_size = ((global_work_size + local_work_size - 1) / local_work_size) * local_work_size;

        ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &adjusted_global_work_size, &local_work_size, 0, NULL, NULL);
        if (ret != CL_SUCCESS) {
            std::cerr << "Failed to enqueue NDRange kernel." << std::endl;
            return -1;
        }
    }

    // Read the memory buffer dist_mem_obj on the device to the local variable dist
    ret = clEnqueueReadBuffer(command_queue, dist_mem_obj, CL_TRUE, 0, n * n * sizeof(int), dist.data(), 0, NULL, NULL);
    if (ret != CL_SUCCESS) {
        std::cerr << "Failed to read dist buffer." << std::endl;
        return -1;
    }

    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    std::cout << "OpenCL function time: " << duration.count() << " seconds" << std::endl;

    // Verify results
    bool match = true;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (graph[i * n + j] != dist[i * n + j]) {
                match = false;
                std::cout << "Mismatch at (" << i << ", " << j << "): Host = " << graph[i * n + j] << ", OpenCL = " << dist[i * n + j] << std::endl;
            }
        }
    }

    if (match) {
        std::cout << "Results match!" << std::endl;
    } else {
        std::cout << "Results do not match!" << std::endl;
    }

    // Clean up OpenCL resources
    ret = clFlush(command_queue);
    ret = clFinish(command_queue);
    ret = clReleaseKernel(kernel);
    ret = clReleaseProgram(program);
    ret = clReleaseMemObject(graph_mem_obj);
    ret = clReleaseMemObject(dist_mem_obj);
    ret = clReleaseMemObject(visited_mem_obj);
    ret = clReleaseCommandQueue(command_queue);
    ret = clReleaseContext(context);

    return 0;
}