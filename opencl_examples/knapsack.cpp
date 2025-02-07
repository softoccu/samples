#include <CL/cl.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <iomanip>

// OpenCL kernel source code

const char* kernelSource = R"(
__kernel void knapsack(__global int* weights, __global int* values, __global int* dp_old, __global int* dp_new, int W, int n) {
    int w = get_global_id(0); // Weight index
    if (w > W) return;

    for (int i = 1; i <= n; ++i) {
        if (weights[i - 1] <= w) {
            dp_new[w] = max(dp_old[w], values[i - 1] + dp_old[w - weights[i - 1]]);
        } else {
            dp_new[w] = dp_old[w];
        }
        barrier(CLK_GLOBAL_MEM_FENCE);
        dp_old[w] = dp_new[w];
        barrier(CLK_GLOBAL_MEM_FENCE);
    }
}
)";

void knapsack_host(const std::vector<int>& weights, const std::vector<int>& values, std::vector<int>& dp, int W, int n) {
    for (int i = 1; i <= n; ++i) {
        for (int w = W; w >= 0; --w) {
            if (weights[i - 1] <= w) {
                dp[w] = std::max(dp[w], dp[w - weights[i - 1]] + values[i - 1]);
            }
        }
    }
}

void print_dp(const std::vector<int>& dp, int W, const std::vector<int>& dp_host, bool match) {
    for (int w = 0; w <= W; ++w) {
        int value = dp[w];
        if (!match && value != dp_host[w]) {
            // Print mismatched values in red
            std::cout << "\033[31m" << std::setw(5) << value << "\033[0m ";
        } else {
            std::cout << std::setw(5) << value << " ";
        }
    }
    std::cout << "\n";
}

int main() {
    // Knapsack problem parameters
    int W = 50; // Maximum weight
    int n = 1000; // Number of items

    // Generate random weights and values
    std::vector<int> weights(n);
    std::vector<int> values(n);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> weight_dist(1, 10);
    std::uniform_int_distribution<> value_dist(1, 100);

    for (int i = 0; i < n; ++i) {
        weights[i] = weight_dist(gen);
        values[i] = value_dist(gen);
    }

    // Print generated weights and values
    std::cout << "Weights: ";
    for (const auto& w : weights) {
        std::cout << w << " ";
    }
    std::cout << "\nValues: ";
    for (const auto& v : values) {
        std::cout << v << " ";
    }
    std::cout << "\n";

    // Host DP array
    std::vector<int> dp_host(W + 1, 0);

    // Measure host execution time
    auto start = std::chrono::high_resolution_clock::now();
    knapsack_host(weights, values, dp_host, W, n);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Host execution time: " << duration.count() << " seconds\n";

    // OpenCL setup
    cl_platform_id platform_id = nullptr;
    cl_device_id device_id = nullptr;
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;
    cl_int ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, &ret_num_devices);
    cl_context context = clCreateContext(nullptr, 1, &device_id, nullptr, nullptr, &ret);
    // Create command queue with properties
    cl_queue_properties properties[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
    cl_command_queue command_queue = clCreateCommandQueueWithProperties(context, device_id, properties, &ret);

    cl_program program = clCreateProgramWithSource(context, 1, &kernelSource, nullptr, &ret);
    ret = clBuildProgram(program, 1, &device_id, nullptr, nullptr, nullptr);

    cl_kernel kernel = clCreateKernel(program, "knapsack", &ret);

    // Device buffers
    cl_mem bufferWeights = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * weights.size(), weights.data(), &ret);
    cl_mem bufferValues = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * values.size(), values.data(), &ret);
    cl_mem bufferDpOld = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int) * (W + 1), nullptr, &ret);
    cl_mem bufferDpNew = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int) * (W + 1), nullptr, &ret);

    // Initialize device DP arrays to zero
    std::vector<int> dp_device(W + 1, 0);
    ret = clEnqueueWriteBuffer(command_queue, bufferDpOld, CL_TRUE, 0, sizeof(int) * (W + 1), dp_device.data(), 0, nullptr, nullptr);
    ret = clEnqueueWriteBuffer(command_queue, bufferDpNew, CL_TRUE, 0, sizeof(int) * (W + 1), dp_device.data(), 0, nullptr, nullptr);

    // Set kernel arguments
    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufferWeights);
    ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufferValues);
    ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufferDpOld);
    ret = clSetKernelArg(kernel, 3, sizeof(cl_mem), &bufferDpNew);
    ret = clSetKernelArg(kernel, 4, sizeof(int), &W);
    ret = clSetKernelArg(kernel, 5, sizeof(int), &n);

    // Measure device execution time
    start = std::chrono::high_resolution_clock::now();
    size_t global_item_size = W + 1; // One work-item per weight
    ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, nullptr, &global_item_size, nullptr, 0, nullptr, nullptr);
    ret = clFinish(command_queue);
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    std::cout << "Device execution time: " << duration.count() << " seconds\n";

    // Read back result
    ret = clEnqueueReadBuffer(command_queue, bufferDpNew, CL_TRUE, 0, sizeof(int) * (W + 1), dp_device.data(), 0, nullptr, nullptr);

    // Verify results
    bool match = true;
    for (int w = 0; w <= W; ++w) {
        if (dp_host[w] != dp_device[w]) {
            match = false;
            break;
        }
    }

    if (match) {
        std::cout << "Results match! " <<  " Same result is " << dp_host[W] << std::endl;
    } else {
        std::cout << "Results do not match!\n";
        std::cout << "Host DP array:\n";
        print_dp(dp_host, W, dp_host, match);
        std::cout << "Device DP array:\n";
        print_dp(dp_device, W, dp_host, match);
    }

    // Clean up
    ret = clReleaseKernel(kernel);
    ret = clReleaseProgram(program);
    ret = clReleaseMemObject(bufferWeights);
    ret = clReleaseMemObject(bufferValues);
    ret = clReleaseMemObject(bufferDpOld);
    ret = clReleaseMemObject(bufferDpNew);
    ret = clReleaseCommandQueue(command_queue);
    ret = clReleaseContext(context);

    return 0;
}
