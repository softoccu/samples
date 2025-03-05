#include <CL/cl.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>

// OpenCL kernel to perform numerical integration using the trapezoidal rule
const char *kernelSource = R"(
__kernel void integral_kernel(__global float* results, float a, float b, int n) {
    int id = get_global_id(0);
    float h = (b - a) / n;
    float x = a + id * h;
    float f_x = x * x; // Example function: f(x) = x^2
    results[id] = f_x;
}
)";

// Helper function to perform numerical integration on host using the trapezoidal rule
float integral_host(float a, float b, int n) {
    float h = (b - a) / n;
    float sum = 0.5f * (a * a + b * b); // Example function: f(x) = x^2
    for (int i = 1; i < n; ++i) {
        float x = a + i * h;
        sum += x * x;
    }
    return sum * h;
}

int main() {
    float a = 0.0f;
    float b = 1.0f;
    int n = 1000000;
    std::vector<float> results(n, 0.0f);

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
    cl_mem results_mem_obj = clCreateBuffer(context, CL_MEM_WRITE_ONLY, n * sizeof(float), NULL, &ret);

    if (ret != CL_SUCCESS) {
        std::cerr << "Failed to create buffer objects." << std::endl;
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
    cl_kernel kernel = clCreateKernel(program, "integral_kernel", &ret);

    if (ret != CL_SUCCESS) {
        std::cerr << "Failed to create kernel." << std::endl;
        return -1;
    }

    // Set kernel arguments
    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&results_mem_obj);
    ret = clSetKernelArg(kernel, 1, sizeof(float), (void *)&a);
    ret = clSetKernelArg(kernel, 2, sizeof(float), (void *)&b);
    ret = clSetKernelArg(kernel, 3, sizeof(int), (void *)&n);

    if (ret != CL_SUCCESS) {
        std::cerr << "Failed to set kernel arguments." << std::endl;
        return -1;
    }

    size_t global_item_size = n;
    size_t local_item_size = 64;
    size_t adjusted_global_item_size = ((global_item_size + local_item_size - 1) / local_item_size) * local_item_size;

    // Execute the integral calculation on host
    auto start = std::chrono::high_resolution_clock::now();
    float result_host = integral_host(a, b, n);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Host function time: " << duration.count() << " seconds" << std::endl;
    std::cout << "Host integral result: " << result_host << std::endl;

    // Execute the integral calculation on device
    start = std::chrono::high_resolution_clock::now();
    ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &adjusted_global_item_size, &local_item_size, 0, NULL, NULL);

    if (ret != CL_SUCCESS) {
        std::cerr << "Failed to enqueue NDRange kernel." << std::endl;
        return -1;
    }

    ret = clFinish(command_queue);
    if (ret != CL_SUCCESS) {
        std::cerr << "Failed to finish command queue." << std::endl;
        return -1;
    }

    // Read the memory buffer results_mem_obj on the device to the local variable results
    ret = clEnqueueReadBuffer(command_queue, results_mem_obj, CL_TRUE, 0, n * sizeof(float), results.data(), 0, NULL, NULL);
    if (ret != CL_SUCCESS) {
        std::cerr << "Failed to read results buffer." << std::endl;
        return -1;
    }

    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    std::cout << "OpenCL function time: " << duration.count() << " seconds" << std::endl;

    // Compute the final integral result on the host
    float result_opencl = (results[0] + results[n - 1]) * 0.5f;
    for (int i = 1; i < n - 1; ++i) {
        result_opencl += results[i];
    }
    result_opencl *= (b - a) / n;

    std::cout << "OpenCL integral result: " << result_opencl << std::endl;

    // Verify results
    if (std::fabs(result_host - result_opencl) < 1e-5) {
        std::cout << "Results match!" << std::endl;
    } else {
        std::cout << "Results do not match!" << std::endl;
    }

    // Clean up OpenCL resources
    ret = clFlush(command_queue);
    ret = clFinish(command_queue);
    ret = clReleaseKernel(kernel);
    ret = clReleaseProgram(program);
    ret = clReleaseMemObject(results_mem_obj);
    ret = clReleaseCommandQueue(command_queue);
    ret = clReleaseContext(context);

    return 0;
}
