#include <CL/cl.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>

// OpenCL kernel to find prime numbers
const char *kernelSource = R"(
__kernel void find_primes(__global int* primes, int n) {
    int id = get_global_id(0) + 2; // Start from 2
    int is_prime = 1;

    for (int i = 2; i <= sqrt((float)n); i++) {
        if (id % i == 0 && id != i) {
            is_prime = 0;
            break;
        }
    }

    primes[id] = is_prime;
}
)";

// Host function to find prime numbers
void find_primes_host(std::vector<int>& primes, int range) {
    for (int num = 2; num <= range; num++) {
        bool is_prime = true;
        for (int i = 2; i <= std::sqrt(num); i++) {
            if (num % i == 0) {
                is_prime = false;
                break;
            }
        }
        primes[num] = is_prime ? 1 : 0;
    }
}

int main() {
    const int range = 1000000;
    std::vector<int> primes_opencl(range + 1, 1); // Initialize all numbers as prime
    std::vector<int> primes_host(range + 1, 1); // Initialize all numbers as prime

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

    // Create memory buffers on the device for the primes array
    cl_mem primes_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE, (range + 1) * sizeof(int), NULL, &ret);

    if (ret != CL_SUCCESS) {
        std::cerr << "Failed to create buffer object." << std::endl;
        return -1;
    }

    // Copy the primes array to the memory buffer
    ret = clEnqueueWriteBuffer(command_queue, primes_mem_obj, CL_TRUE, 0, (range + 1) * sizeof(int), primes_opencl.data(), 0, NULL, NULL);

    if (ret != CL_SUCCESS) {
        std::cerr << "Failed to write buffer." << std::endl;
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
    cl_kernel kernel = clCreateKernel(program, "find_primes", &ret);

    if (ret != CL_SUCCESS) {
        std::cerr << "Failed to create kernel." << std::endl;
        return -1;
    }

    // Set the arguments of the kernel
    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&primes_mem_obj);
    ret = clSetKernelArg(kernel, 1, sizeof(int), (void *)&range);

    if (ret != CL_SUCCESS) {
        std::cerr << "Failed to set kernel arguments." << std::endl;
        return -1;
    }

    // Execute the OpenCL kernel on the list
    size_t local_item_size = 256; // Divide work items into groups of 64
    // alignment with local_item_size
    size_t global_item_size = (range + local_item_size - 1) & ~0xffULL; // Process the range from 2 to 1000000
    
    ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_item_size, &local_item_size, 0, NULL, NULL);

    if (ret != CL_SUCCESS) {
        std::cerr << "Failed to enqueue NDRange kernel." << std::endl;
        return -1;
    }

    // Read the memory buffer primes_mem_obj on the device to the local variable primes_opencl
    ret = clEnqueueReadBuffer(command_queue, primes_mem_obj, CL_TRUE, 0, (range + 1) * sizeof(int), primes_opencl.data(), 0, NULL, NULL);

    if (ret != CL_SUCCESS) {
        std::cerr << "Failed to read buffer." << std::endl;
        return -1;
    }

    // Clean up OpenCL resources
    ret = clFlush(command_queue);
    ret = clFinish(command_queue);
    ret = clReleaseKernel(kernel);
    ret = clReleaseProgram(program);
    ret = clReleaseMemObject(primes_mem_obj);
    ret = clReleaseCommandQueue(command_queue);
    ret = clReleaseContext(context);

    // Find primes using the host function
    auto start = std::chrono::high_resolution_clock::now();
    find_primes_host(primes_host, range);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Host function time: " << duration.count() << " seconds" << std::endl;

    // Verify and count results
    bool match = true;
    int count_opencl = 0;
    int count_host = 0;
    for (int i = 2; i <= range; i++) {
        if (primes_opencl[i]) count_opencl++;
        if (primes_host[i]) count_host++;
        if (primes_opencl[i] != primes_host[i]) {
            match = false;
            std::cout << "Mismatch at " << i << ": OpenCL = " << primes_opencl[i] << ", Host = " << primes_host[i] << std::endl;
        }
    }

    if (match) {
        std::cout << "Results match!" << std::endl;
    } else {
        std::cout << "Results do not match!" << std::endl;
    }

    std::cout << "Number of primes found by OpenCL: " << count_opencl << std::endl;
    std::cout << "Number of primes found by Host: " << count_host << std::endl;

    return 0;
}