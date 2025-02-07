#include <CL/cl.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <random>


// OpenCL kernel to perform matrix multiplication
const char *kernelSource = R"(
__kernel void matrix_multiplication_kernel(__global float* A, __global float* B, __global float* C, int N) {
    int row = get_global_id(0);
    int col = get_global_id(1);
    if(row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}
)";

// Helper function to perform matrix multiplication on host
void matrix_multiplication_host(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C, int N) {
    for (int row = 0; row < N; row++) {
        for (int col = 0; col < N; col++) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++) {
                sum += A[row * N + k] * B[k * N + col];
            }
            C[row * N + col] = sum;
        }
    }
}

int main() {
    const int N = 1024;  // Matrix size (N x N)
    std::vector<float> A(N * N);
    std::vector<float> B(N * N);
    std::vector<float> C_host(N * N);
    std::vector<float> C_device(N * N);

    // Generate random matrices A and B
    std::random_device rd;
    std::default_random_engine generator(rd());
    std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
    for (int i = 0; i < N * N; i++) {
        A[i] = distribution(generator);
        B[i] = distribution(generator);
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
    cl_mem A_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, A.size() * sizeof(float), A.data(), &ret);
    cl_mem B_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, B.size() * sizeof(float), B.data(), &ret);
    cl_mem C_mem_obj = clCreateBuffer(context, CL_MEM_WRITE_ONLY, C_device.size() * sizeof(float), NULL, &ret);

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
    cl_kernel kernel = clCreateKernel(program, "matrix_multiplication_kernel", &ret);

    if (ret != CL_SUCCESS) {
        std::cerr << "Failed to create kernel." << std::endl;
        return -1;
    }

    // Set kernel arguments
    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&A_mem_obj);
    ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&B_mem_obj);
    ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&C_mem_obj);
    ret = clSetKernelArg(kernel, 3, sizeof(int), (void *)&N);

    if (ret != CL_SUCCESS) {
        std::cerr << "Failed to set kernel arguments." << std::endl;
        return -1;
    }

    size_t global_item_size[] = { (size_t)N, (size_t)N };
    size_t local_item_size[] = { 16, 16 };

    // Execute the matrix multiplication on host
    auto start = std::chrono::high_resolution_clock::now();
    matrix_multiplication_host(A, B, C_host, N);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Host function time: " << duration.count() << " seconds" << std::endl;

    // Execute the matrix multiplication on device
    start = std::chrono::high_resolution_clock::now();
    ret = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, global_item_size, local_item_size, 0, NULL, NULL);

    if (ret != CL_SUCCESS) {
        std::cerr << "Failed to enqueue NDRange kernel." << std::endl;
        return -1;
    }

    ret = clFinish(command_queue);
    if (ret != CL_SUCCESS) {
        std::cerr << "Failed to finish command queue." << std::endl;
        return -1;
    }

    // Read the memory buffer C_mem_obj on the device to the local variable C_device
    ret = clEnqueueReadBuffer(command_queue, C_mem_obj, CL_TRUE, 0, C_device.size() * sizeof(float), C_device.data(), 0, NULL, NULL);
    if (ret != CL_SUCCESS) {
        std::cerr << "Failed to read results buffer." << std::endl;
        return -1;
    }

    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    std::cout << "OpenCL function time: " << duration.count() << " seconds" << std::endl;

    // Verify results
    bool match = true;
    for (int i = 0; i < N * N; i++) {
        if (std::fabs(C_host[i] - C_device[i]) > 1e-4) {
            std::cout << "host=" << C_host[i] << ",device = " << C_device[i] << " not match, diff =  " << C_host[i] - C_device[i] << std::endl;
            match = false;
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
    ret = clReleaseMemObject(A_mem_obj);
    ret = clReleaseMemObject(B_mem_obj);
    ret = clReleaseMemObject(C_mem_obj);
    ret = clReleaseCommandQueue(command_queue);
    ret = clReleaseContext(context);

    return 0;
}
