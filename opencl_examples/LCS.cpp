#include <CL/cl.h>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <chrono>
#include <random>

// OpenCL kernel to perform LCS
const char *kernelSource = R"(
__kernel void lcs_kernel(__global const char* X, __global const char* Y, __global int* L, int m, int n, int offset) {
    int idx = get_global_id(0);

    int i, j;
    if(offset < n){
        i = idx + 1;
        j = offset - idx + 1;
    } else {
        i = offset - (n - 1) + idx + 1;
        j = (n - 1) - idx + 1;
    }

    if (i >= 1 && i <= m && j >= 1 && j <= n) {
        if (X[i - 1] == Y[j - 1])
            L[i * (n + 1) + j] = L[(i - 1) * (n + 1) + (j - 1)] + 1;
        else
            L[i * (n + 1) + j] = max(L[(i - 1) * (n + 1) + j], L[i * (n + 1) + (j - 1)]);
    }
}
)";

// Helper function to perform LCS on host
int lcs_host(const std::string& X, const std::string& Y, int m, int n) {
    std::vector<std::vector<int>> L(m + 1, std::vector<int>(n + 1, 0));

    for (int i = 1; i <= m; ++i) {
        for (int j = 1; j <= n; ++j) {
            if (X[i - 1] == Y[j - 1]) {
                L[i][j] = L[i - 1][j - 1] + 1;
            } else {
                L[i][j] = std::max(L[i - 1][j], L[i][j - 1]);
            }
        }
    }

    return L[m][n];
}

int main() {
    const int length = 2048;
    std::string X(length, ' ');
    std::string Y(length, ' ');

    // Generate random strings
    std::random_device rd;
    std::default_random_engine generator(rd());
    std::uniform_int_distribution<int> distribution(0, 25);
    for (int i = 0; i < length; ++i) {
        X[i] = 'A' + distribution(generator);
        Y[i] = 'A' + distribution(generator);
    }

    int m = X.size();
    int n = Y.size();

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
    cl_mem X_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, m * sizeof(char), X.data(), &ret);
    cl_mem Y_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, n * sizeof(char), Y.data(), &ret);
    cl_mem L_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE, (m + 1) * (n + 1) * sizeof(int), NULL, &ret);

    if (ret != CL_SUCCESS) {
        std::cerr << "Failed to create buffer objects." << std::endl;
        return -1;
    }

    std::vector<int> L((m + 1) * (n + 1), 0);
    ret = clEnqueueWriteBuffer(command_queue, L_mem_obj, CL_TRUE, 0, (m + 1) * (n + 1) * sizeof(int), L.data(), 0, NULL, NULL);
    if (ret != CL_SUCCESS) {
        std::cerr << "Failed to write L buffer." << std::endl;
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
    cl_kernel kernel = clCreateKernel(program, "lcs_kernel", &ret);

    if (ret != CL_SUCCESS) {
        std::cerr << "Failed to create kernel." << std::endl;
        return -1;
    }

    // Execute the LCS on host
    auto start = std::chrono::high_resolution_clock::now();
    int lcs_length_host = lcs_host(X, Y, m, n);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Host function time: " << duration.count() << " seconds" << std::endl;

    // Set kernel arguments
    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&X_mem_obj);
    ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&Y_mem_obj);
    ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&L_mem_obj);
    ret = clSetKernelArg(kernel, 3, sizeof(int), (void *)&m);
    ret = clSetKernelArg(kernel, 4, sizeof(int), (void *)&n);

    // Execute the LCS on device
    start = std::chrono::high_resolution_clock::now();
    for(int offset = 0; offset < (m+n); ++offset){

        ret = clSetKernelArg(kernel, 5, sizeof(int), (void *)&offset);

        if (ret != CL_SUCCESS) {
            std::cerr << "Failed to set kernel arguments." << std::endl;
            return -1;
        }

        size_t global_work_size = std::min(m,n);
        size_t local_work_size = 256 ;

        ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL);
        if (ret != CL_SUCCESS) {
            std::cerr << "Failed to enqueue NDRange kernel." << std::endl;
            return -1;
        }
    }
    ret = clFinish(command_queue);
    if (ret != CL_SUCCESS) {
        std::cerr << "Failed to finish command queue." << std::endl;
        return -1;
    }

    // Read the memory buffer L_mem_obj on the device to the local variable L
    ret = clEnqueueReadBuffer(command_queue, L_mem_obj, CL_TRUE, 0, (m + 1) * (n + 1) * sizeof(int), L.data(), 0, NULL, NULL);
    if (ret != CL_SUCCESS) {
        std::cerr << "Failed to read L buffer." << std::endl;
        return -1;
    }

    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    std::cout << "OpenCL function time: " << duration.count() << " seconds" << std::endl;

    // Verify results
    int lcs_length_opencl = L[m * (n + 1) + n];
    if (lcs_length_host == lcs_length_opencl) {
        std::cout << "Results match! LCS length: " << lcs_length_host << std::endl;
    } else {
        std::cout << "Results do not match! Host LCS length: " << lcs_length_host << ", OpenCL LCS length: " << lcs_length_opencl << std::endl;
    }

    // Clean up OpenCL resources
    ret = clFlush(command_queue);
    ret = clFinish(command_queue);
    ret = clReleaseKernel(kernel);
    ret = clReleaseProgram(program);
    ret = clReleaseMemObject(X_mem_obj);
    ret = clReleaseMemObject(Y_mem_obj);
    ret = clReleaseMemObject(L_mem_obj);
    ret = clReleaseCommandQueue(command_queue);
    ret = clReleaseContext(context);

    return 0;
}
