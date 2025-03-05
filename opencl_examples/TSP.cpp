#include <CL/cl.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include <chrono>
#include <random>
#include <algorithm>

const int NUM_CITIES = 64;
const int NUM_ANTS = 64;
const int MAX_ITERATIONS = 1000;
const float ALPHA = 1.0f;
const float BETA = 5.0f;
const float EVAPORATION_RATE = 0.5f;
const float Q = 100.0f;

// OpenCL kernel for updating pheromones
const char *kernelSource = R"(
__kernel void update_pheromones(__global float* pheromones, __global int* ant_routes, __global float* distances, int num_ants, int num_cities, float evaporation_rate, float Q) {
    int i = get_global_id(0);
    int j = get_global_id(1);
    
    pheromones[i * num_cities + j] *= (1.0f - evaporation_rate);

    for (int k = 0; k < num_ants; k++) {
        float route_length = 0.0f;
        for (int l = 0; l < num_cities; l++) {
            int city1 = ant_routes[k * num_cities + l];
            int city2 = ant_routes[k * num_cities + (l + 1) % num_cities];
            route_length += distances[city1 * num_cities + city2];
        }
        pheromones[i * num_cities + j] += Q / route_length;
    }
}
)";

// Function to calculate the distance between two cities
float calculate_distance(const std::pair<float, float>& city1, const std::pair<float, float>& city2) {
    return std::sqrt(std::pow(city1.first - city2.first, 2) + std::pow(city1.second - city2.second, 2));
}

// Function to initialize pheromones
void initialize_pheromones(std::vector<float>& pheromones, int num_cities) {
    std::fill(pheromones.begin(), pheromones.end(), 1.0f);
}

// Function to initialize distances
void initialize_distances(const std::vector<std::pair<float, float>>& cities, std::vector<float>& distances) {
    int num_cities = cities.size();
    for (int i = 0; i < num_cities; i++) {
        for (int j = 0; j < num_cities; j++) {
            distances[i * num_cities + j] = calculate_distance(cities[i], cities[j]);
        }
    }
}

// Function to generate random cities
std::vector<std::pair<float, float>> generate_cities(int num_cities) {
    std::vector<std::pair<float, float>> cities(num_cities);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 100.0f);
    for (int i = 0; i < num_cities; i++) {
        cities[i] = {dis(gen), dis(gen)};
    }
    return cities;
}

// Function to perform ACO on host
std::pair<std::vector<int>, float> aco_host(const std::vector<std::pair<float, float>>& cities) {
    int num_cities = cities.size();
    std::vector<float> distances(num_cities * num_cities);
    initialize_distances(cities, distances);

    std::vector<float> pheromones(num_cities * num_cities);
    initialize_pheromones(pheromones, num_cities);

    std::vector<int> best_route(num_cities);
    float best_route_length = std::numeric_limits<float>::max();

    std::random_device rd;
    std::mt19937 gen(rd());

    for (int iteration = 0; iteration < MAX_ITERATIONS; iteration++) {
        std::vector<int> ant_routes(NUM_ANTS * num_cities);

        for (int ant = 0; ant < NUM_ANTS; ant++) {
            std::vector<bool> visited(num_cities, false);
            int current_city = gen() % num_cities;
            ant_routes[ant * num_cities] = current_city;
            visited[current_city] = true;

            for (int step = 1; step < num_cities; step++) {
                std::vector<float> probabilities(num_cities, 0.0f);
                float sum_probabilities = 0.0f;

                for (int city = 0; city < num_cities; city++) {
                    if (!visited[city]) {
                        probabilities[city] = std::pow(pheromones[current_city * num_cities + city], ALPHA) *
                                              std::pow(1.0f / distances[current_city * num_cities + city], BETA);
                        sum_probabilities += probabilities[city];
                    }
                }

                std::uniform_real_distribution<float> dis(0.0f, sum_probabilities);
                float random_value = dis(gen);

                for (int city = 0; city < num_cities; city++) {
                    if (!visited[city]) {
                        if (random_value < probabilities[city]) {
                            current_city = city;
                            break;
                        } else {
                            random_value -= probabilities[city];
                        }
                    }
                }

                ant_routes[ant * num_cities + step] = current_city;
                visited[current_city] = true;
            }
        }

        for (int ant = 0; ant < NUM_ANTS; ant++) {
            float route_length = 0.0f;
            for (int step = 0; step < num_cities; step++) {
                int city1 = ant_routes[ant * num_cities + step];
                int city2 = ant_routes[ant * num_cities + (step + 1) % num_cities];
                route_length += distances[city1 * num_cities + city2];
            }

            if (route_length < best_route_length) {
                best_route_length = route_length;
                std::copy(ant_routes.begin() + ant * num_cities, ant_routes.begin() + (ant + 1) * num_cities, best_route.begin());
            }
        }

        initialize_pheromones(pheromones, num_cities);
        for (int ant = 0; ant < NUM_ANTS; ant++) {
            float route_length = 0.0f;
            for (int step = 0; step < num_cities; step++) {
                int city1 = ant_routes[ant * num_cities + step];
                int city2 = ant_routes[ant * num_cities + (step + 1) % num_cities];
                route_length += distances[city1 * num_cities + city2];
            }

            for (int step = 0; step < num_cities; step++) {
                int city1 = ant_routes[ant * num_cities + step];
                int city2 = ant_routes[ant * num_cities + (step + 1) % num_cities];
                pheromones[city1 * num_cities + city2] += Q / route_length;
            }
        }
    }

    return {best_route, best_route_length};
}

int main() {
    // Generate cities
    std::vector<std::pair<float, float>> cities = generate_cities(NUM_CITIES);

    // Print cities
    std::cout << "Cities:" << std::endl;
    for (const auto& city : cities) {
        std::cout << "(" << city.first << ", " << city.second << ")" << std::endl;
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

    // Initialize distances
    std::vector<float> distances(NUM_CITIES * NUM_CITIES);
    initialize_distances(cities, distances);

    // Initialize pheromones
    std::vector<float> pheromones(NUM_CITIES * NUM_CITIES);
    initialize_pheromones(pheromones, NUM_CITIES);

    // Create memory buffers on the device
    cl_mem distances_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, distances.size() * sizeof(float), distances.data(), &ret);
    cl_mem pheromones_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, pheromones.size() * sizeof(float), pheromones.data(), &ret);
    cl_mem ant_routes_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE, NUM_ANTS * NUM_CITIES * sizeof(int), NULL, &ret);

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
    cl_kernel kernel = clCreateKernel(program, "update_pheromones", &ret);

    if (ret != CL_SUCCESS) {
        std::cerr << "Failed to create kernel." << std::endl;
        return -1;
    }

    // Set kernel arguments
    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&pheromones_mem_obj);
    ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&ant_routes_mem_obj);
    ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&distances_mem_obj);
    ret = clSetKernelArg(kernel, 3, sizeof(int), (void *)&NUM_ANTS);
    ret = clSetKernelArg(kernel, 4, sizeof(int), (void *)&NUM_CITIES);
    ret = clSetKernelArg(kernel, 5, sizeof(float), (void *)&EVAPORATION_RATE);
    ret = clSetKernelArg(kernel, 6, sizeof(float), (void *)&Q);

    if (ret != CL_SUCCESS) {
        std::cerr << "Failed to set kernel arguments." << std::endl;
        return -1;
    }

    size_t global_item_size[] = { (size_t)NUM_CITIES, (size_t)NUM_CITIES };
    size_t local_item_size[] = { 16, 16 };

    // Perform ACO on host
    auto start = std::chrono::high_resolution_clock::now();
    auto [best_route_host, best_route_length_host] = aco_host(cities);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Host function time: " << duration.count() << " seconds" << std::endl;

    // Print best route and length from host
    std::cout << "Best route length (host): " << best_route_length_host << std::endl;
    std::cout << "Best route (host): ";
    for (const auto& city : best_route_host) {
        std::cout << city << " ";
    }
    std::cout << std::endl;

    // Perform ACO on device
    std::vector<int> ant_routes(NUM_ANTS * NUM_CITIES);
    start = std::chrono::high_resolution_clock::now();
    for (int iteration = 0; iteration < MAX_ITERATIONS; iteration++) {
        // Generate ant routes on host
        for (int ant = 0; ant < NUM_ANTS; ant++) {
            std::vector<bool> visited(NUM_CITIES, false);
            int current_city = rand() % NUM_CITIES;
            ant_routes[ant * NUM_CITIES] = current_city;
            visited[current_city] = true;
            for (int step = 1; step < NUM_CITIES; step++) {
                std::vector<float> probabilities(NUM_CITIES, 0.0f);
                float sum_probabilities = 0.0f;
                for (int city = 0; city < NUM_CITIES; city++) {
                    if (!visited[city]) {
                        probabilities[city] = std::pow(pheromones[current_city * NUM_CITIES + city], ALPHA) *
                                              std::pow(1.0f / distances[current_city * NUM_CITIES + city], BETA);
                        sum_probabilities += probabilities[city];
                    }
                }
                float random_value = static_cast<float>(rand()) / RAND_MAX * sum_probabilities;
                for (int city = 0; city < NUM_CITIES; city++) {
                    if (!visited[city]) {
                        if (random_value < probabilities[city]) {
                            current_city = city;
                            break;
                        } else {
                            random_value -= probabilities[city];
                        }
                    }
                }
                ant_routes[ant * NUM_CITIES + step] = current_city;
                visited[current_city] = true;
            }
        }

        // Write ant routes to device
        ret = clEnqueueWriteBuffer(command_queue, ant_routes_mem_obj, CL_TRUE, 0, ant_routes.size() * sizeof(int), ant_routes.data(), 0, NULL, NULL);

        // Execute the kernel to update pheromones
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

        // Read pheromones from device
        ret = clEnqueueReadBuffer(command_queue, pheromones_mem_obj, CL_TRUE, 0, pheromones.size() * sizeof(float), pheromones.data(), 0, NULL, NULL);
        if (ret != CL_SUCCESS) {
            std::cerr << "Failed to read results buffer." << std::endl;
            return -1;
        }
    }
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    std::cout << "OpenCL function time: " << duration.count() << " seconds" << std::endl;

    // Find the best route on device
    std::vector<int> best_route_device(NUM_CITIES);
    float best_route_length_device = std::numeric_limits<float>::max();
    for (int ant = 0; ant < NUM_ANTS; ant++) {
        float route_length = 0.0f;
        std::copy(ant_routes.begin() + ant * NUM_CITIES, ant_routes.begin() + (ant + 1) * NUM_CITIES, best_route_device.begin());
        for (int step = 0; step < NUM_CITIES; step++) {
            int city1 = best_route_device[step];
            int city2 = best_route_device[(step + 1) % NUM_CITIES];
            route_length += distances[city1 * NUM_CITIES + city2];
        }
        if (route_length < best_route_length_device) {
            best_route_length_device = route_length;
        }
    }

    // Print best route and length from device
    std::cout << "Best route length (device): " << best_route_length_device << std::endl;
    std::cout << "Best route (device): ";
    for (const auto& city : best_route_device) {
        std::cout << city << " ";
    }
    std::cout << std::endl;

    // Clean up OpenCL resources
    ret = clFlush(command_queue);
    ret = clFinish(command_queue);
    ret = clReleaseKernel(kernel);
    ret = clReleaseProgram(program);
    ret = clReleaseMemObject(distances_mem_obj);
    ret = clReleaseMemObject(pheromones_mem_obj);
    ret = clReleaseMemObject(ant_routes_mem_obj);
    ret = clReleaseCommandQueue(command_queue);
    ret = clReleaseContext(context);

    return 0;
}