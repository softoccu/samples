#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include <random>
#include <omp.h>

const int NUM_CITIES = 50;
const int NUM_ANTS = 50;
const int MAX_ITERATIONS = 1000;
const double ALPHA = 1.0;
const double BETA = 2.0;
const double EVAPORATION_RATE = 0.5;
const double Q = 100.0;
const double INITIAL_PHEROMONE = 1.0;

std::vector<std::vector<double>> distances(NUM_CITIES, std::vector<double>(NUM_CITIES));
std::vector<std::vector<double>> pheromones(NUM_CITIES, std::vector<double>(NUM_CITIES, INITIAL_PHEROMONE));
std::vector<int> best_tour(NUM_CITIES);
double best_tour_length = std::numeric_limits<double>::max();

struct Ant {
    std::vector<int> tour;
    std::vector<bool> visited;
    double tour_length;

    Ant() : tour(NUM_CITIES), visited(NUM_CITIES, false), tour_length(0.0) {}
};

double calculate_distance(const std::pair<double, double>& city1, const std::pair<double, double>& city2) {
    return std::sqrt(std::pow(city1.first - city2.first, 2) + std::pow(city1.second - city2.second, 2));
}

void initialize_distances(const std::vector<std::pair<double, double>>& cities) {
    for (int i = 0; i < NUM_CITIES; ++i) {
        for (int j = 0; j < NUM_CITIES; ++j) {
            if (i != j) {
                distances[i][j] = calculate_distance(cities[i], cities[j]);
            }
        }
    }
}

void initialize_ants(std::vector<Ant>& ants) {
    for (auto& ant : ants) {
        ant.tour_length = 0.0;
        std::fill(ant.visited.begin(), ant.visited.end(), false);
        
        int start_city = rand() % NUM_CITIES;
        ant.tour[0] = start_city;
        ant.visited[start_city] = true;
    }
}

int select_next_city(const Ant& ant, int current_city) {
    std::vector<double> probabilities(NUM_CITIES, 0.0);
    double sum = 0.0;
    for (int i = 0; i < NUM_CITIES; ++i) {
        if (!ant.visited[i]) {
            probabilities[i] = std::pow(pheromones[current_city][i], ALPHA) * std::pow(1.0 / distances[current_city][i], BETA);
            sum += probabilities[i];
        }
    }
    double threshold = ((double)rand() / RAND_MAX) * sum;
    sum = 0.0;
    for (int i = 0; i < NUM_CITIES; ++i) {
        if (!ant.visited[i]) {
            sum += probabilities[i];
            if (sum >= threshold) {
                return i;
            }
        }
    }
    return -1;
}

void update_pheromones(const std::vector<Ant>& ants) {
    for (int i = 0; i < NUM_CITIES; ++i) {
        for (int j = 0; j < NUM_CITIES; ++j) {
            pheromones[i][j] *= (1.0 - EVAPORATION_RATE);
        }
    }
    for (const auto& ant : ants) {
        for (int i = 0; i < NUM_CITIES - 1; ++i) {
            int from = ant.tour[i];
            int to = ant.tour[i + 1];
            pheromones[from][to] += Q / ant.tour_length;
            pheromones[to][from] += Q / ant.tour_length;
        }
        int from = ant.tour[NUM_CITIES - 1];
        int to = ant.tour[0];
        pheromones[from][to] += Q / ant.tour_length;
        pheromones[to][from] += Q / ant.tour_length;
    }
}

int main() {
    // Generate random cities
    std::vector<std::pair<double, double>> cities(NUM_CITIES);
    std::random_device rd;
    std::default_random_engine generator(rd());
    std::uniform_real_distribution<double> distribution(0.0, 100.0);
    for (int i = 0; i < NUM_CITIES; ++i) {
        cities[i] = {distribution(generator), distribution(generator)};
    }

    // Initialize distances
    initialize_distances(cities);

    // Initialize ants
    std::vector<Ant> ants(NUM_ANTS);

    // Main optimization loop
    for (int iteration = 0; iteration < MAX_ITERATIONS; ++iteration) {
        initialize_ants(ants);

        #pragma omp parallel for
        for (int k = 0; k < NUM_ANTS; ++k) {
            for (int i = 1; i < NUM_CITIES; ++i) {
                int current_city = ants[k].tour[i - 1];
                int next_city = select_next_city(ants[k], current_city);
                ants[k].tour[i] = next_city;
                ants[k].visited[next_city] = true;
                ants[k].tour_length += distances[current_city][next_city];
            }
            ants[k].tour_length += distances[ants[k].tour[NUM_CITIES - 1]][ants[k].tour[0]];

            #pragma omp critical
            {
                if (ants[k].tour_length < best_tour_length) {
                    best_tour_length = ants[k].tour_length;
                    best_tour = ants[k].tour;
                }
            }
        }

        update_pheromones(ants);
    }

    // Print best tour and its length
    std::cout << "Best tour length: " << best_tour_length << std::endl;
    std::cout << "Best tour: ";
    for (int city : best_tour) {
        std::cout << city << " ";
    }
    std::cout << std::endl;

    return 0;
}
