import argparse
import os
import random
import sys
import time

import matplotlib.pyplot as plt
import numpy as np


# Function to calculate the Euclidean distance between two points
def euclidean_distance(point1, point2):
    return np.linalg.norm(point1 - point2)


# Calculate the total distance of a path
def calculate_total_distance(path, coordinates):
    return sum(
        euclidean_distance(coordinates[path[i]], coordinates[path[i + 1]])
        for i in range(len(path) - 1)
    ) + euclidean_distance(coordinates[path[-1]], coordinates[path[0]])


# Generate initial population
def initialize_population(pop_size, num_cities):
    return [random.sample(range(num_cities), num_cities) for _ in range(pop_size)]


# Selection - Select parents using tournament selection
def selection(population, coordinates, tournament_size=5):
    tournament = random.sample(population, tournament_size)
    tournament.sort(key=lambda path: calculate_total_distance(path, coordinates))
    return tournament[0]  # Return the best individual in the tournament


# Crossover - Ordered Crossover (OX)
def crossover(parent1, parent2):
    start, end = sorted(random.sample(range(len(parent1)), 2))
    child = [None] * len(parent1)
    child[start:end] = parent1[start:end]

    # Fill remaining cities from parent2
    fill_pos = end
    for city in parent2:
        if city not in child:
            if fill_pos >= len(child):
                fill_pos = 0
            child[fill_pos] = city
            fill_pos += 1

    return child


# Mutation - Swap Mutation
def mutate(path, mutation_rate=0.01):
    for i in range(len(path)):
        if random.random() < mutation_rate:
            swap_idx = random.randint(0, len(path) - 1)
            path[i], path[swap_idx] = path[swap_idx], path[i]


# Evolve the population
def evolve_population(population, coordinates, mutation_rate=0.01, elite_size=2):
    new_population = sorted(
        population, key=lambda path: calculate_total_distance(path, coordinates)
    )[:elite_size]

    # Generate the rest of the new population
    while len(new_population) < len(population):
        parent1 = selection(population, coordinates)
        parent2 = selection(population, coordinates)
        child = crossover(parent1, parent2)
        mutate(child, mutation_rate)
        new_population.append(child)

    return new_population


# Genetic Algorithm function
def genetic_algorithm(coordinates, pop_size=100, generations=500, mutation_rate=0.01):
    num_cities = len(coordinates)
    population = initialize_population(pop_size, num_cities)
    best_path = None
    best_distance = float("inf")

    for generation in range(generations):
        population = evolve_population(population, coordinates, mutation_rate)
        current_best = min(
            population, key=lambda path: calculate_total_distance(path, coordinates)
        )
        current_distance = calculate_total_distance(current_best, coordinates)

        if current_distance < best_distance:
            best_distance = current_distance
            best_path = current_best

        if generation % 50 == 0:  # Print progress every 50 generations
            print(f"Generation {generation} - Best Distance: {best_distance}")

    return best_path, best_distance


# Visualization function
def plot_tsp_path(optimal_path, coordinates):
    plt.figure(figsize=(8, 6))
    for i, coord in enumerate(coordinates):
        plt.scatter(coord[0], coord[1], color="blue")
        plt.text(coord[0] + 0.3, coord[1] + 0.3, f"City {i}", fontsize=12)

    for i in range(len(optimal_path)):
        start_city = optimal_path[i]
        end_city = optimal_path[(i + 1) % len(optimal_path)]
        plt.plot(
            [coordinates[start_city][0], coordinates[end_city][0]],
            [coordinates[start_city][1], coordinates[end_city][1]],
            "r-",
        )

    plt.title(
        f"Optimal TSP Path with Distance: {calculate_total_distance(optimal_path, coordinates):.2f}"
    )
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid(True)
    


# Held-Karp Algorithm for TSP (Dynamic Programming)
def tsp_held_karp(coordinates):
    n = len(coordinates)
    dist = [
        [euclidean_distance(coordinates[i], coordinates[j]) for j in range(n)]
        for i in range(n)
    ]

    # Memoization table: dp[mask][i] - minimum distance to visit all cities in 'mask' ending at city 'i'
    dp = [[float("inf")] * n for _ in range(1 << n)]
    dp[1][0] = 0  # Start from city 0

    # Iterate through all subsets of cities
    for mask in range(1, 1 << n):
        for u in range(n):
            if not (mask & (1 << u)):
                continue
            for v in range(n):
                if mask & (1 << v) or u == v:
                    continue
                next_mask = mask | (1 << v)
                dp[next_mask][v] = min(dp[next_mask][v], dp[mask][u] + dist[u][v])

    # Reconstruct the optimal path
    mask = (1 << n) - 1
    last = 0
    optimal_distance = min(dp[mask][i] + dist[i][0] for i in range(1, n))
    path = [0]

    for _ in range(n - 1):
        next_city = min(
            range(n),
            key=lambda i: (
                dp[mask][i] + dist[i][last] if mask & (1 << i) else float("inf")
            ),
        )
        path.append(next_city)
        mask ^= 1 << next_city
        last = next_city
    path.append(0)

    return path, optimal_distance


def main():
    parser = argparse.ArgumentParser(
        description="Analyze algorithms from a specified file."
    )
    parser.add_argument(
        "--file", required=False, help="Path to the file containing algorithms"
    )

    output_folder = {time.strftime("%Y-%m-%d_%H-%M-%S")}
    if not os.path.exists(f"results/{output_folder}"):
        os.mkdir(f"results/{output_folder}")

    args = parser.parse_args()

    file_path = args.file

    np.random.seed(0)

    number_of_cities = 10

    # Randomly generate coordinates for each city
    coordinates = np.random.randint(0, 100, (number_of_cities, 2))

    # Run the Genetic Algorithm
    optimal_path_genetic_algorithm, optimal_distance_genetic_algorithm = (
        genetic_algorithm(
            coordinates, pop_size=100, generations=500, mutation_rate=0.01
        )
    )
    print(f"Optimal Path: {optimal_path_genetic_algorithm}")
    print(f"Optimal Distance: {optimal_distance_genetic_algorithm}")

    # Plot the result
    plot_tsp_path(optimal_path_genetic_algorithm, coordinates)
    plt.savefig(f"results/{output_folder}/genetic_algorithm.png")

    # Run the Held-Karp Algorithm
    optimal_path_held_karp, optimal_distance_held_karp = tsp_held_karp(coordinates)
    print(f"Optimal Path (Held-Karp): {optimal_path_held_karp}")
    print(f"Optimal Distance (Held-Karp): {optimal_distance_held_karp}")

    # Plot the result
    plot_tsp_path(optimal_path_held_karp, coordinates)
    plt.savefig(f"results/{output_folder}/held_karp.png")

    # Add optimal path for both algorithms to a text file
    with open(f"results/{output_folder}/optimal_paths.txt", "w") as f:
        f.write("Genetic Algorithm\n")
        f.write(f"Optimal Path: {optimal_path_genetic_algorithm}\n")
        f.write(f"Optimal Distance: {optimal_distance_genetic_algorithm}\n\n")

        f.write("Held-Karp Algorithm\n")
        f.write(f"Optimal Path: {optimal_path_held_karp}\n")
        f.write(f"Optimal Distance: {optimal_distance_held_karp}\n")


if __name__ == "__main__":
    main()
