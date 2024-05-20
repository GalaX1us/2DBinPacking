import os
import time
import csv
from itertools import product

import tqdm

from data_manager import load_items_from_file
from genetic_algo.gen_algo import genetic_algo
from genetic_algo.lgfi import lgfi
from genetic_algo.solutions_generation import get_corresponding_sequence_by_id

# Define the parameters ranges
kappa = [1, 5, 10]
delta = [1, 5, 10]
mutation_rate = [0.1, 0.5, 1.0]
crossover_rate = [0.6, 0.7, 0.8]

# Directory paths
INPUT_DATA_DIRECTORY = "data"
RESULTS_FILE = "genetic_algorithm_results.csv"

with open(RESULTS_FILE, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["filename", "kappa", "delta", "crossover_rate", "mutation_rate",  "bins", "time_elapsed"])

# Create a list of all parameter combinations
all_combinations = list(product(kappa, delta, crossover_rate, mutation_rate))

for k, d, cr, mr in tqdm.tqdm(all_combinations, desc="Processing parameter combinations"):
    for filename in os.listdir(INPUT_DATA_DIRECTORY):
        if filename.endswith(".bp2d"):
            full_path = os.path.join(INPUT_DATA_DIRECTORY, filename)
            
            bin_width, bin_height, items = load_items_from_file(full_path)
            
            start = time.perf_counter()
            best_solution, best_fitness = genetic_algo(
                items=items,
                bin_dimensions=(bin_width, bin_height),
                population_size=50,
                nb_generations=200,
                crossover_rate=cr,
                mutation_rate=mr,
                kappa=k,
                delta=d,
                guillotine_cut=True,
                rotation=True
            )
            
            ordered_items = get_corresponding_sequence_by_id(items, best_solution)
            solution = lgfi(ordered_items, bin_width=bin_width, bin_height=bin_height, 
                            guillotine_cut=True, rotation=True)
            time_elapsed = time.perf_counter() - start
            
            # Save the results to the CSV file
            with open(RESULTS_FILE, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([filename, k, d, cr, mr, len(solution), time_elapsed])

# Find the best combination of parameters based on the least number of bins used
import pandas as pd

# Analyze results to find the best solution for each filename
results_df = pd.read_csv(RESULTS_FILE)
best_results_per_file = results_df.loc[results_df.groupby('filename')['bins'].idxmin()]

print("Best combinations of parameters for each file:")
print(best_results_per_file)