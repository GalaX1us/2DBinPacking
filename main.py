from enum import Enum
from data_manager import load_items_from_file, export_solutions_to_json, import_solution_from_json
from genetic_algo.gen_algo import genetic_algo, initialize_numba_functions
from genetic_algo.lgfi import lgfi
from genetic_algo.solutions_generation import get_corresponding_sequence_by_id
from visualization import visualize_bins
from tabu_search import tabu_search
import os
import time

class Metaheuristic(Enum):
    GA = 0
    TABU = 1

# Parameters for Genetic Algorithm
POPULATION_SIZE = 20
NB_GENERATIONS = 200
CROSSOVER_RATE = 0.7

MUTATION_RATE = 1

# Parameters for the Unified Tabu Search
ITERATION_NUMBER = 1000
TABU_LIST_SIZE = 20

# Genetic Parameters
KAPPA = 5 # Must be >= 1
DELTA = 5 # Must be >= 1

GUILLOTINE = True
ROTATION = True

INPUT_DATA_DIRECTORY = "data"
OUTPUT_DATA_DIRECTORY = "solutions"

SELECTED_METAHEURISTIC = Metaheuristic.GA

if __name__ == "__main__":

    
    # ====================== Compile code ======================
    # You should first execute this single line to compile everything and cache the compiled code
    # This allows future compilations time to be waaaaaaay faster
    
    initialize_numba_functions(advanced=True)
    
    # ====================== Generate Solutions ======================

    for filename in os.listdir(INPUT_DATA_DIRECTORY):
        if filename.endswith(".bp2d"):
            full_path = os.path.join(INPUT_DATA_DIRECTORY, filename)
            
            bin_width, bin_height, items = load_items_from_file(full_path)
            
            print(f"===================== {filename} =====================")
            print(f"Bin dimensions: {bin_width}x{bin_height}")
            print(f"Number of items: {len(items)}")
            
            start = time.perf_counter()
            
            # Check if the selcted metaheursitic is a enum value
            if SELECTED_METAHEURISTIC == Metaheuristic.TABU:
            # ====================== Tabu Search ======================
                best_solution, best_fitness = tabu_search(items = items,
                                                          bin_dimensions=(bin_width, bin_height),
                                                          iteration_number=ITERATION_NUMBER,
                                                          tabu_list_size=TABU_LIST_SIZE,
                                                          kappa=KAPPA,
                                                          guillotine_cut=GUILLOTINE,
                                                          rotation=ROTATION)
                
                print(best_solution)
            else:
            # ====================== Genetic Algo ======================
                best_solution, best_fitness = genetic_algo(items = items,
                                                        bin_dimensions=(bin_width, bin_height),
                                                        population_size=POPULATION_SIZE,
                                                        nb_generations=NB_GENERATIONS,
                                                        crossover_rate=CROSSOVER_RATE,
                                                        mutation_rate=MUTATION_RATE,
                                                        kappa=KAPPA,
                                                        delta=DELTA,
                                                        guillotine_cut=GUILLOTINE,
                                                        rotation=ROTATION)
            
            ordered_items = get_corresponding_sequence_by_id(items, best_solution)
            solution = lgfi(ordered_items, bin_width=bin_width, bin_height=bin_height, 
                            guillotine_cut=GUILLOTINE, rotation=ROTATION)
            time_elapsed = time.perf_counter() - start
            
            solution_file_path = os.path.join(OUTPUT_DATA_DIRECTORY, filename.split('.')[0] + "-solution.bp2d") 
            export_solutions_to_json(solution, solution_file_path)
            
            
            print(f"Time elapsed: {time_elapsed:.1f} seconds")
            print(f"Best solution: {len(solution)} bins")
            print(f"Solution saved to: {solution_file_path}\n")
    
    # ====================== Visualize Solutions ======================
           
    # solution_file_path = os.path.join(OUTPUT_DATA_DIRECTORY, "binpacking2d-04" + "-solution.bp2d") 
    # bins = import_solution_from_json(solution_file_path)
    # visualize_bins(bins)