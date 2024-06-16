from enum import Enum
import os
import time

from binpacking.data_manager import export_solutions_to_json, import_solution_from_json, load_items_from_file
from binpacking.genetic_algo.gen_algo import genetic_algo, initialize_numba_functions
from binpacking.lgfi import lgfi
from binpacking.population_generation import get_corresponding_sequence_by_id
from binpacking.tabu_search import tabu_search
from binpacking.visualization import visualize_bins
from solutions_helper import generate_all_solutions, generate_single_solution, Metaheuristic, visualize_solution
    
INPUT_DATA_DIRECTORY = "data"
OUTPUT_DATA_DIRECTORY = "solutions"

# Parameters for Genetic Algorithm
POPULATION_SIZE = 10
NB_GENERATIONS = 1000
CROSSOVER_RATE = 0.7

MUTATION_RATE = 0.5

# Parameters for the Unified Tabu Search
ITERATION_NUMBER = 200
TABU_LIST_SIZE = 10

# Genetic Parameters
KAPPA = 5 # Must be >= 1 (For both GA and TABU)
DELTA = 20 # Must be >= 1 (Only for GA)

GUILLOTINE = True
ROTATION = True

SELECTED_METAHEURISTIC = Metaheuristic.GA

if __name__ == "__main__":

    
    # ====================== Compile code ======================
    # You should first execute this single line to compile everything and cache the compiled code
    # This allows future compilations time to be waaaaaaay faster
    
    initialize_numba_functions(advanced=True)
    
    # ====================== Generate All Solutions ======================

    generate_all_solutions(SELECTED_METAHEURISTIC, ITERATION_NUMBER, TABU_LIST_SIZE, KAPPA, GUILLOTINE, ROTATION, 
                           POPULATION_SIZE, NB_GENERATIONS, CROSSOVER_RATE, MUTATION_RATE, DELTA, 
                           INPUT_DATA_DIRECTORY, OUTPUT_DATA_DIRECTORY)
            
    # =================== Generate One Solutions ==================
    
    # file = "binpacking2d-04.bp2d" # Just chnage the dataset number to generate another solution
    
    # generate_single_solution(file, SELECTED_METAHEURISTIC, ITERATION_NUMBER, TABU_LIST_SIZE, KAPPA, GUILLOTINE, ROTATION,
    #                          POPULATION_SIZE, NB_GENERATIONS, CROSSOVER_RATE, MUTATION_RATE, DELTA, INPUT_DATA_DIRECTORY, OUTPUT_DATA_DIRECTORY)
    
    
    # ====================== Visualize Solutions ======================
    
    file = "binpacking2d-04-solution.json" # Just chnage the solution number to visualize another solution
    visualize_solution(file, OUTPUT_DATA_DIRECTORY)