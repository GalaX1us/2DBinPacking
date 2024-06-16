from numba import config

# Easy way to disable or Enable JIT Compilation with Numba
# Useful for performance comparison
config.DISABLE_JIT = False

from binpacking.genetic_algo.gen_algo import initialize_numba_functions
from solutions_helper import generate_all_solutions, generate_single_solution, Metaheuristic, visualize_solution
    
INPUT_DATA_DIRECTORY = "data"
OUTPUT_DATA_DIRECTORY = "solutions"

# Parameters for Genetic Algorithm
POPULATION_SIZE = 20
NB_GENERATIONS = 1000
CROSSOVER_RATE = 0.7

MUTATION_RATE = 0.5

# Parameters for the Unified Tabu Search
ITERATION_NUMBER = 200
TABU_LIST_SIZE = 10

# Genetic Parameters
KAPPA = 5 # Must be >= 1 (For both GA and TABU)
DELTA = 5 # Must be >= 1 (Only for GA)

GUILLOTINE = True
ROTATION = True

SELECTED_METAHEURISTIC = Metaheuristic.GA

assert KAPPA >= 1, "KAPPA must be >= 1"
assert DELTA >= 1, "DELTA must be >= 1"

if __name__ == "__main__":

    # ====================== Compile code ======================
    # You should first execute this single line to compile everything and cache the compiled code
    # This allows future compilations time to be waaaaaaay faster
    
    initialize_numba_functions(advanced=True)
    
    # ====================== Generate All Solutions ======================

    # generate_all_solutions(SELECTED_METAHEURISTIC, ITERATION_NUMBER, TABU_LIST_SIZE, KAPPA, GUILLOTINE, ROTATION, 
    #                        POPULATION_SIZE, NB_GENERATIONS, CROSSOVER_RATE, MUTATION_RATE, DELTA, 
    #                        INPUT_DATA_DIRECTORY, OUTPUT_DATA_DIRECTORY)
            
    # =================== Generate One Solutions ==================
    
    # file = "binpacking2d-11.bp2d" # Just chnage the dataset number to generate another solution
    
    # generate_single_solution(file, SELECTED_METAHEURISTIC, ITERATION_NUMBER, TABU_LIST_SIZE, KAPPA, GUILLOTINE, ROTATION,
    #                          POPULATION_SIZE, NB_GENERATIONS, CROSSOVER_RATE, MUTATION_RATE, DELTA, INPUT_DATA_DIRECTORY, OUTPUT_DATA_DIRECTORY)
    
    
    # ====================== Visualize Solutions ======================
    
    file = "binpacking2d-13-solution.json" # Just chnage the solution number to visualize another solution
    visualize_solution(file, OUTPUT_DATA_DIRECTORY)