import time
from typing import Tuple

import tqdm
from genetic_algo.mutation import *
from genetic_algo.solutions_generation import *
from genetic_algo.crossover import *
from structures import *
from fitness import *
from genetic_algo.lgfi import *

def genetic_algo(items: np.ndarray,
                 bin_dimensions: Tuple[int, int],
                 population_size: int,
                 nb_generations: int,
                 crossover_rate: float,
                 mutation_rate: float,
                 kappa: float,
                 delta: float,
                 guillotine_cut: bool,
                 rotation: bool) -> Tuple[np.ndarray, float]:
    """
    Apply the genetic algorithm to optimize the packing of items into bins.

    Parameters:
    - items (np.ndarray): Array of items to be packed.
    - bin_dimensions (tuple): Tuple containing the width and height of the bin.
    - population_size (int): The size of the population in each generation.
    - nb_generations (int): The number of generations to run the genetic algorithm.
    - crossover_rate (float): The probability of crossover between two parents.
    - mutation (float): The probability of mutation of an individual.
    - kappa (float): Parameter controlling the probability distribution in generating solutions.
    - delta (float): Parameter controlling the randomness in crossover.
    - guillotine_cut (bool): Should the guillotine cut rule be applied.
    - rotation (bool): Should the items be able to rotate

    Returns:
    - tuple: A tuple containing the best solution found and its corresponding fitness.
    """
    
    population = generate_population(items, population_size, kappa)
    
    best_solution = np.zeros_like(population[0], dtype=np.int32)
    best_fitness = np.inf
    
    
    for _ in range(nb_generations):
        
        fitnesses = compute_fitnesses(population, items, bin_dimensions, guillotine_cut, rotation)
        
        # Store best generation
        best_index = np.argmin(fitnesses)
        current_best_fitness = fitnesses[best_index]
        if current_best_fitness < best_fitness:
            best_fitness = current_best_fitness
            best_solution[:] = population[best_index]
        
        num_crossover = int(crossover_rate * population_size)
        # Create the new population with crossover
        population[:num_crossover] = crossover(population, fitnesses, np.float64(crossover_rate), delta)
        # Fill the rest with a simple roulette wheel selection based on the deterministic sequence
        population[num_crossover:] = generate_population(items, population_size - num_crossover, kappa)
        
        population = mutation(population, mutation_rate)
        
    return best_solution, best_fitness 
    
def initialize_numba_functions(advanced=False):
    """
    Initialize Numba-compiled functions with dummy arguments and measure compilation time.
    This function compiles all specified functions into machine code using Numba and caches the results for faster execution in future calls.

    Parameters:
    - advanced (bool): If True, prints detailed compilation times for each function. Default is False.
    """
    
    if advanced:
        print("===================== Compilation (Advanced Mode) =====================",flush=True)

    bin = create_bin(1, 100, 100)
    
    # Functions to copmile with their corresponding dummy arguments
    functions_with_args = {
        create_bin: (1, 10, 10),
        create_free_rectangle: (0, 0, 5, 5),
        create_item: (1, 3, 3),
        add_item_to_bin: (bin, create_item(0, 0, 0), 0, 0),
        add_free_rect_to_bin: (bin, create_free_rectangle(0, 0, 0, 0)),
        remove_free_rect_from_bin: (bin, create_free_rectangle(0, 0, 0, 0)),
        remove_free_rect_from_bin_by_idx: (bin, 0),
        get_item_by_id: (np.zeros(5, dtype=Item), 1),
        custom_choice: (np.arange(5), np.random.random(5).astype(np.float64)),
        generate_population: (np.zeros(5, dtype=Item), 10, 0.5),
        get_corresponding_sequence_by_id: (np.zeros(5, dtype=Item), np.arange(5)),
        mutation: (np.zeros((5, 5), dtype=np.int32), 0.5),  
        swap_individual: (np.arange(5),),  
        rotate_individual: (np.arange(5),),  
        remove_item_from_remaining: (np.zeros(5, dtype=Item), 1),  
        spliting_process_guillotine: (True, bin, create_free_rectangle(0, 0, 0, 0), create_item(0, 0, 0)),  
        merge_rec_guillotine: (bin,),  
        handle_wastage: (bin, create_free_rectangle(0, 0, 0, 0), 0, 0),  
        check_fit_and_rotation: (np.zeros(5, dtype=Item), 0, 0, True),  
        perform_placement: (bin, create_free_rectangle(0, 0, 0, 0), create_item(0, 0, 0), True, 0, 0, True),  
        insert_item_lgfi: (bin, np.zeros(5, dtype=Item), True, True),  
        find_current_position_idx: (bin,),  
        lgfi: (np.empty(0, dtype=Item), 10, 10, True, True),
        calculate_bin_fill: (bin,),  
        compute_fitness: (np.array([create_item(0, 6, 6), create_item(1, 6, 6), create_item(2, 4, 4), create_item(3, 3, 3)]), np.array([0, 1, 2, 3]), (10, 10), True, True),  
        compute_fitnesses: (np.array([np.array([0, 1, 2])]), np.array([create_item(0, 6, 6),create_item(1, 6, 6),create_item(2, 4, 4)]), (10, 10), True, True),  
        offspring_generation: (np.zeros(5, dtype=np.int32), np.zeros(5, dtype=np.int32), 0, 0),  
        crossover: (np.zeros((1, 1), dtype=np.int32), np.random.random(1).astype(np.float64), 0.5, 2.0),  
    }
    
    total_compilation_time: float = 0.0
    max_func_name_len = 0
    compilation_times = {}

    for func, args in (pbar := tqdm.tqdm(functions_with_args.items())):
        
        pbar.set_description(f"Compiling {func.__name__}", refresh=True)
        
        start_time = time.perf_counter()
        func(*args)
        end_time = time.perf_counter()
        
        function_name = func.__name__
        max_func_name_len = max(max_func_name_len, len(function_name))
        
        comp_time = end_time - start_time
        
        compilation_times[func.__name__] = end_time - start_time
        
        total_compilation_time += comp_time
    
    pbar.set_description(f"Finished !", refresh=True)
    
    if advanced:
        for name, duration in compilation_times.items():
            print(f"{name:>{max_func_name_len}} -> {duration:<8.4f} seconds")
        
    print("====================================================================")    
    print(f"Total compilation time -> {total_compilation_time:.4f} seconds")
    print("====================================================================\n")