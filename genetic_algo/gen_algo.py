import time
from typing import Tuple
from genetic_algo.solutions_generation import *
from genetic_algo.crossover import crossover
from genetic_algo.structures import *
from genetic_algo.fitness import compute_fitnesses
from genetic_algo.lgfi import lgfi

def genetic_algo(items: np.ndarray,
                 bin_dimensions: Tuple[int, int],
                 population_size: int,
                 nb_generations: int,
                 crossover_rate: float,
                 kappa: float,
                 delta: float) -> Tuple[np.ndarray, float]:
    """
    Apply the genetic algorithm to optimize the packing of items into bins.

    Parameters:
    - items (np.ndarray): Array of items to be packed.
    - bin_dimensions (tuple): Tuple containing the width and height of the bin.
    - population_size (int): The size of the population in each generation.
    - nb_generations (int): The number of generations to run the genetic algorithm.
    - crossover_rate (float): The probability of crossover between two parents.
    - kappa (float): Parameter controlling the probability distribution in generating solutions.
    - delta (float): Parameter controlling the randomness in crossover.

    Returns:
    - tuple: A tuple containing the best solution found and its corresponding fitness.
    """
    
    # Unpack the bin dimensions
    bin_width, bin_height = bin_dimensions
    
    population = generate_population(items, population_size, kappa)
    
    best_solution = np.zeros_like(population[0])
    best_fitness = 0
    
    
    for _ in range(nb_generations):
        
        fitnesses = compute_fitnesses(population, items, bin_width, bin_height)
        
        # Store best generation
        best_index = np.argmax(fitnesses)
        current_best_fitness = fitnesses[best_index]
        if current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
            best_solution[:] = population[best_index]
        
        crossover_population = crossover(population, fitnesses, crossover_rate, delta)
        remaining_population = generate_population(items, population_size - crossover_population.shape[0] ,kappa)
        population = np.concatenate([crossover_population, remaining_population])
        
    return best_solution, best_fitness 

def compile_everything() -> None:
    """
    Compile all the functions in machine code with numba.
    """
    
    bin_width, bin_height = 10, 10
    
    start = time.perf_counter()
    
    items = np.array([
        create_item(0, 6, 6),
        create_item(1, 6, 6), 
    ])
    
    lgfi(items, bin_width, bin_height)
    pop = generate_population(items, 5, 1.0)
    fit = compute_fitnesses(pop, items, bin_width, bin_height)
    crossover(pop, fit, 0.7, 1.0)
    
    print("===========================")
    print(f"Compilation took : {time.perf_counter() - start:.1f} sec")
    print("===========================\n")
    