from genetic_algo.structures import *

from numba import prange

from genetic_algo.lgfi import lgfi
from genetic_algo.solutions_generation import get_corresponding_sequence_by_id

@njit(cache = True)
def calculate_bin_fill(bin: np.ndarray) -> int:
    """
    Calculate the total fill of a bin based on the items placed in it.
    
    Parameters:
    - bin (np.ndarray): A structured array representing a bin, containing items with their placements.
    
    Returns:
    - int: Total filled area of the bin.
    """
    total_fill = 0
    for i in range(bin['items'].shape[0]):
        item = bin['items'][i]
        
        if item['id'] == -1 or item['width'] == 0:
            break
        
        total_fill += item['width'] * item['height']
        
    return total_fill

@njit(cache = True)
def compute_fitness(bins: np.ndarray, capacity: int, k: float = 2) -> float:
    """
    Calculate the fitness of a bin packing solution.

    Parameters:
    - bins (np.ndarray): An array of structured arrays representing the bins.
    - capacity (int): The maximum capacity (area) of each bin.
    - k (float): The exponent to control the preference for more filled bins.

    Returns:
    - float: The calculated fitness value of the bin packing solution.
    """
    n = len(bins)
    if n == 0:
        return 0.0 

    total_fitness = 0
    for i in range(n):
        fill_ratio = calculate_bin_fill(bins[i]) / capacity
        
        # THis is to break ties by the load of the last bin
        if i == (n-1):
            fill_ratio /= 100
        
        total_fitness += fill_ratio ** k

    return total_fitness / n

@njit(parallel = True, cache = True)
def compute_fitnesses(population: np.ndarray, items: np.ndarray, bin_width: int, bin_height: int) -> np.ndarray:
    """
    Calculate the fitnesses of a population of bin packing solutions.

    Parameters:
    - population (np.ndarray): An array representing the population of solutions.
    - items (np.ndarray): An array of items to be packed.
    - bin_width (int): The width of each bin.
    - bin_height (int): The height of each bin.

    Returns:
    - np.ndarray: An array of fitness values for the population.
    """
    population_size = len(population)
    
    fitnesses = np.zeros(population_size)
    
    for i in prange(population_size):
        
        # This gives an array of items with a specific ordering
        sequence = get_corresponding_sequence_by_id(items, population[i])
        
        # Apply the placement heuristic to get the bins
        solution = lgfi(sequence, bin_width, bin_height)
        
        # Compute the fitness of this specfic solution (bins)
        fitnesses[i] = compute_fitness(solution, bin_width*bin_height)
        
    return fitnesses