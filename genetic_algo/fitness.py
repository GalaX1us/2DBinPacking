from genetic_algo.structures import *

from numba import prange

from genetic_algo.lgfi import lgfi
from genetic_algo.solutions_generation import get_corresponding_sequence_by_id

@njit(cache = True)
def calculate_bin_fill(bin):
    """
    Calculate the total fill of a bin based on the items placed in it.
    
    Parameters:
    - bin (np.ndarray): A bin structured array with items and their placements.
    
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
def compute_fitness(bins, capacity, k=2):
    """
    Calculate the fitness of a bin packing solution.

    Parameters:
    - bins (np.ndarray): Array of bin structured arrays.
    - capacity (int): The maximum capacity (area) of each bin.
    - k (float): Exponent to control the preference for more filled bins.

    Returns:
    - float: The calculated fitness value of the bin packing solution.
    """
    n = len(bins)
    if n == 0:
        return 0.0 

    total_fitness = 0
    for i in range(n):
        fill_ratio = calculate_bin_fill(bins[i]) / capacity
        total_fitness = total_fitness + fill_ratio ** k

    return total_fitness / n

@njit(parallel = True)
def compute_fitnesses(population, items, bin_width, bin_height):
    fitnesses = np.zeros(len(population))
    
    for i in prange(population.shape[0]):
        
        sequence = get_corresponding_sequence_by_id(items, population[i])
        
        solution = lgfi(sequence, bin_width, bin_height)
        fitnesses[i] = compute_fitness(solution, bin_width*bin_height)
        
    return fitnesses