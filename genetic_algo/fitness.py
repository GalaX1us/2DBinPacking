from typing import Tuple
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
def compute_fitness_fill(bins: np.ndarray, capacity: int, k: float = 2) -> float:
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
        
        total_fitness += fill_ratio ** k

    return total_fitness / n

@njit(cache = True)
def compute_fitness(items: np.ndarray, id_ordering: np.ndarray, bin_dimensions: Tuple[int, int], 
                    guillotine_cut: bool, rotation: bool):
    """
    Calculate the fitness of a bin packing solution using a specific order of items.

    This function evaluates a bin packing solution based on a specified ordering of items. It uses a placement heuristic
    to determine the arrangement of items in the bins and calculates the fitness based on the number of bins used and
    the fill rate of the last bin to handle ties.

    Parameters:
    - items (np.ndarray): An array of structured arrays representing the items to be placed in the bins.
    - id_ordering (np.ndarray): An array of identifiers representing the order in which the items should be placed.
    - bin_dimentions (tuple): Tuple containing the width and height of the bin.
    - guillotine_cut (bool): Indicates whether guillotine cuts are used for placing the items.
    - rotation (bool): Indicates whether rotation of items is allowed during placement.

    Returns:
    - float: The calculated fitness value of the bin packing solution, based on the number of bins used and the fill rate of the last bin.
    """
    
    bin_width, bin_height = bin_dimensions
    
    # This gives an array of items with a specific ordering
    sequence = get_corresponding_sequence_by_id(items, id_ordering)
    
    # Apply the placement heuristic to get the bins
    solution = lgfi(sequence, bin_width, bin_height, guillotine_cut, rotation)
    
    # Compute the fitness of this specfic solution (bins)
    solution_fitness = len(solution) + (calculate_bin_fill(solution[-1]) / (bin_width*bin_height))
    
    return solution_fitness

@njit(parallel = True, cache = True)
def compute_fitnesses(population: np.ndarray, items: np.ndarray, bin_dimensions: Tuple[int, int], 
                      guillotine_cut: bool, rotation: bool) -> np.ndarray:
    """
    Calculate the fitnesses of a population of bin packing solutions.

    Parameters:
    - population (np.ndarray): An array representing the population of solutions.
    - items (np.ndarray): An array of items to be packed.
    - bin_dimentions (tuple): Tuple containing the width and height of the bin.
    - guillotine_cut (bool): Should the guillotine cut rule be applied.
    - rotation (bool): Should the items be able to rotate

    Returns:
    - np.ndarray: An array of fitness values for the population.
    """
    
    population_size = len(population)
    
    fitnesses = np.zeros(population_size)
    
    for i in prange(population_size):
        
        # Compute the fitness of this specfic solution (bins)
        fitnesses[i] = compute_fitness(items, population[i], bin_dimensions, guillotine_cut, rotation)
        
    return fitnesses