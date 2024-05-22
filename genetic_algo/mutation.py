import numpy as np
from numba import njit, prange
from genetic_algo.structures import *

@njit(void(int32[:]), cache=True)
def rotate_individual(individual: np.ndarray) -> None:
    """
    Perform mutation on an individual by rotating one element.

    Parameters:
    - individual (np.ndarray): An individual solution.
    """
    n = len(individual)

    idx1 = np.random.choice(n, 1)

    # Rotate the corresponding individual 
    individual[idx1] = -individual[idx1]
    
@njit(void(int32[:]), cache=True)
def swap_individual(individual: np.ndarray) -> None:
    """
    Perform mutation on an individual by swapping two elements.

    Parameters:
    - individual (np.ndarray): An individual solution.
    """
    n = len(individual)
    if n <= 1:
        return  # No mutation possible for individual of length 1 or less

    # Choose two distinct indices to swap
    idx1, idx2 = np.random.choice(n, 2, replace=False)

    # Swap the elements
    individual[idx2], individual[idx1] = individual[idx1], individual[idx2]

@njit(int32[:, :](int32[:, :], float64), parallel = True, cache=True)
def mutation(population: np.ndarray, mutation_rate: float) -> np.ndarray:
    """
    Perform mutation on a population by swapping elements in each individual.

    Parameters:
    - population (np.ndarray): Array of individual solutions.
    - mutation_rate (float): Probability of mutation for each individual.

    Returns:
    - np.ndarray: Mutated population.
    """
    mutated_population = population.copy()
    population_size, individual_length = population.shape
    
    for i in prange(population_size):
        if np.random.random() < mutation_rate:
            
            # Perform mutation on this individual
            rotate_individual(mutated_population[i])

    return mutated_population
    

    
