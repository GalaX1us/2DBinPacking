from typing import Tuple
from genetic_algo.fitness import compute_fitness
from genetic_algo.structures import *
from numba import njit, prange
from genetic_algo.solutions_generation import custom_choice

@njit(int32[:](int32[:], int32[:], float64, float64), cache = True)
def offspring_generation(parent1: np.ndarray, parent2: np.ndarray, fitness1: float, fitness2: float) -> np.ndarray:
    """
    Perform a order-based crossover between two parent solutions to generate an offspring.
    This crossover method starts by aligning two parent solutions, checks for identical items at corresponding
    positions, and directly transfers matching items to the offspring. Non-matching items are probabilistically 
    chosen based on parent fitness, favoring the item from the "better" parent. This process ensures diversity 
    while maintaining some degree of inheritance from both parents.
    
    Absolute are there to handle the representation of a rotated Item. 
    An item needs to be rotated if it's index in the population is negative.

    Parameters:
    - parent1 (np.ndarray): First parent permutation of item indices.
    - parent2 (np.ndarray): Second parent permutation of item indices.
    - fitness1 (float): Fitness score of the first parent.
    - fitness2 (float): Fitness score of the second parent.

    Returns:
    - np.ndarray: Offspring permutation of item indices.
    """
    
    n = len(parent1)
    offspring = np.full(n, -1, dtype=np.int32)
    used_items = set()
    k = l = r = 0

    while r < n:
        if parent1[k] == parent2[l]:
            offspring[r] = parent1[k]
            used_items.add(abs(parent1[k]))
            
        else:
            prob = np.array([0.75, 0.25] if fitness1 < fitness2 else [0.25, 0.75], dtype=np.float64)
            choice = custom_choice(np.array([parent1[k], parent2[l]], dtype=np.int32), p=prob)
            
            offspring[r] = choice
            used_items.add(abs(choice))
            
        r += 1

        # Move pointers if they are pointing to already used items
        while k < n and abs(parent1[k]) in used_items:
            k += 1
        while l < n and abs(parent2[l]) in used_items:
            l += 1

    return offspring

@njit(int32[:, :](int32[:, :], float64[:], float64, float64), parallel = True, cache = True)
def crossover(population: np.ndarray, fitnesses: np.ndarray, crossover_rate: float, delta: float) -> np.ndarray:
    """
    Perform crossover on a subset of the population P. Each solution in the
    subset is paired with another solution from P, and uniform order-based
    crossover is performed to generate a new solution.

    Parameters:
    - population (np.ndarray): Array of individual solutions, each a permutation of item indices.
    - fitnesses (np.ndarray): Array of individual fitnesses.
    - crossover_rate (float): Proportion of the population to undergo crossover.
    - delta (float): Exponent used to adjust selection probabilities based on fitness ranking.

    Returns:
    - np.ndarray: New population subset generated through crossover.
    """
    
    psize, n = population.shape
    num_crossover = int(crossover_rate * psize)
    
    # Get the indices of the sorted fitnesses (lower is the best)
    sorted_indices = np.argsort(fitnesses)
    
    # Probabilities to do roulette wheel selection
    probabilities = ((psize - np.arange(num_crossover)).astype(np.float64)) ** delta
    probabilities /= probabilities.sum()
    probabilities = probabilities.astype(np.float64)
    
    new_population = np.empty((num_crossover, n), dtype=np.int32)
    
    pop_idx_array = np.arange(psize, dtype=np.int32)
    
    for i in prange(num_crossover):
        
        idx = sorted_indices[i]
        
        parent1 = population[idx]
        
        # Select parent by roulette wheel selection
        sorted_idx = custom_choice(pop_idx_array, p=probabilities)
        partner_idx = sorted_indices[sorted_idx]
        
        while partner_idx == idx:
            sorted_idx = custom_choice(pop_idx_array, p=probabilities)
            partner_idx = sorted_indices[sorted_idx]
        
        parent2 = population[partner_idx]
        
        offspring = offspring_generation(parent1, parent2, fitnesses[idx], fitnesses[partner_idx])
        
        new_population[i] = offspring
        
    return new_population