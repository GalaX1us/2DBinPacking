import numpy as np
from fitness import compute_fitnesses
from numba import njit, prange
from solutions_generation import custom_choice

def crossover(population: np.ndarray, bin_width:int, bin_height:int, crossover_rate: float, delta: float):
    """
    Perform crossover on a subset of the population P. Each solution in the
    subset is paired with another solution from P, and uniform order-based
    crossover is performed to generate a new solution.

    Parameters:
    - population (np.ndarray): Array of individual solutions, each a permutation of item indices.
    - bin_width (int): Width of the bin
    - bin_height (int): Height of the bin
    - crossover_rate (float): Proportion of the population to undergo crossover.
    - delta (float): Exponent used to adjust selection probabilities based on fitness ranking.

    Returns:
    - np.ndarray: New population subset generated through crossover.
    """
    
    psize = len(population)
    n = len(population[0])
    
    num_crossover = int(crossover_rate * psize)
    fitnesses = compute_fitnesses(population, bin_width, bin_height)
    
    sorted_indices = np.argsort(-fitnesses)
    selected_indices = sorted_indices[:num_crossover]
    
    ranks = np.argsort(sorted_indices)
    probabilities = (psize - ranks) ** delta
    probabilities /= probabilities.sum()
    
    new_population = np.empty((num_crossover, n), dtype=np.int32)
    
    for i in prange(num_crossover):
        idx = selected_indices[i]
        parent1 = population[idx]
        partner_idx = np.random.choice(psize, p=probabilities)
        while partner_idx == idx:
            partner_idx = np.random.choice(psize, p=probabilities)
        
        parent2 = population[partner_idx]
        new_population[i] = offspring_generation(parent1, parent2, fitnesses[idx], fitnesses[partner_idx])

    return new_population

@njit
def offspring_generation(parent1, parent2, fitness1, fitness2):
    """
    Perform a order-based crossover between two parent solutions to generate an offspring.
    This crossover method starts by aligning two parent solutions, checks for identical items at corresponding
    positions, and directly transfers matching items to the offspring. Non-matching items are probabilistically 
    chosen based on parent fitness, favoring the item from the "better" parent. This process ensures diversity 
    while maintaining some degree of inheritance from both parents.

    Parameters:
    - parent1 (np.ndarray): First parent permutation of item indices.
    - parent2 (np.ndarray): Second parent permutation of item indices.
    - fitness1 (float): Fitness score of the first parent.
    - fitness2 (float): Fitness score of the second parent.

    Returns:
    - np.ndarray: Offspring permutation of item indices.
    """
    
    n = len(parent1)
    offspring = np.full(n, -1, dtype=int)
    used_items = set()
    k = l = r = 0

    while r < n:
        if parent1[k] == parent2[l]:
            offspring[r] = parent1[k]
            used_items.add(parent1[k])
            
        else:
            choice = custom_choice([parent1[k], parent2[l]], p=[0.75, 0.25] if fitness1 < fitness2 else [0.25, 0.75])
            
            offspring[r] = choice
            used_items.add(choice)
            
        r += 1

        # Move pointers if they are pointing to already used items
        while k < n and parent1[k] in used_items:
            k += 1
        while l < n and parent2[l] in used_items:
            l += 1

    return offspring