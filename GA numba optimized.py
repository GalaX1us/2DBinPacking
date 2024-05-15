from collections import namedtuple
import random
import numpy as np
from numba import jit, njit, prange
import time

Item = np.dtype([
    ('width', np.int32), 
    ('height', np.int32), 
    ('rotated', np.bool_)
])

@jit(nopython=True)
def custom_choice(indices, probs):
    cumulative_probs = np.cumsum(probs)
    rnd = np.random.random() * cumulative_probs[-1]
    idx = 0
    while idx < len(cumulative_probs):
        if rnd < cumulative_probs[idx]:
            return indices[idx]
        idx += 1
    return indices[-1]  

@jit(nopython=True)
def remove_index(indices, chosen_index):
    new_indices = np.empty(indices.shape[0] - 1, dtype=indices.dtype)
    j = 0
    for i in range(indices.shape[0]):
        if indices[i] != chosen_index:
            new_indices[j] = indices[i]
            j += 1
    return new_indices

@njit
def generate_initial_population(items, psize, kappa):
    """
    Generate an initial population of solutions using a probabilistic method based on the 
    deterministic sequence of items sorted by non-increasing area.

    Each solution in the population is generated using a roulette wheel selection mechanism
    where the selection probability of each item is influenced by its position in the
    deterministic sequence, adjusted by the kappa parameter.

    Parameters:
    -----------
    items : np.ndarray
        An array of structured dtype Items. This structured array is used to compute the 
        area and sort items for the deterministic sequence.
    psize : int
        The size of the population to generate. This specifies the number of individual
        solutions in the population.
    kappa : int or float
        A parameter that influences the selection probability of each item. Higher values
        make the selection probability closer to the deterministic sequence order. The
        influence is calculated as (n - position)^kappa, where position is the index of an
        item in the sorted deterministic sequence.

    Returns:
    --------
    np.ndarray
        A numpy array containing the generated population. Each element of the array is
        an array itself, representing a single solution composed of selected items based
        on the described probabilistic mechanism.

    """
    
    # Sorting by non-increasing area
    areas = items['width'] * items['height']
    deterministic_sequence = items[np.argsort(-areas)]
    
    n = len(items)
    
    # Initialize the population
    population = np.empty((psize, n), dtype=Item)
    
    # Calculate vi for each item based on its position in the deterministic sequence
    vi = np.array([(n - pos)**kappa for pos in range(n)])
    
    for i in range(psize):
        
        # Prepare for roulette wheel selection
        available_indices = np.arange(n)
        
        item_idx = 0
        
        while available_indices.size > 0:
            # Calculate selection probabilities
            vi_available = vi[available_indices]
            probabilities = vi_available / vi_available.sum()
            
            # Select an item index from available_indices using the computed probabilities
            chosen_index = custom_choice(available_indices, probabilities)
            
            population[i, item_idx] = deterministic_sequence[chosen_index]
            
             # Remove the chosen index from available indices
            available_indices = remove_index(available_indices, chosen_index)
            # available_indices = available_indices[available_indices!=chosen_index]
            
            item_idx+=1
        
    return population

def crossover(population: np.ndarray, crossover_rate: float, delta: float):
    """
    Perform crossover on a subset of the population P. Each solution in the
    subset is paired with another solution from P, and uniform order-based
    crossover is performed to generate a new solution.

    Parameters:
    -----------
    P : list of np.ndarray
        Population of solutions, each solution is a permutation of item indices.
    crate : float
        Crossover rate determining the fraction of the population to be selected for crossover.
    delta : float
        Determines the bias towards selecting better solutions as crossover partners.

    Returns:
    --------
    list of np.ndarray
        New population after crossover has been applied.
    """
    psize = len(population)
    num_crossover = int(crossover_rate * psize)
    fitnesses = np.array([fitness(ind) for ind in population])
    
    # Sorting indices by fitness
    sorted_indices = np.argsort(fitnesses)
    
    # Selecting indices for crossover based on fitness
    selected_indices = sorted_indices[:num_crossover]

    # Compute selection probabilities for the entire population
    ranks = np.argsort(sorted_indices)  
    probabilities = (psize - ranks) ** delta
    probabilities /= probabilities.sum() 

    new_population = []

    for idx in selected_indices:
        parent1 = population[idx]
        # Select a partner with probability bias towards better fitness
        partner_idx = np.random.choice(psize, p=probabilities)
        
        # Ensure both idx are different
        while partner_idx==idx:
            partner_idx = np.random.choice(psize, p=probabilities)
        
        parent2 = population[partner_idx]

        # Generate offspring using uniform order-based crossover
        offspring = offspring_generation(parent1, parent2, fitnesses[idx], fitnesses[partner_idx])
        new_population.append(offspring)

    return np.array(new_population)

def offspring_generation(parent1, parent2, fitness1, fitness2):
    """
    Perform a detailed order-based crossover between two parent solutions to generate an offspring.
    This crossover method starts by aligning two parent solutions, checks for identical items at corresponding
    positions, and directly transfers matching items to the offspring. Non-matching items are probabilistically 
    chosen based on parent fitness, favoring the item from the "better" parent. This process ensures diversity 
    while maintaining some degree of inheritance from both parents.

    The method handles items already placed in the offspring to avoid duplicates, and dynamically adjusts pointers
    to fill all positions of the offspring based on parent contributions. If an offspring proves to be more fit or
    efficient than one of the parents (based on a fitness function and criteria such as load), it replaces the less 
    fit parent in the new population.

    Parameters:
    -----------
    parent1 : np.ndarray
        First parent solution, a permutation of item indices.
    parent2 : np.ndarray
        Second parent solution, a permutation of item indices.
    fitness1 : float
        Fitness score of the first parent; lower scores indicate better fitness.
    fitness2 : float
        Fitness score of the second parent; lower scores indicate better fitness.

    Returns:
    --------
    np.ndarray
        The offspring solution generated from the parents.
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
            choice = np.random.choice([parent1[k], parent2[l]], p=[0.75, 0.25] if fitness1 < fitness2 else [0.25, 0.75])
            
            offspring[r] = choice
            used_items.add(choice)
            
        r += 1

        # Move pointers if they are pointing to already used items
        while k < n and parent1[k] in used_items:
            k += 1
        while l < n and parent2[l] in used_items:
            l += 1

    return offspring
    

def genetic_algo(items, bin_dimensions, population_size, nb_generations, crossover_rate, kappa, delta):
    population = generate_initial_population(population_size, kappa)
    
    for i in range(nb_generations):
        new_population = crossover(population, crossover_rate, delta)

if __name__ == "__main__":
    items = np.array([
        (30, 40, False),
        (60, 70, False),
        (50, 50, False),
        (20, 80, False),
        ], dtype=Item)
    
    
    start = time.perf_counter()
    pop = generate_initial_population(items, 1, 1)
    print(time.perf_counter()-start)
    
    start = time.perf_counter()
    pop = generate_initial_population(items, 100, 1)
    print(time.perf_counter()-start)
    
    start = time.perf_counter()
    pop = generate_initial_population(items, 100, 1)
    print(time.perf_counter()-start)
    
    start = time.perf_counter()
    pop = generate_initial_population(items, 100, 1)
    print(time.perf_counter()-start)
    
