from collections import namedtuple
import numpy as np

Item = np.dtype([
    ('id', np.int32)
    ('width', np.int32), 
    ('height', np.int32), 
    ('rotated', np.bool_)
])


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
    
    n = items.shape[0]
    
    # Initialize the population
    population = []
    
    # Calculate vi for each item based on its position in the deterministic sequence
    vi = np.array([(n - pos)**kappa for pos in range(n)])
    
    for _ in range(psize):
        
        # Prepare for roulette wheel selection
        selected_items = []
        available_indices = list(range(n))
        
        while available_indices:
            # Calculate selection probabilities
            vi_available = vi[available_indices]
            probabilities = vi_available / vi_available.sum()
            
            # Select an item index from available_indices using the computed probabilities
            chosen_index = np.random.choice(available_indices, p=probabilities)
            
            # Add the chosen item to the solution
            selected_items.append(deterministic_sequence[chosen_index])
            
            # Remove the chosen index from available indices
            available_indices.remove(chosen_index)
        
        # Add the generated solution to the population
        population.append(np.array(selected_items))
    
    return np.array(population)

def crossover(population: np.ndarray, crossover_rate: float, delta: float):
    """
    Perform crossover on a subset of the population P. Each solution in the
    subset is paired with another solution from P, and uniform order-based
    crossover is performed to generate a new solution.

    Parameters:
    -----------
    population : list of np.ndarray
        Population of solutions, each solution is a permutation of item indices.
    crossover_rate : float
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
    new_population = []
    

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
    
    pop = generate_initial_population(items, 100, 1)
    print(pop)