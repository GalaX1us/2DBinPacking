import numpy as np
from structs import Item
from numba import njit, jit

def custom_choice(indices, probs):
    cumulative_probs = np.cumsum(probs)
    rnd = np.random.random() * cumulative_probs[-1]
    idx = 0
    while idx < len(cumulative_probs):
        if rnd < cumulative_probs[idx]:
            return indices[idx]
        idx += 1
    return indices[-1]  

@njit
def remove_index(indices, chosen_index):
    new_indices = np.empty(indices.shape[0] - 1, dtype=indices.dtype)
    j = 0
    for i in range(indices.shape[0]):
        if indices[i] != chosen_index:
            new_indices[j] = indices[i]
            j += 1
    return new_indices

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
            chosen_index = np.random.choice(available_indices, p=probabilities)
            
            population[i, item_idx] = deterministic_sequence[chosen_index]
            
            available_indices = available_indices[available_indices!=chosen_index]
            # available_indices = remove_index(available_indices, chosen_index)
            
            item_idx+=1
        
    return population