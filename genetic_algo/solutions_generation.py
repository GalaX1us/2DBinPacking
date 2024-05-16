import numpy as np
from structures import Item
from numba import njit

@njit
def custom_choice(indices, p):
    cumulative_probs = np.cumsum(p)
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

@njit
def generate_population(items, psize, kappa):
    """
    Generate a population of solutions using a probabilistic method based on the 
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
        an array itself, representing a single solution composed of selected items indices based
        on the described probabilistic mechanism.

    """
    
    # Sorting by non-increasing area
    areas = items['width'] * items['height']
    deterministic_sequence = items[np.argsort(-areas)]
    
    n = len(items)
    
    # Initialize the population
    population = np.empty((psize, n), dtype=np.int32)
    
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
            chosen_index = custom_choice(available_indices, p=probabilities)
            
            population[i, item_idx] = deterministic_sequence[chosen_index]['id']
            
            available_indices = available_indices[available_indices!=chosen_index]
            # available_indices = remove_index(available_indices, chosen_index)
            
            item_idx+=1
        
    return population

@njit
def get_corresponding_sequence_by_id(items, id_ordering):
    """
    Create an array of items based on the provided ordering of item IDs.

    Parameters:
    - items (np.ndarray): Array of items to be reordered.
    - id_ordering (list or np.ndarray): Sequence of item IDs representing the desired order of items.

    Returns:
    - np.ndarray: An array of items reordered according to the specified ID ordering.
    """
    # Create a copy of the items array to avoid modifying the original
    items_copy = items.copy()

    # Create a mapping from item IDs to their indices
    id_to_index = {item['id']: idx for idx, item in enumerate(items_copy)}

    # Translate id_ordering into indices using the mapping
    indices = [id_to_index[id_] for id_ in id_ordering if id_ in id_to_index]

    # Use the indices to reorder the copied items
    ordered_items = items_copy[np.array(indices)]
    return ordered_items