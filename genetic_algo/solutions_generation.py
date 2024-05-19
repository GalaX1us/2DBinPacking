from genetic_algo.structures import *

@njit(cache = True)
def custom_choice(indices: np.ndarray, p: np.ndarray) -> int:
    """
    Perform a probabilistic selection from a list of indices based on provided probabilities.

    Parameters:
    - indices (np.ndarray): Array of indices to choose from.
    - p (np.ndarray): Array of probabilities corresponding to each index.

    Returns:
    - int: Selected index based on the provided probabilities.
    """
    
    cumulative_probs = np.cumsum(p)
    rnd = np.random.random() * cumulative_probs[-1]
    idx = 0
    while idx < len(cumulative_probs):
        if rnd < cumulative_probs[idx]:
            return indices[idx]
        idx += 1
    return indices[-1]  

@njit(cache = True)
def remove_index(indices: np.ndarray, chosen_index: int) -> np.ndarray:
    """
    Remove a chosen index from an array of indices.

    Parameters:
    - indices (np.ndarray): Array of indices.
    - chosen_index (int): The index to be removed.

    Returns:
    - np.ndarray: New array with the chosen index removed.
    """
    new_indices = np.empty(len(indices) - 1, dtype=indices.dtype)
    j = 0
    for i in range(indices.shape[0]):
        if indices[i] != chosen_index:
            new_indices[j] = indices[i]
            j += 1
    return new_indices

@njit(cache = True)
def generate_population(items: np.ndarray, psize: int, kappa: np.float32) -> np.ndarray:
    """
    Generate a population of solutions using a probabilistic method based on the 
    deterministic sequence of items sorted by non-increasing area.

    Each solution in the population is generated using a roulette wheel selection mechanism
    where the selection probability of each item is influenced by its position in the
    deterministic sequence, adjusted by the kappa parameter.

    Parameters:
    - items (np.ndarray): Array of structured dtype Items. This structured array is used to compute the 
                          area and sort items for the deterministic sequence.
    - psize (int): The size of the population to generate. This specifies the number of individual
                   solutions in the population.
    - kappa (np.float32): A parameter that influences the selection probability of each item. Higher values
                            make the selection probability closer to the deterministic sequence order. The
                            influence is calculated as (n - position)^kappa, where position is the index of an
                            item in the sorted deterministic sequence.

    Returns:
    - np.ndarray: A numpy array containing the generated population. Each element of the array is
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
            
            # if np.random.random() < 0.5:
            #     population[i, item_idx] = -population[i, item_idx]
            
            available_indices = available_indices[available_indices!=chosen_index]
            
            item_idx+=1
        
    return population

@njit(cache = True)
def get_corresponding_sequence_by_id(items: np.ndarray, id_ordering: np.ndarray) -> np.ndarray:
    """
    Create an array of items based on the provided ordering of item IDs.

    Parameters:
    - items (np.ndarray): Array of items to be reordered.
    - id_ordering (np.ndarray): Sequence of item IDs representing the desired order of items.

    Returns:
    - np.ndarray: An array of items reordered according to the specified ID ordering.
    """
    
    # Initialize an empty array for the ordered items with the same type as items
    ordered_items = np.empty(len(items), dtype=Item)

    # Create a mapping from item IDs to their indices in the original items array
    id_to_index = {item['id']: i for i, item in enumerate(items)}

    # Fill the ordered_items array by mapping each id in id_ordering to the corresponding item
    for idx, item_id in enumerate(id_ordering):
        if item_id in id_to_index:
            ordered_items[idx] = items[id_to_index[item_id]]

    return ordered_items