from binpacking.structures import *

@njit(int32(int32[:], float64[:]), cache = True)
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

@njit(int32[:, :](from_dtype(Item)[:], int32, float64), cache = True)
def generate_population(items: np.ndarray, psize: int, kappa: np.float64) -> np.ndarray:
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
    - kappa (np.float64): A parameter that influences the selection probability of each item. Higher values
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
    vi = np.array([np.float64(n - pos)**kappa for pos in range(n)], dtype=np.float64)
    
    for i in range(psize):
        
        # Prepare for roulette wheel selection
        available_indices = np.arange(n, dtype=np.int32)
        
        item_idx = 0
        
        while available_indices.size > 0:
            # Calculate selection probabilities
            vi_available = vi[available_indices]
            probabilities = vi_available / vi_available.sum()
            probabilities = probabilities.astype(np.float64)
            
            # Select an item index from available_indices using the computed probabilities
            chosen_index = custom_choice(available_indices, p=probabilities)
            
            population[i, item_idx] = deterministic_sequence[chosen_index]['id']
            
            # if np.random.random() < 0.5:
            #     population[i, item_idx] = -population[i, item_idx]
            
            available_indices = available_indices[available_indices!=chosen_index]
            
            item_idx+=1
        
    return population

@njit(from_dtype(Item)[:](from_dtype(Item)[:], int32[:]), cache = True)
def get_corresponding_sequence_by_id(items: np.ndarray, id_ordering: np.ndarray) -> np.ndarray:
    """
    Create an array of items based on the provided ordering of item IDs.
    
    Absolute are there to handle the representation of a rotated Item. 
    An item needs to be rotated if it's index in the population is negative.

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
        if abs(item_id) in id_to_index:
            ordered_items[idx] = items[id_to_index[abs(item_id)]]
            
            # If the ID is negative the item should be rotated
            if item_id < 0:
                ordered_items[idx]['width'], ordered_items[idx]['height'] = ordered_items[idx]['height'], ordered_items[idx]['width']
                ordered_items[idx]['rotated'] = not ordered_items[idx]['rotated']

    return ordered_items