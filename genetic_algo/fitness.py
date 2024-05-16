import numpy as np

def calculate_bin_fill(bin):
    """
    Calculate the total fill of a bin based on the items placed in it.
    
    Parameters:
    - bin (np.ndarray): A bin structured array with items and their placements.
    
    Returns:
    - int: Total filled area of the bin.
    """
    total_fill = 0
    for i in range(len(bin['items'])):
        item = bin['items'][i]
        
        if item['id'] == -1 or item['width'] == 0:
            break
        
        total_fill += item['width'] * item['height']
        
    return total_fill

def compute_fitness(bins, k=2):
    """
    Calculate the fitness of a bin packing solution.

    Parameters:
    - bins (np.ndarray): Array of bin structured arrays.
    - capacity (int): The maximum capacity (area) of each bin.
    - k (float): Exponent to control the preference for more filled bins.

    Returns:
    - float: The calculated fitness value of the bin packing solution.
    """
    n = len(bins)
    if n == 0:
        return 0
    
    fills = np.array([calculate_bin_fill(bin) / ( bin['width']*bin['height']) for bin in bins])
    fitness_value = np.sum(fills**k) / n
    
    return fitness_value