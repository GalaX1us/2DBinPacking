import faulthandler
import os
from binpacking.data_manager import load_items_from_file
from binpacking.structures import Tabu, Neighbor
from binpacking.population_generation import *
from binpacking.fitness import compute_fitness, compute_fitnesses
from numba import njit

faulthandler.enable()

MIN_INT32 = np.int32(-2147483647)

@njit(cache = True)
def  create_tabu_list(size):
    # Initialize the tabu_list with the specified size
    tabu_list = np.empty(size, dtype=Tabu)
    
    # Fill the tabu_list with default values
    for i in range(size):
        tabu_list[i]['permutation'][0] = MIN_INT32
        tabu_list[i]['permutation'][1] = MIN_INT32
        tabu_list[i]['rotation'] = MIN_INT32
        tabu_list[i]['insertion'] = MIN_INT32
    
    return tabu_list


@njit(cache = True)
def get_permutation_neighborhood(solution, tabu_list):
    """
    Generate the permutation neighborhood for a given solution.

    Args:
        solution (np.ndarray): The current solution.
        tabu_list (np.ndarray): The current tabu list.

    Returns:
        np.ndarray: The permutation neighborhood.
    """
    tabu_list_permutation = tabu_list['permutation'][tabu_list['permutation'][:, 0] != MIN_INT32]
    
    len_solution = len(solution)
    neighborhood_size = len_solution -1
    neighborhood = np.zeros(neighborhood_size, dtype=Neighbor)
    
    mask = np.zeros(neighborhood_size, dtype=np.bool_)
    
    for i in range(neighborhood_size):
        if np.any((tabu_list_permutation[:, 0] == i) & (tabu_list_permutation[:, 1] == i+1)):
            continue
        
        new_solution = np.copy(solution)
        new_solution[i], new_solution[i+1] = new_solution[i+1], new_solution[i]
        
        neighborhood[i]['solution'][:len_solution] = new_solution
        neighborhood[i]['solution'][len_solution:] = MIN_INT32
        neighborhood[i]['tabu']['permutation'][0] = i
        neighborhood[i]['tabu']['permutation'][1] = i+1
        
        neighborhood[i]['tabu']['rotation'] = MIN_INT32
        neighborhood[i]['tabu']['insertion'] = MIN_INT32
        mask[i] = True
        
    valid_neighborhood = neighborhood[mask]
    return valid_neighborhood

@njit(cache = True)
def get_rotation_neighborhood(solution, tabu_list):
    """
    Generate the rotation neighborhood for a given solution.

    Args:
        solution (np.ndarray): The current solution.
        tabu_list (np.ndarray): The current tabu list.

    Returns:
        np.ndarray: The rotation neighborhood.
    """
    tabu_list_rotation = tabu_list['rotation'][tabu_list['rotation'] != MIN_INT32]
            
    neighborhood = np.empty(len(solution), dtype=Neighbor)
    len_solution = len(solution)
    counter = 0
    
    for i in range(len(solution)):
        if i in tabu_list_rotation:
            continue
        
        new_solution = np.copy(solution)
        new_solution[i] = -new_solution[i]

        neighborhood[counter]['solution'][:len_solution] = new_solution
        neighborhood[counter]['solution'][len_solution:] = MIN_INT32
        neighborhood[counter]['tabu']['rotation'] = i
        
        neighborhood[counter]['tabu']['permutation'][0] = MIN_INT32
        neighborhood[counter]['tabu']['permutation'][1] = MIN_INT32
        neighborhood[counter]['tabu']['insertion']= MIN_INT32
        
        counter += 1
    return neighborhood[:counter]

@njit(cache = True)
def get_insertion_neighborhood(solution, tabu_list):
    """
    Generate the insertion neighborhood for a given solution.

    Args:
        solution (np.ndarray): The current solution.
        tabu_list (np.ndarray): The current tabu list.

    Returns:
        np.ndarray: The insertion neighborhood.
    """
    tabu_list_insertion = tabu_list['insertion'][tabu_list['insertion'] != MIN_INT32]
    
    len_solution = len(solution)
    neighborhood_size = len_solution-1
    neighborhood = np.zeros(neighborhood_size, dtype=Neighbor)
    counter = 0
    
    for i in range(1, len_solution):
        if i in tabu_list_insertion:
            continue
        
        # Insert ith element at jth position
        new_solution = solution.copy()
        new_solution[1:i+1] = solution[0:i]  # Shift elements from j to i to the right
        new_solution[0] = solution[i]  # Insert element i at position j
                    
        neighborhood[counter]['solution'][:len_solution] = new_solution
        neighborhood[counter]['solution'][len_solution:] = MIN_INT32
        neighborhood[counter]['tabu']['insertion'] = i
        
        neighborhood[counter]['tabu']['permutation'][0] = MIN_INT32
        neighborhood[counter]['tabu']['permutation'][1] = MIN_INT32
        neighborhood[counter]['tabu']['rotation'] = MIN_INT32
        
        counter += 1
    return neighborhood[:counter]


@njit(cache = True)
def get_neighborhood(solution, tabu_list):
    """
    Generate the complete neighborhood for a given solution.

    Args:
        solution (np.ndarray): The current solution.
        tabu_list (np.ndarray): The current tabu list.
    Returns:
        np.ndarray: The complete neighborhood, combining permutation, rotation, and insertion neighborhoods.
    """
    
    # permutation_neighborhood = get_permutation_neighborhood(solution, tabu_list)
    rotation_neighborhood = get_rotation_neighborhood(solution, tabu_list)
    insertion_neighborhood = get_insertion_neighborhood(solution, tabu_list)
    permutation_neighborhood = get_permutation_neighborhood(solution, tabu_list)
    
    
    return np.concatenate((permutation_neighborhood, rotation_neighborhood, insertion_neighborhood))

@njit(cache = True)
def get_best_neighbor(neighborhood, items, bin_dimensions, guillotine_cut, rotation):
    """
    Find the best neighbor in the neighborhood based on fitness.

    Args:
        neighborhood (np.ndarray): The neighborhood of solutions.
        items (list): The list of items.
        bin_dimensions (tuple): Dimensions of the bin (width, height).
        guillotine_cut (bool): Whether guillotine cut is allowed.
        rotation (bool): Whether rotation is allowed.

    Returns:
        np.ndarray: The best neighbor solution.
    """
    len_solution = len(neighborhood[0]['solution'][neighborhood[0]['solution'] != MIN_INT32])
    solutions = neighborhood['solution']
    solutions = solutions[:, :len_solution]
    
    # Make the array contiguous
    solutions_fixed = np.zeros((len(neighborhood), len_solution), dtype=np.int32)
    solutions_fixed[:, :] = solutions
    
    # Compute fitnesses for all neighbors
    fitnesses = compute_fitnesses(solutions_fixed, items, bin_dimensions, guillotine_cut, rotation)
    
    # Find the best fitness value
    best_fitness = np.min(fitnesses)
    
    # Find all indices of neighbors with the best fitness
    best_indices = np.where(fitnesses == best_fitness)[0]
    # Randomly select one of these indices
    random_index = np.random.choice(best_indices)
    
    # Return the corresponding neighbor
    return neighborhood[random_index]



@njit(cache = True)
def id_tabu_empty(tabu):
    """
    Check if a tabu entry is empty.

    Args:
        tabu (np.ndarray): The tabu entry.

    Returns:
        bool: True if the tabu entry is empty, False otherwise.
    """
    return tabu['permutation'][0] == MIN_INT32 and \
            tabu['permutation'][1] == MIN_INT32 and \
            tabu['insertion'] == MIN_INT32 and \
            tabu['rotation'] == MIN_INT32

@njit(cache = True)
def add_tabu_list(tabu_list, tabu):
    """
    Add a new tabu move to the tabu list.

    Args:
        tabu_list (np.ndarray): The current tabu list.
        tabu (np.ndarray): The new tabu move.

    Returns:
        np.ndarray: The updated tabu list.
    """
    # If the tabu list isn't full, we add the permutation at the first empty spot
    for i in range(len(tabu_list)):
        if id_tabu_empty(tabu_list[i]):
            tabu_list[i] = tabu
            return tabu_list

    # If the tabu list is full, we remove the oldest element and add the new one at the end
    for i in range(len(tabu_list)-1):
        tabu_list[i] = tabu_list[i+1]
    tabu_list[len(tabu_list)-1] = tabu
    
    return tabu_list

def tabu_search(items, bin_dimensions, iteration_number, tabu_list_size, kappa, guillotine_cut, rotation) :
    """
    Perform tabu search for the bin packing problem.

    Args:
        items (list): The list of items.
        bin_dimensions (tuple): Dimensions of the bin (width, height).
        iteration_number (int): Number of iterations.
        tabu_list_size (int): Size of the tabu list.
        kappa (int): Parameter for solution generation.
        guillotine_cut (bool): Whether guillotine cut is allowed.
        rotation (bool): Whether rotation is allowed.

    Returns:
        tuple: Best solution and its fitness value.
    """
    
    assert tabu_list_size < 3*len(items), "Tabu list size must be lower than 3 x number of items"
    
    bin_width, bin_height = bin_dimensions
    
    # Create initial solution
    solution = np.zeros(1, dtype=Neighbor)[0]
    solution['tabu']['insertion'] = MIN_INT32
    solution['tabu']['permutation'][0] = MIN_INT32
    solution['tabu']['permutation'][1] = MIN_INT32
    solution['tabu']['rotation'] = MIN_INT32
    
    initial_solution = generate_population(items, 1, kappa)[0]
    len_solution = len(initial_solution)
    solution['solution'][:len_solution] = initial_solution
    solution['solution'][len_solution:] = MIN_INT32  # Mark unused spots as min value int32
    best_solution = np.zeros(len_solution, dtype=np.int32)
    best_solution[:] = initial_solution
    
    solution_fixed = np.zeros(len_solution, dtype=np.int32)
    solution_fixed[:] = solution['solution'][:len_solution]

    # Compute fitness
    fitness = compute_fitness(items, solution['solution'][:len_solution], (bin_width, bin_height), guillotine_cut, rotation)
    best_fitness = fitness
    # Create empty tabu list
    tabu_list = create_tabu_list(tabu_list_size)
    
    for i in range(iteration_number):
        # Create neighborhood
        neighborhood = get_neighborhood(solution['solution'][:len_solution], tabu_list)
        # Find best neighbor
        solution = get_best_neighbor(neighborhood, items, (bin_width, bin_height), guillotine_cut, rotation)
        old_fitness = fitness
        fitness = compute_fitness(items, solution['solution'][:len_solution], (bin_width, bin_height), guillotine_cut, rotation)
        
        # Update tabu list
        if fitness >= old_fitness:
            tabu_list = add_tabu_list(tabu_list, solution['tabu'])
        
        # Update best solution
        elif fitness < best_fitness:
            best_fitness = fitness
            best_solution[:] = solution['solution'][:len_solution]
    
    return best_solution, best_fitness