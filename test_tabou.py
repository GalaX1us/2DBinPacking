from genetic_algo.structures import *
from genetic_algo.solutions_generation import *
from data_manager import load_items_from_file
from genetic_algo.fitness import compute_fitness
from random import randint



INPUT_DATA_DIRECTORY = "data"
OUTPUT_DATA_DIRECTORY = "solutions"

ITERATION_NUMBER = 100
TABU_LIST_SIZE = 10

KAPPA = 5 # Must be >= 1
DELTA = 5 # Must be >= 1

GUILLOTINE = True
ROTATION = True



# voisinages possible :
# - permutation
# - rotation
# - insertion (on met élément i à la place j et on décale tout le reste dans l'ordre)
#           1234567 -> 1723456  (7 en position 1)


def is_tabu_in_tabu_list_spe(tabu_list_spe, i, j, reversible):
    for k in range(len(tabu_list_spe)):
        if (tabu_list_spe[k][0] == i and tabu_list_spe[k][1] == j) \
        or (reversible==True and tabu_list_spe[k][0] == j and tabu_list_spe[k][1] == i):
            return True
    return False

# voisinage par permutation
def get_permutation_neighborhood(solution, tabu_list):
    tabu_list_permutation = []
    for i in range(len(tabu_list)):
        if tabu_list[i]['permutation'].all != -1:
            tabu_list_permutation.append(tabu_list[i]['permutation'])
            
    neighborhood_size = len(solution) * (len(solution) - 1) // 2
    neighborhood = np.empty(neighborhood_size, dtype=Neighbor)
    counter = 0
    
    for i in range(len(solution)):
        for j in range(i+1, len(solution)):
            if is_tabu_in_tabu_list_spe(tabu_list_permutation, i, j, True):
                continue
            
            new_solution = np.copy(solution)
            tampon = new_solution[i]
            new_solution[i] = new_solution[j]
            new_solution[j] = tampon
            
            neighborhood[counter]['solution'] = new_solution
            neighborhood[counter]['tabu']['permutation'] = (i, j)
            
            neighborhood[counter]['tabu']['rotation'] = -1
            neighborhood[counter]['tabu']['insertion'] = (-1, -1)
            counter += 1
    return neighborhood


# voisinage par rotation
def get_rotation_neighborhood(solution, tabu_list):
    tabu_list_rotation = []
    for i in range(len(tabu_list)):
        if tabu_list[i]['rotation'] != -1:
            tabu_list_rotation.append(tabu_list[i]['rotation'])
            
    neighborhood = np.empty(len(solution), dtype=Neighbor)
    
    for i in range(len(solution)):
        if i in tabu_list_rotation:
            continue
        
        new_solution = np.copy(solution)
        new_solution[i] = -new_solution[i]

        neighborhood[i]['solution'] = new_solution
        neighborhood[i]['tabu']['rotation'] = i
        
        neighborhood[i]['tabu']['permutation'] = (-1, -1)
        neighborhood[i]['tabu']['insertion'] = (-1, -1)
    return neighborhood


# voisinage par insertion
def get_insertion_neighborhood(solution, tabu_list):
    tabu_list_insertion = []
    for i in range(len(tabu_list)):
        if tabu_list[i]['insertion'].all != -1:
            tabu_list_insertion.append(tabu_list[i]['insertion'])
    
    
    neighborhood_size = (len(solution)-2) * (len(solution)-1)
    neighborhood = np.empty(neighborhood_size, dtype=Neighbor)
    counter = 0
    
    for i in range(len(solution)):
        for j in range(len(solution)):
            # On ne peut pas insérer un élément à la même place
            # On ne peut pas insérer un élément à la place juste avant ou juste après = permutation
            if i == j or i == j-1 or i == j+1:
                continue
            
            if is_tabu_in_tabu_list_spe(tabu_list_insertion, i, j, False):
                continue
            
            # insertion de l'élément i à la place j
            new_solution = np.copy(solution)
            tampon = new_solution[i]
    
            if i < j:
                for k in range(i, j):
                    new_solution[k] = new_solution[k+1]
            else:
                for k in range(i, j, -1):
                    new_solution[k] = new_solution[k-1]
                                        
            new_solution[j] = tampon
                        
            neighborhood[counter]['solution'] = new_solution
            neighborhood[counter]['tabu']['insertion'] = (i, j)
            
            neighborhood[counter]['tabu']['permutation'] = (-1, -1)
            neighborhood[counter]['tabu']['rotation'] = -1
            
            counter += 1
    return neighborhood



def get_neighborhood(solution, tabu_list):
    return np.concatenate((
        get_permutation_neighborhood(solution, tabu_list),
        get_rotation_neighborhood(solution, tabu_list),
        get_insertion_neighborhood(solution, tabu_list)))



def get_best_neighbor(neighborhood, items, bin_dimensions, guillotine_cut, rotation):
    best_neighbor = []
    best_fitness = np.inf
    
    for i in range(len(neighborhood)):
        fitness = compute_fitness(items, neighborhood[i]['solution'], bin_dimensions, guillotine_cut, rotation)
        if fitness < best_fitness:
            best_neighbor = [neighborhood[i]]
            best_fitness = fitness
        elif fitness == best_fitness:
            best_neighbor.append(neighborhood[i])

    return best_neighbor[randint(0, len(best_neighbor)-1)]



def id_tabu_empty(tabu):
    return tabu['permutation'][0] == -1 and \
            tabu['permutation'][1] == -1 and \
            tabu['insertion'][0] == -1 and \
            tabu['insertion'][1] == -1 and \
            tabu['rotation'] == -1


def add_tabu_list(tabu_list, tabu):
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




def tabu_search(items, bin_width, bin_height, GUILLOTINE, ROTATION) :
    # créer solution initiale
    solution = np.empty(1, dtype=Neighbor)[0]
    solution['solution'] = generate_population(items, 1, KAPPA)[0]
    best_solution = solution

    # calculer la fitness
    fitness = compute_fitness(items, solution['solution'], (bin_width, bin_height), GUILLOTINE, ROTATION)
    best_fitness = fitness
    
    # créer liste tabou vide
    tabu_list = np.empty(TABU_LIST_SIZE, dtype=Tabu)
    tabu_list.fill(([-1, -1], -1, [-1, -1]))
    
    
    for i in range(ITERATION_NUMBER):        
        # créer un voisinage
        neighborhood = get_neighborhood(solution['solution'], tabu_list)
        
        # trouver le meilleur voisin
        solution = get_best_neighbor(neighborhood, items, (bin_width, bin_height), GUILLOTINE, ROTATION)
        old_fitness = fitness
        fitness = compute_fitness(items, solution['solution'], (bin_width, bin_height), GUILLOTINE, ROTATION)
        
        # mettre à jour la liste tabou
        if fitness >= old_fitness:
            tabu_list = add_tabu_list(tabu_list, solution['tabu'])        
        
        # mettre à jour la meilleure solution
        elif fitness < best_fitness:
            best_fitness = fitness
            best_solution = solution['solution']
            print(i,"- best_solution :",fitness,"   ",best_solution)
    
    
    return best_solution, best_fitness




bin_width, bin_height, items = load_items_from_file(INPUT_DATA_DIRECTORY + "/binpacking2d-01.bp2d")


Tabu = np.dtype([
    ('permutation', np.int32, (2,)), #tuple avec les deux indices permutés pour arriver à cette solution, -1 si pas de permutation
    ('rotation',  np.int32), # indices de l'élément à tourner, -1 si pas de rotation
    ('insertion', np.int32, (2,)) # tuple avec les deux indices pour l'insertion (i,j) l'élément i inseré à l'indice j, -1 si pas d'insertion
])

Neighbor = np.dtype([
    ('solution', np.int32, (len(items),)),
    ('tabu', Tabu)
])


result = tabu_search(items, bin_width, bin_height, GUILLOTINE, ROTATION)
print("-----------------")
print(result)

