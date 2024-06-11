from genetic_algo.structures import *
from genetic_algo.solutions_generation import *
from data_manager import load_items_from_file
from genetic_algo.fitness import compute_fitness



# def un nombre d'itération
# créer solution initiale

# créer un voisinage
# calculer la fitness
# définir la liste tabou (taille fixe)
# stoquer les transformations




INPUT_DATA_DIRECTORY = "data"
OUTPUT_DATA_DIRECTORY = "solutions"

ITERATION_NUMBER = 100

KAPPA = 5 # Must be >= 1
DELTA = 5 # Must be >= 1

GUILLOTINE = True
ROTATION = True



bin_width, bin_height, items = load_items_from_file(INPUT_DATA_DIRECTORY + "/binpacking2d-01.bp2d")


print("item :\n",items)


# créer solution initiale
initial_solution = generate_population(items, 1, KAPPA)[0]
print("initial_solution :\n",initial_solution)


# calculer la fitness
fitness = compute_fitness(items, initial_solution, (bin_width, bin_height), GUILLOTINE, ROTATION)
print("fitness :\n",fitness)




# voisinages possible :
# - permutation
# - rotation



Neighbor = np.dtype([
    ('solution', np.int32, (10,)), 
    ('permutation', np.int32, (2,)), #tuple avec les deux indices permutés pour arriver à cette solution (si les 2 sont égaux, alors c'est une rotation)
])


# voisinage par permutation
def get_permutation_neighborhood(solution):
    neighborhood_size = len(solution) * (len(solution) - 1) // 2
    neighborhood = np.empty(neighborhood_size, dtype=Neighbor)
    counter = 0
    
    for i in range(len(solution)):
        for j in range(i+1, len(solution)):
            new_solution = np.copy(solution)
            tampon = new_solution[i]
            new_solution[i] = new_solution[j]
            new_solution[j] = tampon
            
            neighborhood[counter]['solution'] = new_solution
            neighborhood[counter]['permutation'] = (i, j)
            counter += 1
    return neighborhood
  

# voisinage par rotation
def get_rotation_neighborhood(solution):
    neighborhood = np.empty(len(solution), dtype=Neighbor)
    
    for i in range(len(solution)):
        new_solution = np.copy(solution)
        new_solution[i] = -new_solution[i]

        neighborhood[i]['solution'] = new_solution
        neighborhood[i]['permutation'] = (i, i)
    return neighborhood


neighborhood = get_permutation_neighborhood(initial_solution)
# TODO : ajouter le voisinage par rotation
#neighborhood += get_rotation_neighborhood(initial_solution)
#print("neighborhood :\n",neighborhood)


# tabu_list = np.int32, (2,) #tuple avec les deux indices permutés 
def get_best_neighbor(neighborhood, tabu_list, items, bin_dimensions, guillotine_cut, rotation):
    best_neighbor = None
    best_fitness = - np.inf
    
    for i in range(len(neighborhood)):
        excluded = False
        for j in range(len(tabu_list)):
            if neighborhood[i]['permutation'][0] == tabu_list[j][0] and neighborhood[i]['permutation'][1] == tabu_list[j][1]:
                excluded = True
                break
        if excluded:
            continue
        
        fitness = compute_fitness(items, neighborhood[i]['solution'], bin_dimensions, guillotine_cut, rotation)
        if fitness < best_fitness:
            best_fitness = fitness
            best_neighbor = neighborhood[i]

    return best_neighbor


best_neighbor = get_best_neighbor(neighborhood, [], items, (bin_width, bin_height), GUILLOTINE, ROTATION)
print("best_neighbor :\n",best_neighbor)
print("fitness :\n",compute_fitness(items, best_neighbor['solution'], (bin_width, bin_height), GUILLOTINE, ROTATION))   


def add_tabu_list(tabu_list, permutation):
    # If the tabu list isn't full, we add the permutation at the first empty spot
    for i in range(len(tabu_list)):
        # TODO : test de sécu à remove quand tout marche
        if tabu_list[i][0] == permutation[0] and tabu_list[i][1] == permutation[1]:
            print("ATTEION !PROBLÈME !!!! permutation déjà dans la liste tabou")
        
        
        if tabu_list[i][0] == -1 and tabu_list[i][1] == -1:
            tabu_list[i] = permutation
            return tabu_list

    # If the tabu list is full, we remove the oldest element and add the new one at the end
    for i in range(len(tabu_list)-1):
        tabu_list[i] = tabu_list[i+1]
    tabu_list[len(tabu_list)-1] = permutation
    
    return tabu_list


def tabu_test() :
    fitness_best_neighbor = compute_fitness(items, best_neighbor['solution'], (bin_width, bin_height), GUILLOTINE, ROTATION)

    old_fitness = fitness
    fitness = fitness_best_neighbor
    solution = best_neighbor['solution']

    if fitness_best_neighbor < old_fitness:
        tabu_list = add_tabu_list(tabu_list, best_neighbor['permutation'])
    elif fitness > best_fitness:
        best_fitness = fitness
        best_solution = solution

    