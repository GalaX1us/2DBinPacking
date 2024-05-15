import numpy as np
from solutions_generation import generate_initial_population
from genetic_algo.structures import Item


# def genetic_algo(items, bin_dimensions, population_size, nb_generations, crossover_rate, kappa, delta):
#     population = generate_initial_population(population_size, kappa)
    
#     for i in range(nb_generations):
#         new_population = crossover(population, crossover_rate, delta)

if __name__ == "__main__":
    import time
    items = np.array([
        (0, 10, 20, False),
        (1, 30, 40, False),
        (2, 50, 60, False),
        (3, 70, 80, False),
        ], dtype=Item)
    
    pop = generate_initial_population(items, 10, 1)
    
    print(pop)
    