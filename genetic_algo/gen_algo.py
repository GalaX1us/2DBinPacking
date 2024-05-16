import numpy as np
from crossover import crossover
from solutions_generation import generate_population
from structures import Item, create_item


def genetic_algo(items, bin_dimensions, population_size, nb_generations, crossover_rate, kappa, delta):
    
    population = generate_population(items, population_size, kappa)
    
    # Unpack the bin dimensions
    bin_width, bin_height = bin_dimensions
    
    for i in range(nb_generations):
        
        crossover_population = crossover(population, bin_width, bin_height, crossover_rate, delta)
        remaining_population = generate_population(items, population_size - crossover_population.shape[0] ,kappa)
        population = np.concatenate([crossover_population, remaining_population])

if __name__ == "__main__":
    import time
    items = np.array([
        create_item(0, 10, 20),
        create_item(1, 30, 40),
        create_item(2, 50, 60, False),
        create_item(3, 70, 80, False),
        ], dtype=Item)
    
    pop = generate_population(items, 10, 0.5)
    
    print(pop)
    