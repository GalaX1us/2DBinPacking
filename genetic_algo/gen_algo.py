from genetic_algo.solutions_generation import *
from genetic_algo.crossover import crossover
from genetic_algo.structures import *
from genetic_algo.fitness import compute_fitnesses
from genetic_algo.lgfi import lgfi

def genetic_algo(items, bin_dimensions, population_size, nb_generations, crossover_rate, kappa, delta):
    
    # Unpack the bin dimensions
    bin_width, bin_height = bin_dimensions
    
    population = generate_population(items, population_size, kappa)
    
    best_solution = np.zeros_like(population[0])
    best_fitness = 0
    
    
    for i in range(nb_generations):
        
        fitnesses = compute_fitnesses(population, items, bin_width, bin_height)
        
        # Store best generation
        best_index = np.argmax(fitnesses)
        current_best_fitness = fitnesses[best_index]
        if current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
            best_solution[:] = population[best_index]
        
        crossover_population = crossover(population, fitnesses, crossover_rate, delta)
        remaining_population = generate_population(items, population_size - crossover_population.shape[0] ,kappa)
        population = np.concatenate([crossover_population, remaining_population])
        
    return best_solution, best_fitness 

def compile_everything():
    bin_width, bin_height = 10, 10
    
    items = np.array([
        create_item(0, 6, 6),
        create_item(1, 6, 6), 
    ])
    
    lgfi(items, bin_width, bin_height)
    pop = generate_population(items, 5, 1.0)
    fit = compute_fitnesses(pop, items, bin_width, bin_height)
    crossover(pop, fit, 0.7, 1.0)    