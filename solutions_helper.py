from enum import Enum
import os
import time

from binpacking.data_manager import export_solutions_to_json, import_solution_from_json, load_items_from_file
from binpacking.genetic_algo.gen_algo import genetic_algo, initialize_numba_functions
from binpacking.lgfi import lgfi
from binpacking.population_generation import get_corresponding_sequence_by_id
from binpacking.tabu_search import tabu_search
from binpacking.visualization import visualize_bins

class Metaheuristic(Enum):
    GA = 0
    TABU = 1


def generate_all_solutions(selected_metaheuristic, iteration_number, tabu_list_size, kappa, guillotine, rotation, 
                        population_size, nb_generations, crossover_rate, mutation_rate, delta, 
                        input_data_directory, output_data_directory):
    
    for file in os.listdir(input_data_directory):
        if file.endswith(".bp2d"):
            
            file_name = "".join(file.split('.')[:-1])
            full_path = os.path.join(input_data_directory, file)
            
            bin_width, bin_height, items = load_items_from_file(full_path)
            
            print(f"===================== {file_name} =====================")
            print(f"Bin dimensions: {bin_width}x{bin_height}")
            print(f"Number of items: {len(items)}")
            
            start = time.perf_counter()
            
            # Check if the selected metaheuristic is an enum value
            if selected_metaheuristic == Metaheuristic.TABU:
                # ====================== Tabu Search ======================
                best_solution, _ = tabu_search(items=items,
                                               bin_dimensions=(bin_width, bin_height),
                                               iteration_number=iteration_number,
                                               tabu_list_size=tabu_list_size,
                                               kappa=kappa,
                                               guillotine_cut=guillotine,
                                               rotation=rotation)
            else:
                # ====================== Genetic Algo ======================
                best_solution, _ = genetic_algo(items=items,
                                                bin_dimensions=(bin_width, bin_height),
                                                population_size=population_size,
                                                nb_generations=nb_generations,
                                                crossover_rate=crossover_rate,
                                                mutation_rate=mutation_rate,
                                                kappa=kappa,
                                                delta=delta,
                                                guillotine_cut=guillotine,
                                                rotation=rotation)
            
            ordered_items = get_corresponding_sequence_by_id(items, best_solution)
            solution = lgfi(ordered_items, bin_width=bin_width, bin_height=bin_height, 
                            guillotine_cut=guillotine, rotation=rotation)
            time_elapsed = time.perf_counter() - start
            
            solution_file_path = os.path.join(output_data_directory, file_name + "-solution.json") 
            export_solutions_to_json(solution, solution_file_path)
            
            print(f"Time elapsed: {time_elapsed:.1f} seconds")
            print(f"Best solution: {len(solution)} bins")
            print(f"Solution saved to: {solution_file_path}\n")
            
def generate_single_solution(file, selected_metaheuristic, iteration_number, tabu_list_size, kappa, 
            guillotine, rotation, population_size, nb_generations, crossover_rate, mutation_rate, delta,
            input_data_directory, output_data_directory):

    file_name = "".join(file.split('.')[:-1])
    
    full_path = os.path.join(input_data_directory, file)
    
    bin_width, bin_height, items = load_items_from_file(full_path)
    
    print(f"===================== {file_name} =====================")
    print(f"Bin dimensions: {bin_width}x{bin_height}")
    print(f"Number of items: {len(items)}")
    
    start = time.perf_counter()
    
    # Check if the selected metaheuristic is an enum value
    if selected_metaheuristic == Metaheuristic.TABU:
        # ====================== Tabu Search ======================
        best_solution, best_fitness = tabu_search(items=items,
                                                  bin_dimensions=(bin_width, bin_height),
                                                  iteration_number=iteration_number,
                                                  tabu_list_size=tabu_list_size,
                                                  kappa=kappa,
                                                  guillotine_cut=guillotine,
                                                  rotation=rotation)
    else:
        # ====================== Genetic Algo ======================
        best_solution, best_fitness = genetic_algo(items=items,
                                                   bin_dimensions=(bin_width, bin_height),
                                                   population_size=population_size,
                                                   nb_generations=nb_generations,
                                                   crossover_rate=crossover_rate,
                                                   mutation_rate=mutation_rate,
                                                   kappa=kappa,
                                                   delta=delta,
                                                   guillotine_cut=guillotine,
                                                   rotation=rotation)
    
    ordered_items = get_corresponding_sequence_by_id(items, best_solution)
    solution = lgfi(ordered_items, bin_width=bin_width, bin_height=bin_height, 
                    guillotine_cut=guillotine, rotation=rotation)
    time_elapsed = time.perf_counter() - start
    
    solution_file_path = os.path.join(output_data_directory, file_name + "-solution.json")
    export_solutions_to_json(solution, solution_file_path)
    
    print(f"Time elapsed: {time_elapsed:.1f} seconds")
    print(f"Best solution: {len(solution)} bins")
    print(f"Solution saved to: {solution_file_path}\n")
    
def visualize_solution(file, output_data_directory):
    solution_file_path = os.path.join(output_data_directory, file) 
    bins = import_solution_from_json(solution_file_path)
    visualize_bins(bins)