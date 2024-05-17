import numba
from data_manager import load_items_from_file, export_solutions_to_json, import_solution_from_json
from genetic_algo.gen_algo import genetic_algo, compile_everything
from genetic_algo.lgfi import lgfi
from genetic_algo.solutions_generation import get_corresponding_sequence_by_id
from genetic_algo.visualization import visualize_bins
import os
import time

input_data_directory = "Data"
output_data_directory = "Data/Solutions"

population_size = 100
nb_generations = 50
crossover_rate = 0.7
kappa = 1
delta = 1

if __name__ == "__main__":

    s = time.perf_counter()
    compile_everything()
    print(f"Compilation took : {time.perf_counter() - s:.1f} sec")

    for filename in os.listdir(input_data_directory):
        if filename.endswith(".bp2d"):
            full_path = os.path.join(input_data_directory, filename)
            
            bin_width, bin_height, items = load_items_from_file(full_path)
            
            best_solution, best_fitness = genetic_algo(items = items,
                                                    bin_dimensions=(bin_width, bin_height),
                                                    population_size=population_size,
                                                    nb_generations=nb_generations,
                                                    crossover_rate=crossover_rate,
                                                    kappa=kappa,
                                                    delta=delta)
            
            solution = lgfi(get_corresponding_sequence_by_id(items, best_solution), bin_width=bin_width, bin_height=bin_height)
            
            solution_file_path = os.path.join(output_data_directory, filename.split('.')[0] + "-solution.bp2d") 
            export_solutions_to_json(solution, solution_file_path)
                    
            print(f"Processed {filename}")
    
    
    # solution_file_path = os.path.join(output_data_directory, "binpacking2d-01" + "-solution.bp2d") 
    # bins = import_solution_from_json(solution_file_path)
    # visualize_bins(bins)