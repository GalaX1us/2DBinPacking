from data_manager import load_items_from_file, export_solutions_to_json, import_solution_from_json
from genetic_algo.gen_algo import genetic_algo, compile_everything
from genetic_algo.lgfi import lgfi
from genetic_algo.solutions_generation import get_corresponding_sequence_by_id
from genetic_algo.visualization import visualize_bins
import os
import time

INPUT_DATA_DIRECTORY = "Data"
OUTPUT_DATA_DIRECTORY = "Data/Solutions"

POPULATION_SIZE = 150
NB_GENERATIONS = 70
CROSSOVER_RATE = 0.7
KAPPA = 1
DELTA = 1

if __name__ == "__main__":

    
    # ====================== Generate Solutions ======================
    
    compile_everything()

    for filename in os.listdir(INPUT_DATA_DIRECTORY):
        if filename.endswith(".bp2d"):
            full_path = os.path.join(INPUT_DATA_DIRECTORY, filename)
            
            bin_width, bin_height, items = load_items_from_file(full_path)
            
            start = time.perf_counter()
            best_solution, best_fitness = genetic_algo(items = items,
                                                    bin_dimensions=(bin_width, bin_height),
                                                    population_size=POPULATION_SIZE,
                                                    nb_generations=NB_GENERATIONS,
                                                    crossover_rate=CROSSOVER_RATE,
                                                    kappa=KAPPA,
                                                    delta=DELTA)
            
            ordered_items = get_corresponding_sequence_by_id(items, best_solution)
            solution = lgfi(ordered_items, bin_width=bin_width, bin_height=bin_height)
            time_elapsed = time.perf_counter() - start
            
            solution_file_path = os.path.join(OUTPUT_DATA_DIRECTORY, filename.split('.')[0] + "-solution.bp2d") 
            export_solutions_to_json(solution, solution_file_path)
            
            print(f"===================== {filename} =====================")
            print(f"Bin dimensions: {bin_width}x{bin_height}")
            print(f"Number of items: {len(items)}")
            print(f"Time elapsed: {time_elapsed:.1f} seconds")
            print(f"Best solution: {len(solution)} bins")
            print(f"Solution saved to: {solution_file_path}\n")
    
    # ====================== Visualize Solutions ======================
           
    # solution_file_path = os.path.join(OUTPUT_DATA_DIRECTORY, "binpacking2d-13" + "-solution.bp2d") 
    # bins = import_solution_from_json(solution_file_path)
    # visualize_bins(bins)