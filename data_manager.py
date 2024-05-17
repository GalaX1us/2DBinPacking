import json
import os
from genetic_algo.structures import *
import numpy as np
from genetic_algo.visualization import visualize_bins

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def load_items_from_file(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()

    # Variables to hold your parsed data
    bin_width = 0
    bin_height = 0
    items = []

    # Process lines
    for line in lines:
        if line.startswith('BIN_WIDTH'):
            bin_width = int(line.split(':')[1].strip())
        elif line.startswith('BIN_HEIGHT'):
            bin_height = int(line.split(':')[1].strip())
        elif line.startswith('ITEMS'):
            continue
        else:
            parts = line.split()
            if len(parts) == 3:
                item_id = int(parts[0])
                width = int(parts[1])
                height = int(parts[2])
                items.append(create_item(item_id, width, height))

    items_array = np.array(items, dtype=Item)

    return bin_width, bin_height, items_array

# Assuming your bins and items are structured using numpy's structured arrays
def export_solutions_to_json(bins, file_path):
    """
    Exports a list of bins and their contents to a JSON file.

    Parameters:
    - bins (list): A list of structured numpy arrays, where each array represents a bin with items.
    - file_path (str): The path to the output JSON file.
    """
    data_to_export = []

    for bin in bins:
        # Extract bin info and contained items
        bin_info = {
            'id': bin['id'],
            'width': bin['width'],
            'height': bin['height'],
            'items': []
        }
        
        # Loop through items in the bin
        for item in bin['items']:
            if item['width'] > 0 and item['height'] > 0:  # Assuming unused items have zero width and height
                item_info = {
                    'id': item['id'],
                    'width': item['width'],
                    'height': item['height'],
                    'rotated': bool(item['rotated']),
                    'corner_x': item['corner_x'],
                    'corner_y': item['corner_y']
                }
                bin_info['items'].append(item_info)
        
        data_to_export.append(bin_info)

    # Write data to a JSON file
    with open(file_path, 'w') as f:
        json.dump(data_to_export, f, cls=NumpyEncoder, indent=4)
        

def import_solution_from_json(file_path):
    
    with open(file_path, 'r') as file:
        data = json.load(file)

    bins = []
    for bin_data in data:
        bin_id = bin_data['id']
        bin_width = bin_data['width']
        bin_height = bin_data['height']
        
        bin_obj = create_bin(bin_id=bin_id,
                             width=bin_width,
                             height=bin_height)
        
        for item_data in bin_data['items']:
            item = create_item(item_data['id'],
                               item_data['width'],
                               item_data['height'],
                               item_data['rotated'])
            
            add_item_to_bin(bin_obj, item, item_data['corner_x'], item_data['corner_y'])
        
        bins.append(bin_obj)

    return bins
