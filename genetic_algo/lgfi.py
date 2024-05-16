import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
from fitness import calculate_fitness
from visualization import visualize_bins
from structures import *
from numba import njit

def remove_item_from_remaining(remaining, item_id):
    """
    Remove an item from the remaining items array based on its id.
    
    Parameters:
    - remaining (np.ndarray): The array of remaining items.
    - item_id (int): The ID of the item to remove.

    Returns:
    np.ndarray: Updated array of remaining items.
    """
    mask = remaining['id'] != item_id
    remaining = remaining[mask]
    
    return remaining

def spliting_process_guillotine(horizontal, bin, old_free_rect, item):
    """
    Perform the guillotine split process after placing an item in a bin.
    
    The direction of the cut (horizontal or vertical) is influenced by the remaining space dimensions:
    - A horizontal cut may be made if the space above the item is less restrictive 
      for future placements than the space to the right.
    - A vertical cut might be preferred if the remaining space to the right of 
      the item offers less flexibility than the space above.

    Parameters:
    - horizontal (bool): Determines if the guillotine cut should be horizontal.
    - bin (np.ndarray): The bin where the item is placed.
    - old_free_rect (np.ndarray): The free rectangle where the item is placed.
    - item (np.ndarray): The item that is placed in the bin.

    Changes the structure of free rectangles within the bin based on where the item was placed.
    """
    
    changes = 0
    
    right_x = old_free_rect['corner_x'] + item['width']
    right_y = old_free_rect['corner_y']
    right_width = old_free_rect['width'] - item['width']
    top_x = old_free_rect['corner_x']
    top_y = old_free_rect['corner_y'] + item['height']
    top_height = old_free_rect['height'] - item['height']

    right_height = item['height'] if horizontal else old_free_rect['height']
    top_width = old_free_rect['width'] if horizontal else item['width']

    if right_width > 0 and right_height > 0:
        old_free_rect['width'], old_free_rect['height'] = right_width, right_height
        old_free_rect['corner_x'], old_free_rect['corner_y'] = right_x, right_y
        
        changes += 1
        
    if top_width > 0 and top_height > 0:
        if changes == 0:
            old_free_rect['width'], old_free_rect['height'] = top_width, top_height
            old_free_rect['corner_x'], old_free_rect['corner_y'] = top_x, top_y
        else:
            add_free_rect_to_bin(bin, create_free_rectangle(top_x, top_y, top_width, top_height))
        
        changes += 1
        
    if changes == 0:
        remove_free_rect_from_bin(bin, old_free_rect)

def merge_rec_guillotine(bin):
    """
    Merge free rectangles in the bin that can be combined either horizontally or vertically.

    Parameters:
    - bin (np.ndarray): The bin containing the free rectangles.

    Modifies the bin in-place by merging adjacent free rectangles to reduce fragmentation.
    """
    
    free_rects = bin['list_of_free_rec']
    i = 0

    while i < len(free_rects) and free_rects[i]['width'] > 0:
        first = free_rects[i] 
        
        # Try to find a rectangle that can be merged with 'first'
        for j in range(i + 1, len(free_rects)):
            
            if free_rects[j]['width'] == 0:
                break
            
            second = free_rects[j]
            
            # Check for vertical merge
            if first['width'] == second['width'] and first['corner_x'] == second['corner_x']:
                if (first['corner_y'] + first['height'] == second['corner_y']) or (second['corner_y'] + second['height'] == first['corner_y']) :
                    
                    first['height'] += second['height']
                    remove_free_rect_from_bin_by_idx(bin, j)
                    i = 0
                    break
            
            # Check for horizontal merge
            if first['height'] == second['height'] and first['corner_y'] == second['corner_y']:
                if (first['corner_x'] + first['width'] == second['corner_x']) or (second['corner_x'] + second['width'] == first['corner_x']):

                    first['width'] += second['width']
                    remove_free_rect_from_bin_by_idx(bin, j)
                    i = 0
                    break
        
        
        i += 1
        
def check_fit_and_rotation(items, horizontal_gap, vertical_gap):
    """
    Check each item to see if it fits in the given gaps with or without rotation.

    Parameters:
    - items (np.ndarray): Array of items to be checked for fitting.
    - horizontal_gap (int): The width of the current free space.
    - vertical_gap (int): The height of the current free space.

    Returns:
    - (np.ndarray): Best fitting item, if found.
    - (bool): Whether the best fitting item needs to be rotated.
    """
    
    best_fit_item = None
    best_fit_rotated = False
    perfect_fit = False
    
    current_gap = min(horizontal_gap, vertical_gap)

    for i in range(len(items)):
        
        current_item = items[i]
        
        for rotated in [False, True]:
            item_width, item_height = current_item['width'], current_item['height']
            
            if rotated:
                item_width, item_height = current_item['height'], current_item['width']

            if item_width <= horizontal_gap and item_height <= vertical_gap:
                
                # Store it if it's the first thatg we encounter
                if not best_fit_item:
                    best_fit_item = current_item
                    best_fit_rotated = rotated
                
                # Check if it fits perfectly    
                if current_gap - item_width == 0:
                    best_fit_item = current_item
                    best_fit_rotated = rotated
                    perfect_fit = True
                    break
                
        if perfect_fit:
            break
    
    return best_fit_item, best_fit_rotated

        
def handle_wastage(bin, current_free_rect, current_y, vertical_gap):
    """
    Handle the situation where no items fit into the current free rectangle, potentially marking it as wasted.

    Parameters:
    - bin (np.ndarray): The bin being processed.
    - current_free_rect (np.ndarray): The free rectangle that might be wasted.
    - current_y (int): The vertical starting point of the free rectangle.
    - vertical_gap (int): The height of the free rectangle.

    Adjusts the free rectangle in the bin to account for wasted space.
    """
    
    wastage_height = vertical_gap

    for item in bin['items']:
        if item['width'] == 0:
            break
        if item['corner_y'] + item['height'] > current_y:
            wastage_height = min(wastage_height, item['corner_y'] + item['height'] - current_y)

    if wastage_height < vertical_gap:
        # Update the current free rectangle for the non-wasted part
        current_free_rect['corner_y'] = current_y + wastage_height
        current_free_rect['height'] = vertical_gap - wastage_height
        
        merge_rec_guillotine(bin)
        
    else:
        # Entire space is wasted, so remove it
        remove_free_rect_from_bin(bin, current_free_rect)
        
        
def perform_placement(bin, current_free_rect, best_fit_item, best_fit_rotated, current_x, current_y):
    """
    Place the selected item into the bin, performing necessary updates to the free rectangles.

    Parameters:
    - bin (np.ndarray): The bin where the item is being placed.
    - current_free_rect (np.ndarray): The free rectangle where the item will be placed.
    - best_fit_item (np.ndarray): The item to be placed.
    - best_fit_rotated (bool): Indicates if the item needs to be rotated for placement.
    - current_x (int): The horizontal starting point of the placement.
    - current_y (int): The vertical starting point of the placement.

    Executes the placement and updates the bin's structure accordingly.
    """
    
    if best_fit_rotated:
        rotate_item(best_fit_item)

    add_item_to_bin(bin, best_fit_item, current_x, current_y)

    new_horizontal_gap = current_free_rect['width'] - best_fit_item['width']
    new_vertical_gap = current_free_rect['height'] - best_fit_item['height']
    guillotine_horizontal = new_horizontal_gap < new_vertical_gap

    spliting_process_guillotine(guillotine_horizontal, bin, current_free_rect, best_fit_item)
    if new_horizontal_gap > 0 and new_vertical_gap > 0:
        merge_rec_guillotine(bin)


def insert_item_lgfi(bin, items):
    """
    Attempt to insert an item into the given bin by finding the best fitting position.

    Parameters:
    - bin (np.ndarray): The bin to attempt item insertion.
    - items (np.ndarray): List of items to be placed.

    Returns:
    - (bool): Whether the insertion was successful.
    - (np.ndarray): The item that was inserted.
    """
    
    current_free_rect_idx = find_current_position_idx(bin)
    if current_free_rect_idx == -1:
        return False, None
    
    current_free_rect = bin['list_of_free_rec'][current_free_rect_idx]
    current_x, current_y = current_free_rect['corner_x'], current_free_rect['corner_y']
    horizontal_gap, vertical_gap = current_free_rect['width'], current_free_rect['height']
    
    best_fit_item, best_fit_rotated = check_fit_and_rotation(items, horizontal_gap, vertical_gap)
    
    if not best_fit_item:
        handle_wastage(bin, current_free_rect, current_y, vertical_gap)
        return False, None
    
    perform_placement(bin, current_free_rect, best_fit_item, best_fit_rotated, current_x, current_y)
    return True, best_fit_item


def find_current_position_idx(bin):
    """
    Find the index of the bottom leftmost free rectangle for placement in the bin.

    Parameters:
    - bin (np.ndarray): The bin being evaluated.

    Returns:
    - (int): Index of the best free rectangle, or -1 if none are suitable.
    """
    
    best_free_rect_idx = -1
    best_free_rect = None
    
    for i in range(len(bin['list_of_free_rec'])):
        rec = bin['list_of_free_rec'][i]
        
        # Check if the rectangle is valid (non-zero width implies it's still available for placement)
        if rec['width'] == 0:
            break
        
        # Initialize best_free_rect if it's the first valid rectangle encountered
        if best_free_rect is None:
            best_free_rect = rec
            best_free_rect_idx = i
            continue
        
        # Update if the current rectangle is lower or at the same y but more to the left
        if (rec['corner_y'] < best_free_rect['corner_y']) or \
           (rec['corner_y'] == best_free_rect['corner_y'] and rec['corner_x'] < best_free_rect['corner_x']):
            best_free_rect = rec
            best_free_rect_idx = i
    
    return best_free_rect_idx


def lgfi(items, bin_width, bin_height):
    """
    Main function to apply the Level Guillotine Fit Insertion algorithm to pack items into bins.

    Parameters:
    - items (np.ndarray): Array of items to be packed.
    - bin_width (int): The width of each new bin.
    - bin_height (int): The height of each new bin.

    Returns:
    - list: A list of bins containing the packed items.
    """
    
    bins = []
    bin_id = 0
    unpacked_items = np.copy(items)
    
    while len(unpacked_items) > 0:
        placed = False
        
        # Attempt to place an item in the existing bins
        for i in range(len(bins)):
            bin = bins[i]
            placed, item = insert_item_lgfi(bin, unpacked_items)
            
            # Remove the item from the remaining list if it has been placed
            if placed:
                unpacked_items = remove_item_from_remaining(unpacked_items, item['id'])
                break
            
        if not placed:
            space_available = any(bin['list_of_free_rec'][0]['width'] != 0 for bin in bins)
            if not space_available:

                new_bin = create_bin(bin_id, bin_width, bin_height)
                bins.append(new_bin)
                bin_id += 1
            
    return bins

def fitness(solution, bin_width, bin_height):
    bins = lgfi(solution, bin_width, bin_height)
    return len(bins)

# Example usage
if __name__ == "__main__":
    
    items = np.array([
        create_item(1, 30, 20),
        create_item(2, 50, 40), 
        create_item(3, 60, 30),  
        create_item(4, 50, 50),  
        create_item(5, 10, 10),  
        create_item(6, 20, 20),
        create_item(7, 20, 20),  
        create_item(8, 80, 80),
        create_item(9, 80, 80)
    ])
    
    # Define bin dimensions
    bin_width = 100
    bin_height = 100
    
    start = time.perf_counter()
    bins = lgfi(items, bin_width, bin_height)
    print(time.perf_counter()-start)
    
    fit = calculate_fitness(bins)
    print(fit) 
