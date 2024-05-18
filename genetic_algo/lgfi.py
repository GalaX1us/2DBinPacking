from typing import Tuple
from genetic_algo.structures import *
from numba.typed import List

@njit(cache = True)
def remove_item_from_remaining(remaining: np.ndarray, item_id: int) -> np.ndarray:
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

@njit(cache = True)
def spliting_process_guillotine(horizontal: bool, bin: np.ndarray, old_free_rect: np.ndarray, item: np.ndarray) -> None:
    """
    Perform the guillotine split process after placing an item in a bin.
    Changes the structure of free rectangles within the bin based on where the item was placed.

    Parameters:
    - horizontal (bool): Determines if the guillotine cut should be horizontal.
    - bin (np.ndarray): The bin where the item is placed.
    - old_free_rect (np.ndarray): The free rectangle where the item is placed.
    - item (np.ndarray): The item that is placed in the bin.
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

@njit(cache = True)
def check_fit_and_rotation(items: np.ndarray, horizontal_gap: int, vertical_gap: int) -> Tuple[int, bool]: 
    """
    Check each item to see if it fits in the given gaps with or without rotation.

    Parameters:
    - items (np.ndarray): Array of items to be checked for fitting.
    - horizontal_gap (int): The width of the current free space.
    - vertical_gap (int): The height of the current free space.

    Returns:
    - (int): The index of the selectioned item. -1 otherwise.
    - (bool): Whether the best fitting item needs to be rotated.
    """
    
    best_fit_item_idx = -1
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
                if best_fit_item_idx == -1:
                    best_fit_item_idx = current_item['id']
                    best_fit_rotated = rotated
                
                # Check if it fits perfectly    
                if current_gap - item_width == 0:
                    best_fit_item_idx = current_item['id']
                    best_fit_rotated = rotated
                    perfect_fit = True
                    break
                
        if perfect_fit:
            break
    
    return best_fit_item_idx, best_fit_rotated

@njit(cache = True)
def perform_placement(bin: np.ndarray, current_free_rect: np.ndarray, best_fit_item: np.ndarray, best_fit_rotated: bool, current_x: int, current_y: int) -> None:
    """
    Place the selected item into the bin, performing necessary updates to the free rectangles.
    
    Splitting rule (Shorter Leftover): The guillotine cut is horizontal if the remaining horizontal space
    is smaller than the remaining vertical space, and vertical otherwise.

    Parameters:
    - bin (np.ndarray): The bin where the item is being placed.
    - current_free_rect (np.ndarray): The free rectangle where the item will be placed.
    - best_fit_item (np.ndarray): The item to be placed.
    - best_fit_rotated (bool): Indicates if the item needs to be rotated for placement.
    - current_x (int): The horizontal starting point of the placement.
    - current_y (int): The vertical starting point of the placement.
    """
    
    if best_fit_rotated:
        best_fit_item['width'], best_fit_item['height'] = best_fit_item['height'], best_fit_item['width']
        best_fit_item['rotated'] = not best_fit_item['rotated']

    
    add_item_to_bin(bin, best_fit_item, current_x, current_y)

    new_horizontal_gap = current_free_rect['width'] - best_fit_item['width']
    new_vertical_gap = current_free_rect['height'] - best_fit_item['height']
    
    # Splitting rule: Shorter Leftover
    guillotine_horizontal = new_horizontal_gap < new_vertical_gap

    spliting_process_guillotine(guillotine_horizontal, bin, current_free_rect, best_fit_item)

@njit(cache = True)
def insert_item_lgfi(bin: np.ndarray, items: np.ndarray) -> int:
    """
    Attempt to insert an item into the given bin by finding the best fitting position.

    Parameters:
    - bin (np.ndarray): The bin to attempt item insertion.
    - items (np.ndarray): Array of items to be placed.

    Returns:
    - int: The ID of the item that was inserted, or -1 if the insertion was unsuccessful.
    """
    
    current_free_rect_idx = find_current_position_idx(bin)
    if current_free_rect_idx == -1:
        return -1
    
    current_free_rect = bin['list_of_free_rec'][current_free_rect_idx]
    current_x, current_y = current_free_rect['corner_x'], current_free_rect['corner_y']
    horizontal_gap, vertical_gap = current_free_rect['width'], current_free_rect['height']
    
    best_fit_item_id, best_fit_rotated = check_fit_and_rotation(items, horizontal_gap, vertical_gap)
    best_fit_item = get_item_by_id(items, best_fit_item_id)
    
    if best_fit_item['width'] == 0 or best_fit_item['height'] == 0:
        remove_free_rect_from_bin(bin, current_free_rect)
        return -1
    
    perform_placement(bin, current_free_rect, best_fit_item, best_fit_rotated, current_x, current_y)
    return best_fit_item_id

@njit(cache = True)
def find_current_position_idx(bin: np.ndarray) -> int:
    """
    Find the index of the bottom leftmost free rectangle for placement in the bin.

    Parameters:
    - bin (np.ndarray): The bin being evaluated.

    Returns:
    - (int): Index of the best free rectangle, or -1 if none are suitable.
    """
    
    best_free_rect_idx = -1
    lowest_y = np.inf
    lowest_x = np.inf

    for i in range(len(bin['list_of_free_rec'])):
        rec = bin['list_of_free_rec'][i]
        if rec['width'] == 0:
            continue
        
        if best_free_rect_idx == -1 or rec['corner_y'] < lowest_y or \
           (rec['corner_y'] == lowest_y and rec['corner_x'] < lowest_x):
            lowest_y = rec['corner_y']
            lowest_x = rec['corner_x']
            best_free_rect_idx = i

    return best_free_rect_idx

@njit(cache = True)
def lgfi(items: np.ndarray, bin_width: int, bin_height: int) -> List:
    """
    Main function to apply the Level Guillotine Fit Insertion algorithm to pack items into bins.

    Parameters:
    - items (np.ndarray): Array of items to be packed.
    - bin_width (int): The width of each new bin.
    - bin_height (int): The height of each new bin.

    Returns:
    - list: A list of bins containing the packed items.
    """
    
    bins = List()
    bin_count = 0
    unpacked_items = np.copy(items)
    
    while unpacked_items.size > 0:
        
        item_id = -1
        # Attempt to place an item in the existing bins
        for i in range(len(bins)):
            bin = bins[i]
            item_id = insert_item_lgfi(bin, unpacked_items)
            
            # Numba Lists use copies and not views like standard Python
            bins[i] = bin
            
            # Remove the item from the remaining list if it has been placed
            if item_id != -1:
                unpacked_items = remove_item_from_remaining(unpacked_items, item_id)
                break
            
        if item_id == -1:
            if bin_count == 0 or not np.any(np.array([bins[i]['list_of_free_rec'][0]['width'] != 0 for i in range(bin_count)], dtype=np.bool_)):
                new_bin = create_bin(bin_count, bin_width, bin_height)
                bins.append(new_bin)
                bin_count += 1
                
    return bins
    