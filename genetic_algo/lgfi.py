import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
from structures import *

def spliting_process_guillotine(horizontal, bin, old_free_rect, item):
    
    right_x = old_free_rect['corner_x'] + item['width']
    right_y = old_free_rect['corner_y']
    right_width = old_free_rect['width'] - item['width']
    top_x = old_free_rect['corner_x']
    top_y = old_free_rect['corner_y'] + item['height']
    top_height = old_free_rect['height'] - item['height']

    right_height = item['height'] if horizontal else old_free_rect['height']
    top_width = old_free_rect['width'] if horizontal else item['width']

    if right_width > 0 and right_height > 0:
        add_free_rect_to_bin(bin, create_free_rectangle(right_x, right_y, right_width, right_height))
    if top_width > 0 and top_height > 0:
        add_free_rect_to_bin(bin, create_free_rectangle(top_x, top_y, top_width, top_height))
        
    remove_free_rect_from_bin(bin, old_free_rect)
        
def merge_rec_guillotine(bin):
    free_rects = bin['list_of_free_rec']
    i = 0

    while i < len(free_rects) and free_rects[i]['width'] > 0:
        first = free_rects[i]
        merged = False
        merged_rec = None
        
        # Try to find a rectangle that can be merged with 'first'
        for j in range(i + 1, len(free_rects)):
            
            if free_rects[j]['width'] == 0:
                break
            
            second = free_rects[j]
            
            # Check for vertical merge
            if first['width'] == second['width'] and first['corner_x'] == second['corner_x']:
                if first['corner_y'] + first['height'] == second['corner_y']:

                    merged_rec = create_free_rectangle(first['corner_x'], first['corner_y'], first['width'], first['height'] + second['height'])
                    merged = True
                    break
            
            # Check for horizontal merge
            if first['height'] == second['height'] and first['corner_y'] == second['corner_y']:
                if first['corner_x'] + first['width'] == second['corner_x']:

                    merged = True
                    merged_rec = create_free_rectangle(first['corner_x'], first['corner_y'], first['width'] + second['width'], first['height'])
                    break
        
        if merged:
            # Remove the old free rectangle and repalce them with the new merged rectangle
            remove_free_rect_from_bin(bin, first)
            remove_free_rect_from_bin(bin, second)
            add_free_rect_to_bin(bin, merged_rec)
        
            i = 0
        
        else:
            i += 1

def insert_item_lgfi(bin, items):
    best_fit = np.inf

    # Store the best fitting item
    best_fit_item = None
    
    # Keeps track if the best fitting item needs to be rotated
    best_fit_rotated = False

    # Find the bottom leftmost free rectangle
    current_free_rect = find_current_position(bin)
    current_x, current_y = current_free_rect['corner_x'], current_free_rect['corner_y']
    horizontal_gap, vertical_gap = current_free_rect['width'], current_free_rect['height']
    
    remove_free_rect_from_bin(bin, current_free_rect)
    
    # Compute the smallest gap
    current_gap = np.min(horizontal_gap, vertical_gap)
    
    item_nb = len(items)
      
    for i in range(item_nb):
        
        # Stop the loop if we finds an item that perfectly fits in the smallest gap
        if best_fit == 0:
            break
        
        current_item = items[i]
        
        # Since the rotation is allowed test both
        for rotated in [False, True]:
                        
            item_width, item_height = current_item['width'], current_item['height']
            
            if rotated:
                item_width, item_height = current_item['height'], current_item['width']
            
            # If an item fits in the gap
            if item_width <= horizontal_gap and item_height <= vertical_gap:
                
                # Store it if it's the first thatg we encounter
                if not best_fit_item:
                    best_fit_item = current_item
                    best_fit_item = rotated
                
                # Check if it fits perfectly    
                fit = current_gap - item_width
                
                if fit == 0:
                    best_fit_item = current_item
                    best_fit_rotated = rotated
    
    # If no items fits in, we need to declare part of this area as wasted                
    if best_fit_item is None:
        
        # Wastage area
        wastage_width = horizontal_gap
        wastage_height = vertical_gap
        
        # Find the upper edge of the lowest neighboring item
        for existing_item in item_nb:
            
            current_item = items[i]
            
            if existing_item['corner_y'] + existing_item['height'] > ['corner_y']:
                wastage_height = min(wastage_height, existing_item['height'])
        
        # Add the wasted space to the bin    
        wasted_space = create_free_rectangle(current_x, current_y, wastage_width, wastage_height, True)
        add_free_rect_to_bin(wasted_space)
        
        # Add the other part tthat is not wasted
        if wastage_height < vertical_gap:
            non_wasted_space = create_free_rectangle(current_x, current_y+wastage_height, horizontal_gap, vertical_gap-wastage_height)
            add_free_rect_to_bin(bin, non_wasted_space)
            
        return False
    
    
    if best_fit_rotated:
        rotate_item(best_fit_item)
        
    add_item_to_bin(bin, best_fit_item, current_x, current_y)
    
    # Should the guillotine cut be vertical or horizontal
    # Shorter Leftover are prioritized
    guillotine_horizontal = False
    if horizontal_gap - best_fit_item['width'] < vertical_gap - best_fit_item['heigth']:
        guillotine_horizontal = True
        
    spliting_process_guillotine(guillotine_horizontal, bin, current_free_rect, best_fit_item)
    merge_rec_guillotine(bin)
    
    return True

def find_current_position(bin):
    best_free_rect = None
    for rec in bin['list_of_free_rec']:
        if rec['width'] == 0:
            break
        
        if rec['wasted']:
            continue
        
        if rec['corner_y'] < best_free_rect['corner_y'] or (rec['corner_y'] == best_free_rect['corner_y'] and rec['corner_x'] < best_free_rect['corner_x']):
            best_free_rect = rec
    return rec

def lgfi(items, bin_width, bin_height):
    bins = []
    bin_id = 0
    unpacked_items = np.copy(items)
    item_nb = len(unpacked_items)
    
    for i in range(item_nb):
        placed = False
        for bin in bins:
            a = insert_item_lgfi(bin, items)
            if a:
                placed = True
                break
        if not placed:
            new_bin = create_bin(bin_id, bin_width, bin_height)
            insert_item_lgfi(new_bin, items)
            bins.append(new_bin)
            bin_id += 1
    return bins

def visualize_bins(bins):
    fig, ax = plt.subplots(len(bins), 1, figsize=(10, 5 * len(bins)))
    
    if len(bins) == 1:
        ax = [ax]

    for bin_index, bin in enumerate(bins):
        ax[bin_index].set_xlim(0, bin['width'])
        ax[bin_index].set_ylim(0, bin['height'])
        ax[bin_index].set_title(f'Bin {bin["id"]}')
        ax[bin_index].set_aspect('equal')
        
        # Draw the bin border
        bin_border = patches.Rectangle((0, 0), bin['width'], bin['height'], edgecolor='black', facecolor='none')
        ax[bin_index].add_patch(bin_border)
        
        # Draw the items
        for item in bin['items']:
            if item['width'] == 0:  # assuming width 0 means uninitialized item
                continue
            rect = patches.Rectangle((item['corner_x'], item['corner_y']), item['width'], item['height'], 
                                     edgecolor='blue', facecolor='lightblue')
            ax[bin_index].add_patch(rect)
            rx, ry = rect.get_xy()
            cx = rx + rect.get_width() / 2.0
            cy = ry + rect.get_height() / 2.0
            ax[bin_index].annotate(f'ID {item["id"]}', (cx, cy), color='black', weight='bold', 
                                   fontsize=8, ha='center', va='center')

    plt.tight_layout()
    plt.show()

def fitness(solution, bin_width, bin_height):
    bins = lgfi(solution, bin_width, bin_height)
    return len(bins)

# Example usage
if __name__ == "__main__":
    
    # Test de merge_rec_guillotine
    bin = create_bin(0, 100, 100)
    remove_free_rect_from_bin(bin, bin['list_of_free_rec'][0])
    free_rect1 = create_free_rectangle(0, 0, 50, 50)
    free_rect2 = create_free_rectangle(0, 50, 50, 50)
    add_free_rect_to_bin(bin, free_rect1)
    add_free_rect_to_bin(bin, free_rect2)
    merge_rec_guillotine(bin)
    print("Test merge_rec_guillotine:")
    print(bin)


