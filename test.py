import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random

# Define the structured array for Item
Item = np.dtype([
    ('id', np.int32),
    ('width', np.int32), 
    ('height', np.int32), 
    ('rotated', np.bool_),
    ('corner_x', np.int32),
    ('corner_y', np.int32)
])

# Define the structured array for FreeRectangle
FreeRectangle = np.dtype([
    ('corner_x', np.int32),
    ('corner_y', np.int32),
    ('width', np.int32),
    ('height', np.int32),
    ('wasted', np.bool_)
])

# Define the structured array for Bin
Bin = np.dtype([
    ('id', np.int32),
    ('width', np.int32), 
    ('height', np.int32), 
    ('items', Item, (50,)),  # Change this value if needed
    ('list_of_free_rec', FreeRectangle, (50,))  # Change this value if needed
])

def initialize_bin(bin_id, width, height):
    bin = np.zeros(1, dtype=Bin)[0]
    bin['id'] = bin_id
    bin['width'] = width
    bin['height'] = height
    bin['list_of_free_rec'][0]['corner_x'] = 0
    bin['list_of_free_rec'][0]['corner_y'] = 0
    bin['list_of_free_rec'][0]['width'] = width
    bin['list_of_free_rec'][0]['height'] = height
    return bin

def create_free_rectangle(x, y, width, height):
    rect = np.zeros(1, dtype=FreeRectangle)[0]
    rect['corner_x'] = x
    rect['corner_y'] = y
    rect['width'] = width
    rect['height'] = height
    return rect

def create_item(id, width, height, rotated):
    return np.array((id, width, height, rotated, -1, -1), dtype=Item)

def rotate_item(item):
    item['width'], item['height'] = item['height'], item['width']
    item['rotated'] = not item['rotated']

def spliting_process_guillotine(horizontal, rec, pack):
    list_of_free_rec = []
    
    right_x = rec['corner_x'] + pack['width']
    right_y = rec['corner_y']
    right_width = rec['width'] - pack['width']
    top_x = rec['corner_x']
    top_y = rec['corner_y'] + pack['height']
    top_height = rec['height'] - pack['height']

    right_height = pack['height'] if horizontal else rec['height']
    top_width = rec['width'] if horizontal else pack['width']

    if right_width > 0 and right_height > 0:
        list_of_free_rec.append(create_free_rectangle(right_x, right_y, right_width, right_height))
    if top_width > 0 and top_height > 0:
        list_of_free_rec.append(create_free_rectangle(top_x, top_y, top_width, top_height))

    return list_of_free_rec

def spliting_guillotine(rec, pack):
    return spliting_process_guillotine(rec['width'] <= rec['height'], rec, pack)

def merge_rec_guillotine(bin):
    i = 0
    while i < len(bin['list_of_free_rec']) and bin['list_of_free_rec'][i]['width'] > 0:
        first = bin['list_of_free_rec'][i]
        check_exist_width = False
        check_exist_height = False
        pos_check_width = -1
        pos_check_height = -1
        for j in range(len(bin['list_of_free_rec'])):
            if j == i or bin['list_of_free_rec'][j]['width'] == 0:
                continue
            second = bin['list_of_free_rec'][j]
            if (first['width'] == second['width'] and first['corner_x'] == second['corner_x'] and 
                second['corner_y'] == first['corner_y'] + first['height']):
                check_exist_width = True
                pos_check_width = j
                break
            if (first['height'] == second['height'] and first['corner_y'] == second['corner_y'] and 
                second['corner_x'] == first['corner_x'] + first['width']):
                check_exist_height = True
                pos_check_height = j
                break
        
        if check_exist_width:
            merged_rec = create_free_rectangle(first['corner_x'], first['corner_y'], first['width'],
                          first['height'] + bin['list_of_free_rec'][pos_check_width]['height'])
            bin['list_of_free_rec'][pos_check_width] = bin['list_of_free_rec'][-1]
            bin['list_of_free_rec'][-1] = create_free_rectangle(0, 0, 0, 0)
            bin['list_of_free_rec'][i] = merged_rec
            i = 0
            continue
        
        if check_exist_height:
            merged_rec = create_free_rectangle(first['corner_x'], first['corner_y'],
                          first['width'] + bin['list_of_free_rec'][pos_check_height]['width'], first['height'])
            bin['list_of_free_rec'][pos_check_height] = bin['list_of_free_rec'][-1]
            bin['list_of_free_rec'][-1] = create_free_rectangle(0, 0, 0, 0)
            bin['list_of_free_rec'][i] = merged_rec
            i = 0
            continue
        
        i += 1

def add_item(bin, item, rotated, x, y):
    if rotated:
        item['width'], item['height'] = item['height'], item['width']
        item['rotated'] = rotated
        
    item['corner_x'] = x
    item['corner_y'] = y        
    
    for i in range(50):
        if bin['items'][i]['width'] == 0:
            bin['items'][i] = item
            break

def best_ranking(bin, pack):
    best_rec = create_free_rectangle(0, 0, 0, 0)
    best_pos = -1
    best_fit = np.inf
    rotated = False
    
    for i, rec in enumerate(bin['list_of_free_rec']):
        if rec['width'] == 0:
            break
        if rec['width'] >= pack['width'] and rec['height'] >= pack['height']:
            fit = rec['width'] * rec['height'] - pack['width'] * pack['height']
            if fit < best_fit:
                best_fit = fit
                best_rec = rec
                best_pos = i
                rotated = False
        if rec['width'] >= pack['height'] and rec['height'] >= pack['width'] and pack['rotated']:
            fit = rec['width'] * rec['height'] - pack['height'] * pack['width']
            if fit < best_fit:
                best_fit = fit
                best_rec = rec
                best_pos = i
                rotated = True
                
    return (best_rec, best_pos), (rotated, best_rec['width'] > 0)

def insert_item_lgfi(bin, items):
    best_fit = np.inf

    first_fit_item = None
    first_fit_rotated = False
    
    best_fit_item = None
    best_fit_rotated = False

    current_free_rect = find_current_position(bin)
    current_x, current_y = current_free_rect['corner_x'], current_free_rect['corner_y']
    horizontal_gap, vertical_gap = current_free_rect['width'], current_free_rect['height']
    
    current_gap = np.min(horizontal_gap, vertical_gap)
    
    item_nb = len(items)
      
    for i in range(item_nb):
        
        if best_fit == 0:
            break
        
        current_item = items[i]
        
        for rotated in [False, True]:
                        
            item_width, item_height = current_item['width'], current_item['height']
            
            if rotated:
                item_width, item_height = current_item['height'], current_item['width']
            
            if item_width <= horizontal_gap and item_height <= vertical_gap:
                
                if not first_fit_item:
                    first_fit_item = current_item
                    first_fit_rotated = rotated
                    
                fit = current_gap - item_width
                
                if fit < best_fit:
                    best_fit_item = current_item
                    best_fit_rotated = rotated
                    
    if first_fit_item is None:
        current_free_rect['wasted'] = True
        return False
    
    if best_fit == 0:
        add_item(bin, best_fit_item, best_fit_rotated, current_x, current_y)
    else:
        add_item(bin, first_fit_item, first_fit_rotated, current_x, current_y)
        
    bin['list_of_free_rec'][best_pos] = bin['list_of_free_rec'][-1]
    bin['list_of_free_rec'][-1] = create_free_rectangle(0, 0, 0, 0)
    new_rec = spliting_guillotine(best_rec, item)
    for rec in new_rec:
        for i in range(100):
            if bin['list_of_free_rec'][i]['width'] == 0:
                bin['list_of_free_rec'][i] = rec
                break
            
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
            new_bin = initialize_bin(bin_id, bin_width, bin_height)
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
    items = np.array([
        (0, 30, 40, False, -1, -1),
        (1, 60, 70, False, -1, -1),
        (2, 50, 50, False, -1, -1),
        (3, 20, 80, False, -1, -1),
    ], dtype=Item)

    rotate_item(items[0])
    print(items[0])