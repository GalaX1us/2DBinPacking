import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class FreeRectangle:
    def __init__(self, x, y, width, height):
        self.corner_x = x
        self.corner_y = y
        self.width = width
        self.height = height

class Item:
    def __init__(self, id, width, height, rotated=False):
        self.id = id
        self.width = width
        self.height = height
        self.rotated = rotated
        self.pos_bin = -1
        self.corner_x = 0
        self.corner_y = 0

class Bin:
    def __init__(self, id, width, height):
        self.id = id
        self.width = width
        self.height = height
        self.items = []
        self.list_of_free_rec = [FreeRectangle(0, 0, width, height)]

def find_bottom_left_position(bin):
    min_y = float('inf')
    min_x = float('inf')
    current_rec = None
    current_pos = -1

    for i, rec in enumerate(bin.list_of_free_rec):
        if rec.corner_y < min_y or (rec.corner_y == min_y and rec.corner_x < min_x):
            min_y = rec.corner_y
            min_x = rec.corner_x
            current_rec = rec
            current_pos = i
            
    return current_rec, current_pos

def calculate_gaps(bin, rec):
    
    # Horizontal gap
    horizontal_gap = rec.width
    for item in bin.items:
        if item.corner_y <= rec.corner_y < item.corner_y + item.height:
            if item.corner_x > rec.corner_x and item.corner_x < rec.corner_x + rec.width:
                horizontal_gap = min(horizontal_gap, item.corner_x - rec.corner_x)
    
    # Vertical gap
    vertical_gap = rec.height
    for item in bin.items:
        if item.corner_x <= rec.corner_x < item.corner_x + item.width:
            if item.corner_y > rec.corner_y and item.corner_y < rec.corner_y + rec.height:
                vertical_gap = min(vertical_gap, item.corner_y - rec.corner_y)
    
    return horizontal_gap, vertical_gap

def add_item_to_bin(bin, item, rec, rotated):
    if rotated:
        item.width, item.height = item.height, item.width
        item.rotated = True
    
    item.corner_x = rec.corner_x
    item.corner_y = rec.corner_y
    item.pos_bin = bin.id
    bin.items.append(item)
    
    # Split the free rectangle
    if rotated:
        new_rec_1 = FreeRectangle(rec.corner_x + item.width, rec.corner_y, rec.width - item.width, rec.height)
        new_rec_2 = FreeRectangle(rec.corner_x, rec.corner_y + item.height, item.width, rec.height - item.height)
    else:
        new_rec_1 = FreeRectangle(rec.corner_x + item.width, rec.corner_y, rec.width - item.width, rec.height)
        new_rec_2 = FreeRectangle(rec.corner_x, rec.corner_y + item.height, rec.width, rec.height - item.height)
    
    bin.list_of_free_rec.append(new_rec_1)
    bin.list_of_free_rec.append(new_rec_2)
    
    # Remove the used rectangle
    bin.list_of_free_rec.remove(rec)

def insert_item_into_bin(bin, item):
    rec, pos = find_bottom_left_position(bin)
    if rec is None:
        return False

    horizontal_gap, vertical_gap = calculate_gaps(bin, rec)
    current_gap = min(horizontal_gap, vertical_gap)
    
    fit = False
    rotated = False

    if current_gap == horizontal_gap:
        if item.width <= current_gap and item.height <= rec.height:
            fit = True
        elif item.rotated and item.height <= current_gap and item.width <= rec.height:
            fit = True
            rotated = True
    else:
        if item.height <= current_gap and item.width <= rec.width:
            fit = True
        elif item.rotated and item.width <= current_gap and item.height <= rec.width:
            fit = True
            rotated = True
    
    if fit:
        add_item_to_bin(bin, item, rec, rotated)
        return True
    else:
        # Handle wastage area
        if current_gap == horizontal_gap:
            wastage_height = rec.height
            for existing_item in bin.items:
                if existing_item.corner_y == rec.corner_y and existing_item.corner_x > rec.corner_x:
                    wastage_height = min(wastage_height, existing_item.height)
            wastage_area = FreeRectangle(rec.corner_x, rec.corner_y, current_gap, wastage_height)
        else:
            wastage_width = rec.width
            for existing_item in bin.items:
                if existing_item.corner_x == rec.corner_x and existing_item.corner_y > rec.corner_y:
                    wastage_width = min(wastage_width, existing_item.width)
            wastage_area = FreeRectangle(rec.corner_x, rec.corner_y, wastage_width, current_gap)

        bin.list_of_free_rec.append(wastage_area)
        bin.list_of_free_rec.remove(rec)
        return False

def solve_guillotine(items, bins):
    for item in items:
        placed = False
        for bin in bins:
            if insert_item_into_bin(bin, item):
                placed = True
                break
        if not placed:
            new_bin = Bin(len(bins), bins[0].width, bins[0].height)
            bins.append(new_bin)
            insert_item_into_bin(new_bin, item)

def visualize_bins(bins):
    fig, ax = plt.subplots(len(bins), 1, figsize=(10, 5 * len(bins)))
    
    if len(bins) == 1:
        ax = [ax]

    for bin_index, bin in enumerate(bins):
        ax[bin_index].set_xlim(0, bin.width)
        ax[bin_index].set_ylim(0, bin.height)
        ax[bin_index].set_title(f'Bin {bin.id}')
        ax[bin_index].set_aspect('equal')
        
        # Draw the bin border
        bin_border = patches.Rectangle((0, 0), bin.width, bin.height, edgecolor='black', facecolor='none')
        ax[bin_index].add_patch(bin_border)
        
        # Draw the items
        for item in bin.items:
            rect = patches.Rectangle((item.corner_x, item.corner_y), item.width, item.height, 
                                     edgecolor='blue', facecolor='lightblue')
            ax[bin_index].add_patch(rect)
            rx, ry = rect.get_xy()
            cx = rx + rect.get_width() / 2.0
            cy = ry + rect.get_height() / 2.0
            ax[bin_index].annotate(f'ID {item.id}', (cx, cy), color='black', weight='bold', 
                                   fontsize=8, ha='center', va='center')

    plt.tight_layout()
    plt.show()

# Example usage
bins = [Bin(i, 100, 100) for i in range(5)]
items = [Item(i, np.random.randint(10, 50), np.random.randint(10, 50), rotated=True) for i in range(20)]

solve_guillotine(items, bins)
visualize_bins(bins)
