import numpy as np
from numba import jit, njit

Item = np.dtype([
    ('id', np.int32), 
    ('width', np.int32), 
    ('height', np.int32), 
    ('rotated', np.bool_),
    ('corner_x', np.int32),
    ('corner_y', np.int32)
])

FreeRectangle = np.dtype([
    ('corner_x', np.int32),
    ('corner_y', np.int32),
    ('width', np.int32),
    ('height', np.int32)
])

Bin = np.dtype([
    ('id', np.int32),
    ('width', np.int32), 
    ('height', np.int32), 
    ('items', Item, (50,)),
    ('list_of_free_rec', FreeRectangle, (50,)) 
])

@njit(cache = True, fastmath = True, nogil = True) 
def create_bin(bin_id, width, height):
    bin = np.zeros(1, dtype=Bin)[0]
    bin['id'] = bin_id
    bin['width'] = width
    bin['height'] = height
    bin['list_of_free_rec'][0]['corner_x'] = 0
    bin['list_of_free_rec'][0]['corner_y'] = 0
    bin['list_of_free_rec'][0]['width'] = width
    bin['list_of_free_rec'][0]['height'] = height
    return bin

@njit(cache = True, fastmath = True, nogil = True)
def create_free_rectangle(x, y, width, height):
    rect = np.zeros(1, dtype=FreeRectangle)[0]
    rect['corner_x'] = x
    rect['corner_y'] = y
    rect['width'] = width
    rect['height'] = height
    return rect

@njit(cache = True, fastmath = True, nogil = True)
def create_item(id, width, height, rotated = False):
    item = np.zeros(1, dtype=Item)[0]
    item['id'] = id
    item['width'] = width
    item['height'] = height
    item['rotated'] = rotated
    item['corner_x'] = -1
    item['corner_y'] = -1

    return item

@njit(cache = True, fastmath = True, nogil = True)
def add_item_to_bin(bin, item, x, y):
    items = bin['items']
    for i in range(len(items)):
        if items[i]['id'] == -1 or items[i]['width'] == 0:  # Find the first empty spot
            items[i]['id'] = item['id']
            items[i]['width'] = item['width']
            items[i]['height'] = item['height']
            items[i]['corner_x'] = x
            items[i]['corner_y'] = y
            return True
        
    return False  # No empty spot available

@njit(cache = True, fastmath = True, nogil = True)
def add_free_rect_to_bin(bin, free_rect):
    free_rects = bin['list_of_free_rec']
    for i in range(len(free_rects)):
        if free_rects[i]['width'] == 0:  # Find the first empty spot
            free_rects[i]['corner_x'] = free_rect['corner_x']
            free_rects[i]['corner_y'] = free_rect['corner_y']
            free_rects[i]['width'] = free_rect['width']
            free_rects[i]['height'] = free_rect['height']
            return True
    return False  # No empty spot available

@njit(cache = True, fastmath = True, nogil = True)
def remove_free_rect_from_bin(bin, free_rect):
    free_rects = bin['list_of_free_rec']
    for i in range(len(free_rects)):
        if (free_rects[i]['corner_x'] == free_rect['corner_x'] and 
            free_rects[i]['corner_y'] == free_rect['corner_y'] and 
            free_rects[i]['width'] == free_rect['width'] and 
            free_rects[i]['height'] == free_rect['height']):
            
            # Shift elements to the left using slicing
            free_rects[i:-1] = free_rects[i+1:]
            free_rects[-1]['width'] = 0  # Mark last element as empty
            free_rects[-1]['height'] = 0 
            free_rects[-1]['corner_x'] = 0
            free_rects[-1]['corner_y'] = 0
            return True
    return False  # Free rectangle not found

@njit(cache = True, fastmath = True, nogil = True)
def remove_free_rect_from_bin_by_idx(bin, idx):
    free_rects = bin['list_of_free_rec']
    
    # Shift elements to the left using slicing
    free_rects[idx:-1] = free_rects[idx+1:]
    free_rects[-1]['width'] = 0  # Mark last element as empty
    free_rects[-1]['height'] = 0 
    free_rects[-1]['corner_x'] = 0
    free_rects[-1]['corner_y'] = 0

@njit(cache = True, fastmath = True, nogil = True)
def get_item_by_id(items, id):
    item = np.zeros(1, dtype=Item)[0]
    
    for i in range(len(items)):
        current_item = items[i]
        
        if current_item['width'] == 0 or current_item['id'] == -1:
            break
        
        if current_item['id'] == id:
            item['id'] = id
            item['width'] = current_item['width']
            item['height'] = current_item['height']
            item['rotated'] = current_item['rotated']
            item['corner_x'] = current_item['corner_x']
            item['corner_y'] = current_item['corner_y']
            
            return item
    
    return item