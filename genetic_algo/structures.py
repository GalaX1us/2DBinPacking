import numpy as np
from numba import njit, float32, int32, boolean, void, from_dtype
from numba.types import UniTuple

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

@njit(from_dtype(Bin)(int32, int32, int32), cache=True)
def create_bin(bin_id: int, width: int, height: int) -> np.ndarray:
    """
    Create a new bin with the specified ID, width, and height.

    Parameters:
    - bin_id (int): The ID of the bin.
    - width (int): The width of the bin.
    - height (int): The height of the bin.

    Returns:
    - np.ndarray: A numpy array representing the created bin.
    """
    bin = np.zeros(1, dtype=Bin)[0]
    
    bin['id'] = bin_id
    bin['width'] = width
    bin['height'] = height
    bin['list_of_free_rec'][0]['corner_x'] = 0
    bin['list_of_free_rec'][0]['corner_y'] = 0
    bin['list_of_free_rec'][0]['width'] = width
    bin['list_of_free_rec'][0]['height'] = height
    
    return bin

@njit(from_dtype(FreeRectangle)(int32, int32, int32, int32), cache=True)
def create_free_rectangle(x: int, y: int, width: int, height: int) -> np.ndarray:
    """
    Create a new free rectangle with the specified parameters.

    Parameters:
    - x (int): The x-coordinate of the rectangle's top-left corner.
    - y (int): The y-coordinate of the rectangle's top-left corner.
    - width (int): The width of the rectangle.
    - height (int): The height of the rectangle.

    Returns:
    - np.ndarray: A numpy array representing the created free rectangle.
    """
    
    rect = np.zeros(1, dtype=FreeRectangle)[0]
    
    rect['corner_x'] = x
    rect['corner_y'] = y
    rect['width'] = width
    rect['height'] = height
    
    return rect

@njit(from_dtype(Item)(int32, int32, int32), cache=True)
def create_item(id: int, width: int, height: int) -> np.ndarray:
    """
    Create a new item with the specified ID, width, height, and rotation status.

    Parameters:
    - id (int): The ID of the item.
    - width (int): The width of the item.
    - height (int): The height of the item.

    Returns:
    - np.ndarray: A numpy array representing the created item.
    """
    
    item = np.zeros(1, dtype=Item)[0]
    
    item['id'] = id
    item['width'] = width
    item['height'] = height
    item['rotated'] = False
    item['corner_x'] = -1
    item['corner_y'] = -1
    
    return item

@njit(boolean(from_dtype(Bin), from_dtype(Item), int32, int32), cache=True)
def add_item_to_bin(bin: np.ndarray, item: np.ndarray, x: int, y: int) -> bool:
    """
    Add an item to a bin at the specified position.

    Parameters:
    - bin (np.ndarray): The bin to which the item will be added.
    - item (np.ndarray): The item to add to the bin.
    - x (int): The x-coordinate of the item's top-left corner.
    - y (int): The y-coordinate of the item's top-left corner.

    Returns:
    - bool: True if the item was successfully added, False otherwise.
    """
    
    items = bin['items']
    
    for i in range(len(items)):
        
        # Find the first empty spot
        if items[i]['id'] == -1 or items[i]['width'] == 0:  
            items[i]['id'] = item['id']
            items[i]['width'] = item['width']
            items[i]['height'] = item['height']
            items[i]['rotated'] = item['rotated']
            items[i]['corner_x'] = x
            items[i]['corner_y'] = y
            return True
    # No empty spot available
    return False  

@njit(boolean(from_dtype(Bin), from_dtype(FreeRectangle)), cache=True)
def add_free_rect_to_bin(bin: np.ndarray, free_rect: np.ndarray) -> bool:
    """
    Add a free rectangle to a bin.

    Parameters:
    - bin (np.ndarray): The bin to which the free rectangle will be added.
    - free_rect (np.ndarray): The free rectangle to add to the bin.

    Returns:
    - bool: True if the free rectangle was successfully added, False otherwise.
    """
    
    free_rects = bin['list_of_free_rec']
    
    for i in range(len(free_rects)):
        
        # Find the first empty spot
        if free_rects[i]['width'] == 0:  
            free_rects[i]['corner_x'] = free_rect['corner_x']
            free_rects[i]['corner_y'] = free_rect['corner_y']
            free_rects[i]['width'] = free_rect['width']
            free_rects[i]['height'] = free_rect['height']
            return True
    
    # No empty spot available
    return False  

@njit(boolean(from_dtype(Bin), from_dtype(FreeRectangle)), cache=True)
def remove_free_rect_from_bin(bin: np.ndarray, free_rect: np.ndarray) -> bool:
    """
    Remove a free rectangle from a bin.

    Parameters:
    - bin (np.ndarray): The bin from which the free rectangle will be removed.
    - free_rect (np.ndarray): The free rectangle to remove from the bin.

    Returns:
    - bool: True if the free rectangle was successfully removed, False otherwise.
    """
    
    free_rects = bin['list_of_free_rec']
    
    for i in range(len(free_rects)):
        if (free_rects[i]['corner_x'] == free_rect['corner_x'] and 
            free_rects[i]['corner_y'] == free_rect['corner_y'] and 
            free_rects[i]['width'] == free_rect['width'] and 
            free_rects[i]['height'] == free_rect['height']):
            
            # Shift elements to the left using slicing
            free_rects[i:-1] = free_rects[i+1:]
            
            # Mark last element as empty
            free_rects[-1]['width'] = 0  
            free_rects[-1]['height'] = 0 
            free_rects[-1]['corner_x'] = 0
            free_rects[-1]['corner_y'] = 0
            return True
    
    # Free rectangle not found
    return False  

@njit(void(from_dtype(Bin), int32), cache=True)
def remove_free_rect_from_bin_by_idx(bin: np.ndarray, idx: int) -> None:
    """
    Remove a free rectangle from a bin by its index.

    Parameters:
    - bin (np.ndarray): The bin from which the free rectangle will be removed.
    - idx (int): The index of the free rectangle to remove.
    """
    free_rects = bin['list_of_free_rec']
    
    # Shift elements to the left using slicing
    free_rects[idx:-1] = free_rects[idx+1:]
    
    # Mark last element as empty
    free_rects[-1]['width'] = 0  
    free_rects[-1]['height'] = 0 
    free_rects[-1]['corner_x'] = 0
    free_rects[-1]['corner_y'] = 0

@njit(from_dtype(Item)(from_dtype(Item)[:], int32), cache=True)
def get_item_by_id(items: np.ndarray, id: int) -> np.ndarray:
    """
    Retrieve an item from an array of items by its ID.

    Parameters:
    - items (np.ndarray): An array of items.
    - id (int): The ID of the item to retrieve.

    Returns:
    - np.ndarray: The item with the specified ID, or an empty item if not found.
    """
    
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