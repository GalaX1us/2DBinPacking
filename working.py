def insert_item_lgfi(bin, items):
    perfect_fit = False

    # Store the best fitting item
    best_fit_item = None
    
    # Keeps track if the best fitting item needs to be rotated
    best_fit_rotated = False

    # Find the bottom leftmost free rectangle
    current_free_rect_idx = find_current_position_idx(bin)
    if current_free_rect_idx == -1:
        return False, best_fit_item
    
    current_free_rect = bin['list_of_free_rec'][current_free_rect_idx]
    
    current_x, current_y = current_free_rect['corner_x'], current_free_rect['corner_y']
    horizontal_gap, vertical_gap = current_free_rect['width'], current_free_rect['height']
    
    # Compute the smallest gap
    current_gap = min(horizontal_gap, vertical_gap)
    
    item_nb = len(items)
      
    for i in range(item_nb):
        
        # Stop the loop if we finds an item that perfectly fits in the smallest gap
        if perfect_fit:
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
                    best_fit_rotated = rotated
                
                # Check if it fits perfectly    
                if current_gap - item_width == 0:
                    best_fit_item = current_item
                    best_fit_rotated = rotated
                    perfect_fit = True
                    break
    
    # If no items fits in, we need to declare part of this area as wasted                
    if best_fit_item is None:
        
        wastage_height = vertical_gap
        
        # Find the upper edge of the lowest neighboring item
        for i in range(len(bin['items'])):
                        
            current_item = bin['items'][i]
            
            if current_item['width'] == 0:
                break
            
            if current_item['corner_y'] + current_item['height'] > current_y:
                wastage_height = min(wastage_height, current_item['corner_y'] + current_item['height'] - current_y)
        
        
        # Add the other part that is not wasted
        if wastage_height < vertical_gap:
            current_free_rect['corner_x'] = current_x
            current_free_rect['corner_y'] = current_y+wastage_height    
            current_free_rect['width'] = horizontal_gap
            current_free_rect['height'] = vertical_gap-wastage_height
            
            merge_rec_guillotine(bin)
        
        # Remove it if everything is wasted
        else:
            remove_free_rect_from_bin(bin, current_free_rect)
        
        return False, best_fit_item
    
    
    if best_fit_rotated:
        rotate_item(best_fit_item)
        
    add_item_to_bin(bin, best_fit_item, current_x, current_y)
    
    # Should the guillotine cut be vertical or horizontal
    # Shorter Leftover are prioritized
    new_horizontal_gap = horizontal_gap - best_fit_item['width']
    new_vertical_gap = vertical_gap - best_fit_item['height']
    
    guillotine_horizontal = True if new_horizontal_gap < new_vertical_gap else False
    
    # It removes the old free space at the end
    spliting_process_guillotine(guillotine_horizontal, bin, current_free_rect, best_fit_item)
    
    # No need to do the guillotine process if the item perfectly fits the free space
    if new_horizontal_gap > 0 and new_vertical_gap > 0:
        merge_rec_guillotine(bin)
    
    return True, best_fit_item