import numpy as np

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

class Bin:
    def __init__(self, id, width, height):
        self.id = id
        self.width = width
        self.height = height
        self.items = []
        self.list_of_free_rec = [FreeRectangle(0, 0, width, height)]

def spliting_process_guillotine(horizontal, rec, pack):
    list_of_free_rec = []
    
    right_x     = rec.corner_x + pack.width
    right_y     = rec.corner_y
    right_width = rec.width - pack.width
    top_x       = rec.corner_x
    top_y       = rec.corner_y + pack.height
    top_height  = rec.height - pack.height

    right_height = pack.height if horizontal else rec.height
    top_width = rec.width if horizontal else pack.width

    if right_width > 0 and right_height > 0:
        list_of_free_rec.append(FreeRectangle(right_x, right_y, right_width, right_height))
    if top_width > 0 and top_height > 0:
        list_of_free_rec.append(FreeRectangle(top_x, top_y, top_width, top_height))

    return list_of_free_rec

def spliting_guillotine(rec, pack):
    return spliting_process_guillotine(rec.width <= rec.height, rec, pack)

def merge_rec_guillotine(bin):
    i = 0
    while i < len(bin.list_of_free_rec):
        first = bin.list_of_free_rec[i]
        check_exist_width = False
        check_exist_height = False
        pos_check_width = -1
        pos_check_height = -1
        for j in range(len(bin.list_of_free_rec)):
            if j == i:
                continue
            second = bin.list_of_free_rec[j]
            if (first.width == second.width and first.corner_x == second.corner_x and 
                second.corner_y == first.corner_y + first.height):
                check_exist_width = True
                pos_check_width = j
                break
            if (first.height == second.height and first.corner_y == second.corner_y and 
                second.corner_x == first.corner_x + first.width):
                check_exist_height = True
                pos_check_height = j
                break
        
        if check_exist_width:
            merged_rec = FreeRectangle(first.corner_x, first.corner_y, first.width,
                                       first.height + bin.list_of_free_rec[pos_check_width].height)
            bin.list_of_free_rec.pop(pos_check_width)
            bin.list_of_free_rec.pop(i)
            bin.list_of_free_rec.append(merged_rec)
            # Reset loop to handle newly merged rectangles
            i = 0
            continue
        
        if check_exist_height:
            merged_rec = FreeRectangle(first.corner_x, first.corner_y,
                                       first.width + bin.list_of_free_rec[pos_check_height].width, first.height)
            bin.list_of_free_rec.pop(pos_check_height)
            bin.list_of_free_rec.pop(i)
            bin.list_of_free_rec.append(merged_rec)
            # Reset loop to handle newly merged rectangles
            i = 0
            continue
        
        i += 1

def best_ranking(bin, pack):
    best_rec = None
    best_pos = -1
    best_fit = float('inf')
    rotated = False
    
    for i, rec in enumerate(bin.list_of_free_rec):
        if rec.width >= pack.width and rec.height >= pack.height:
            fit = rec.width * rec.height - pack.width * pack.height
            if fit < best_fit:
                best_fit = fit
                best_rec = rec
                best_pos = i
                rotated = False
        if rec.width >= pack.height and rec.height >= pack.width and pack.rotated:
            fit = rec.width * rec.height - pack.height * pack.width
            if fit < best_fit:
                best_fit = fit
                best_rec = rec
                best_pos = i
                rotated = True
                
    return (best_rec, best_pos), (rotated, best_rec is not None)

def add_item(bin, pack, rotated, x, y):
    if rotated:
        pack.width, pack.height = pack.height, pack.width
    bin.items.append(pack)
    pack.corner_x = x
    pack.corner_y = y

def insert_item_guillotine(bin, pack):
    best_ranking_return = best_ranking(bin, pack)
    if not best_ranking_return[1][1]:
        return False
    pack.pos_bin = bin.id
    best_rec = best_ranking_return[0][0]
    best_pos = best_ranking_return[0][1]
    rotated = best_ranking_return[1][0]
    add_item(bin, pack, rotated, best_rec.corner_x, best_rec.corner_y)
    bin.list_of_free_rec.pop(best_pos)
    new_rec = spliting_guillotine(best_rec, pack)
    bin.list_of_free_rec.extend(new_rec)
    merge_rec_guillotine(bin)
    return True

def solve_guillotine(items, bins):
    for item in items:
        for bin in bins:
            if insert_item_guillotine(bin, item):
                break

# Example usage
bins = [Bin(i, 100, 100) for i in range(5)]
items = [Item(i, np.random.randint(10, 50), np.random.randint(10, 50), rotated=True) for i in range(20)]

solve_guillotine(items, bins)

for bin in bins:
    print(f"Bin {bin.id} contains items: {[item.id for item in bin.items]}")
