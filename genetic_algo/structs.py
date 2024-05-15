import numpy as np

Item = np.dtype([
    ('id', np.int32), 
    ('width', np.int32), 
    ('height', np.int32), 
    ('rotated', np.bool_)
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
    ('items', Item, (100,)),
    ('list_of_free_rec', FreeRectangle, (100,)) 
])