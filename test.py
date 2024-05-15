import numpy as np

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

Bin = np.dtype([
    ('id', np.int32),
    ('width', np.int32), 
    ('height', np.int32), 
    ('items', Item, (50,)),  # Change this value if needed
    ('list_of_free_rec', FreeRectangle, (50,))  # Change this value if needed
])

bin = np.zeros(1, dtype=Bin)[0]

i = 10

print(bin['items'])
bin['items'][i:-1] = bin['items'][i+1:]
print(bin['items'])