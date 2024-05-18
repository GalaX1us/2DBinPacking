import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tkinter as tk
import random

def random_pastel_color():
    """
    Generate a random pastel color.
    
    Returns:
    - str: Random pastel hex color code.
    """
    r = (random.random() + 1) / 2
    g = (random.random() + 1) / 2
    b = (random.random() + 1) / 2
    return (r, g, b)

def visualize_bins(bins):
    # Get screen resolution
    root = tk.Tk()
    root.withdraw()
    root.update_idletasks()
    root.attributes('-fullscreen', True)
    root.state('iconic')
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    root.destroy()

    screen_aspect_ratio = screen_width / screen_height
    target_aspect_ratio = 16 / 9
    
    num_bins = len(bins)
    
    # Determine the optimal number of rows and columns
    def optimal_grid(num_bins, target_ratio):
        best_diff = float('inf')
        best_rows = best_cols = 1
        
        for rows in range(1, num_bins + 1):
            cols = (num_bins + rows - 1) // rows
            ratio = cols / rows
            diff = abs(ratio - target_ratio)
            
            if diff < best_diff:
                best_diff = diff
                best_rows, best_cols = rows, cols
        
        return best_rows, best_cols
    
    rows, cols = optimal_grid(num_bins, screen_aspect_ratio)
    
    fig, ax = plt.subplots(rows, cols, figsize=(16, 9))
    
    # Ensure ax is a 2D array for easy indexing
    if rows == 1 and cols == 1:
        ax = np.array([[ax]])
    elif rows == 1:
        ax = np.expand_dims(ax, axis=0)
    elif cols == 1:
        ax = np.expand_dims(ax, axis=1)
    
    for i, bin in enumerate(bins):
        row_index = i // cols
        col_index = i % cols
        
        ax[row_index, col_index].set_xlim(0, bin['width'])
        ax[row_index, col_index].set_ylim(0, bin['height'])
        ax[row_index, col_index].set_title(f'Bin {bin["id"]}')
        ax[row_index, col_index].set_aspect('equal')
        
        # Draw the bin border
        bin_border = patches.Rectangle((0, 0), bin['width'], bin['height'], edgecolor='black', facecolor='none')
        ax[row_index, col_index].add_patch(bin_border)
        
        # Draw the items
        for item in bin['items']:
            if item['width'] == 0:  # assuming width 0 means uninitialized item
                continue
            
            color = random_pastel_color()
            rect = patches.Rectangle((item['corner_x'], item['corner_y']), item['width'], item['height'], 
                                     edgecolor='blue', facecolor=color)
            ax[row_index, col_index].add_patch(rect)
            rx, ry = rect.get_xy()
            cx = rx + rect.get_width() / 2.0
            cy = ry + rect.get_height() / 2.0
            ax[row_index, col_index].annotate(f'ID {item["id"]}', (cx, cy), color='black', weight='bold', 
                                              fontsize=8, ha='center', va='center')

    # Hide any unused subplots
    for j in range(i + 1, rows * cols):
        fig.delaxes(ax[j // cols, j % cols])

    plt.tight_layout()
    plt.show()


