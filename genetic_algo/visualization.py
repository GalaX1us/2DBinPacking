from matplotlib import patches, pyplot as plt


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