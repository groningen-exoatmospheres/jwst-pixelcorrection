from scipy.ndimage import label

def classify_dead_pixels(pixels):
    # Create a binary mask for the dead pixels
    mask = np.zeros((max(p[0] for p in pixels) + 1, max(p[1] for p in pixels) + 1), dtype=int)
    for p in pixels:
        mask[p[0], p[1]] = 1

    # Label connected components
    labeled_array, num_features = label(mask)

    # Group pixels by connected components
    groups = {}
    for p in pixels:
        group_id = labeled_array[p[0], p[1]]
        if group_id not in groups:
            groups[group_id] = []
        groups[group_id].append(p)

    # Sort groups by size in descending order
    sorted_groups = sorted(groups.values(), key=len, reverse=True)

    return sorted_groups