def find_closed_shape_contour(pixel_array, closed_shape_pixels):
    mask = np.zeros_like(pixel_array, dtype=bool)

    for pixel in closed_shape_pixels:
        mask[pixel[0], pixel[1]] = 1

    rows, cols = pixel_array.shape
    for row in range(rows):
        for col in range(cols):
            # Check if the pixel is part of the closed shape
            if np.isnan(pixel_array[row, col]):
                mask[row, col] = 1

    # Expand the mask by setting ones to the expanded pixel array size by one pixel
    expanded_mask = np.pad(mask, pad_width=1, mode='constant', constant_values=1)
    expanded_mask[1:-1, 1:-1] |= mask
    mask = expanded_mask

    # Fill patches of 0 surrounded by 1s with 1s, but only if the holes are <= 20 pixels in size
    labeled_array, num_features = ndimage.label(~mask)
    sizes = ndimage.sum(~mask, labeled_array, range(num_features + 1))
    for i, size in enumerate(sizes):
        if size <= 20:
            mask[labeled_array == i] = 1

    # Remove the outer pixels of each shape
    mask = binary_erosion(mask)

    # remove the edges of the mask
    mask = mask[1:-1, 1:-1]

    return mask
