import numpy as np
from .absolute_change import absolute_change
from scipy.ndimage import binary_fill_holes, binary_erosion
from scipy import ndimage

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


def find_dead_pixels(data, threshold=1):
    rows, cols = data.shape
    dead_pixels, closed_shape_pixels = set(), set()

    for row in range(rows - 1):
        for column in range(cols - 1):
            if np.isnan(data[row, column]):
                dead_pixels.add((row, column))

                # Check surrounding pixels for extreme jumps around NaN
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = row + dr, column + dc
                        # Only consider non-NaN neighbors
                        if 0 <= nr < rows and 0 <= nc < cols and not np.isnan(data[nr, nc]):
                            # Compare this neighbor to its own neighboring pixels
                            for dr2 in [-1, 0, 1]:
                                for dc2 in [-1, 0, 1]:
                                    if dr2 == 0 and dc2 == 0:
                                        continue
                                    r2, c2 = nr + dr2, nc + dc2
                                    if 0 <= r2 < rows and 0 <= c2 < cols and not np.isnan(data[r2, c2]):
                                        if absolute_change(data[nr, nc], data[r2, c2]) > threshold:
                                            # Before marking, ensure all 8 around the NaN are valid (not NaN)
                                            neighbors = [(row + adr, column + adc)
                                                         for adr in [-1, 0, 1]
                                                         for adc in [-1, 0, 1]
                                                         if not (adr == 0 and adc == 0)]
                                            if all(0 <= rr < rows and 0 <= cc < cols and not np.isnan(data[rr, cc])
                                                   for rr, cc in neighbors):
                                                # Mark all eight pixels around the NaN
                                                for rr, cc in neighbors:
                                                    dead_pixels.add((rr, cc))
                                            # Stop further checks once flagged
                                            break
                                else:
                                    continue
                                break
                        else:
                            continue
                        break

            else:
                # Only consider “inner” pixels (not on borders)
                if 0 < row < rows-1 and 0 < column < cols-1:
                    jump_count = 0
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            if dr == 0 and dc == 0:
                                continue
                            r, c = row + dr, column + dc
                            if absolute_change(data[row, column], data[r, c]) > threshold:
                                jump_count += 1
                    if jump_count >= 2:
                        closed_shape_pixels.add((row, column))

    mask = find_closed_shape_contour(data, closed_shape_pixels)

    for i in range(rows):
        for j in range(cols):
            if mask[i, j] or np.isnan(data[i, j]):
                dead_pixels.add((i, j))
            else:
                # x-axis checks
                if (j == 0 and absolute_change(data[i, j], data[i, j + 1]) > threshold) or \
                   (j == cols - 1 and absolute_change(data[i, j], data[i, j - 1]) > threshold) or \
                   (0 < j < cols - 1 and absolute_change(data[i, j], data[i, j - 1]) > threshold and \
                    absolute_change(data[i, j], data[i, j + 1]) > threshold):
                    dead_pixels.add((i, j))
                # y-axis checks
                if (i == 0 and absolute_change(data[i, j], data[i + 1, j]) > threshold) or \
                   (i == rows - 1 and absolute_change(data[i, j], data[i - 1, j]) > threshold) or \
                   (0 < i < rows - 1 and absolute_change(data[i, j], data[i - 1, j]) > threshold and \
                    absolute_change(data[i, j], data[i + 1, j]) > threshold):
                    dead_pixels.add((i, j))

    return [list(pixel) for pixel in set(dead_pixels)]
