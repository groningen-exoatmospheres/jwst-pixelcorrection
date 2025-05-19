from .split_into_segments import split_into_segments
from .find_dead_pixels import *
from collections import deque

def interpolate_dead_pixels(data, threshold=1.5, confidence_treshold=4, max_radius=1, iterations=1):
    """
    Interpolate only the corner dead pixels of each rectangular cluster of dead pixels.

    This version first identifies corner pixels from clusters, then computes neighbors just for them.

    Parameters
    ----------
    data : 2D np.ndarray
        Input array with "dead" pixels identified by `find_dead_pixels_2d`.
    threshold : float
        Threshold passed to `find_dead_pixels_2d` to detect dead pixels.
    confidence_treshold : float
        Threshold for confidence in interpolation. If confidence is below this value, the pixel will not be interpolated.
    max_radius : int
        Search radius for candidate neighbors.
    iterations : int
        Number of iterative passes (in case interpolation cascades).

    Returns
    -------
    np.ndarray
        Copy of `data` with only corner pixels of each rectangular dead-pixel cluster interpolated.
    """
    interpolated = data.copy().astype(float)
    initial_dead_pixels = find_dead_pixels(interpolated, threshold)
    #print(initial_dead_pixels)
    for pixel in initial_dead_pixels:
        interpolated[pixel[0], pixel[1]] = np.nan

    low_confidence_pixels, high_confidence_pixels = [], []

    for _ in range(iterations):
        # 1) Detect all dead pixels and normalize to tuple coords
        dead_pixels = find_dead_pixels(interpolated, threshold)
        #print(dead_pixels)
        dead_set = {tuple(dp) for dp in dead_pixels}
        if not dead_set:
            break

        # 2) Cluster into connected components (8-connectivity)
        def get_neighbors(pt):
            r, c = pt
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nbr = (r + dr, c + dc)
                    if nbr in dead_set:
                        yield nbr

        components = []
        visited = set()
        for pix in dead_set:
            if pix in visited:
                continue
            queue = deque([pix])
            comp = []
            visited.add(pix)
            while queue:
                cur = queue.popleft()
                comp.append(cur)
                for nbr in get_neighbors(cur):
                    if nbr not in visited:
                        visited.add(nbr)
                        queue.append(nbr)
            components.append(comp)

        # 3) Identify corner pixels for each component
        corner_pixels = set()
        for comp in components:
            for r, c in comp:
                non_dead_neighbors = 0
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < data.shape[0] and 0 <= nc < data.shape[1] and (nr, nc) not in dead_set:
                            non_dead_neighbors += 1
                if non_dead_neighbors >= 4:
                    corner_pixels.add((r, c))
        
        # Remove low confidence pixels from corner pixels
        corner_pixels.difference_update(low_confidence_pixels)
        #print(corner_pixels)

        pixels_dict = {}
        # 4) For each corner pixel, compute neighbors and interpolate
        for r, c in corner_pixels:
            candidates = []
            for dr in range(-max_radius, max_radius + 1):
                for dc in range(-max_radius, max_radius + 1):
                    nr, nc = r + dr, c + dc
                    if (dr == 0 and dc == 0) or not (0 <= nr < interpolated.shape[0] and 0 <= nc < interpolated.shape[1]):
                        continue
                    val = interpolated[nr, nc]
                    if not np.isnan(val):
                        dist = np.hypot(dr, dc)
                        candidates.append(((nr, nc), val, dist))
            pixels_dict[(r, c)] = candidates

            if not candidates:
                continue

        for pix, candidates in pixels_dict.items():
            x_points = [neighbor_coordinates[1] for neighbor_coordinates, val, dist in candidates if neighbor_coordinates[0] == pix[0]]
            x_points_vals = [val for neighbor_coordinates, val, dist in candidates if neighbor_coordinates[0] == pix[0]]

            y_points = [neighbor_coordinates[0] for neighbor_coordinates, val, dist in candidates if neighbor_coordinates[1] == pix[1]]
            y_points_vals = [val for neighbor_coordinates, val, dist in candidates if neighbor_coordinates[1] == pix[1]]

            if not x_points or not y_points:
                interpolated[pix[0], pix[1]] = np.nan
                continue
        
            # split the x_points into n lists with the break at the point where the list is not continuous
            x_segments = split_into_segments(x_points)
            x_segments = [segment for segment in x_segments if any(abs(x - pix[1]) == 1 for x in segment)]
            x_points_vals = [x_points_vals[x_points.index(x)] for segment in x_segments for x in segment]

            y_segments = split_into_segments(y_points)
            y_segments = [segment for segment in y_segments if any(abs(y - pix[0]) == 1 for y in segment)]
            y_points_vals = [y_points_vals[y_points.index(y)] for segment in y_segments for y in segment]

            x_points = [x for segment in x_segments for x in segment]
            y_points = [y for segment in y_segments for y in segment]
            
            # linear fit
            try:
                popt_x, pcov_x = np.polyfit(x_points, x_points_vals, 1, cov=True)
                popt_y, pcov_y = np.polyfit(y_points, y_points_vals, 1, cov=True)
            except np.linalg.LinAlgError as e:
                #print(f"Error during linear fit for pixel {pix}: {e}")
                interpolated[pix[0], pix[1]] = np.nan
                continue
            except ValueError as e:
                #print(f"ValueError during linear fit for pixel {pix}: {e}")
                interpolated[pix[0], pix[1]] = np.nan
                continue
            except TypeError as e:
                #print(f"TypeError during linear fit for pixel {pix}: {e}")
                interpolated[pix[0], pix[1]] = np.nan
                continue


            # Find the point value
            new_x = np.polyval(popt_x, pix[1])
            new_y = np.polyval(popt_y, pix[0])

            # Ensure correlation values are non-negative for weights
            weight_x = abs(np.std([x - (popt_x[0] * x + popt_x[1]) for x in x_points_vals]))
            weight_y = abs(np.std([y - (popt_y[0] * y + popt_y[1]) for y in y_points_vals]))

            # Normalize weights
            total_weight = 1 / weight_x + 1 / weight_y
            weight_x = (1 / weight_x) / total_weight
            weight_y = (1 / weight_y) / total_weight

            # Weighted average
            interpolated_value = weight_x * new_x + weight_y * new_y

            # Calculate confidence as the inverse of the standard deviation of residuals
            confidence_x = 1 / (np.std([x - (popt_x[0] * x + popt_x[1]) for x in x_points_vals]) + 1e-6)
            confidence_y = 1 / (np.std([y - (popt_y[0] * y + popt_y[1]) for y in y_points_vals]) + 1e-6)
            confidence = (confidence_x + confidence_y) / 2

            # Print or store confidence for debugging or analysis
            #print(f"Pixel {pix}: Interpolated value = {interpolated_value}, Confidence = {confidence}")
            if confidence < confidence_treshold:
                #print(f"Confidence too low for pixel {pix}: {confidence}")
                low_confidence_pixels.append(pix)
            else:
                high_confidence_pixels.append([pix, confidence])

            # Assign the interpolated value to the pixel
            interpolated[pix[0], pix[1]] = interpolated_value

        #remove duplicates from low confidence pixels
        low_confidence_pixels = list(set(low_confidence_pixels))
#        for pixel in low_confidence_pixels:
#            interpolated[pixel[0], pixel[1]] = np.nan

    # turn it into a dictionary
    high_confidence_pixels = {pixel[0]: pixel[1] for pixel in high_confidence_pixels}
    #print(high_confidence_pixels)
    #print(f"Overall average confidence of {np.mean(high_confidence_pixels)}")

    return interpolated, high_confidence_pixels
