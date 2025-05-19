#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Basic import necessary for configuration.
import os
import warnings
import requests

warnings.simplefilter("ignore", RuntimeWarning)

maindir = os.path.join(os.getcwd(), '../Data/')

# Set CRDS cache directory to user home if not already set.
if os.getenv('CRDS_PATH') is None:
    os.environ['CRDS_PATH'] = os.path.join(os.path.expanduser('~'), 'crds_cache')

# Check whether the CRDS server URL has been set. If not, set it.
if os.getenv('CRDS_SERVER_URL') is None:
    os.environ['CRDS_SERVER_URL'] = 'https://jwst-crds.stsci.edu'

# Output the current CRDS path and server URL in use.
print('CRDS local filepath:', os.environ['CRDS_PATH'])
print('CRDS file server:', os.environ['CRDS_SERVER_URL'])


# In[2]:


# ----------------------General Imports----------------------
import time
import glob
import json
import itertools
import numpy as np
import pandas as pd

# --------------------Astroquery Imports---------------------
from astroquery.mast import Observations

# ----------------------Astropy Imports----------------------
# Astropy utilities for opening FITS files, downloading demo files, etc.
from astropy.io import fits
from astropy.table import Table
from astropy.stats import sigma_clip
from astropy.visualization import ImageNormalize, ManualInterval, LogStretch
from astropy.visualization import LinearStretch, AsinhStretch, simple_norm

# ----------------------Plotting Imports---------------------
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection

# -------------------File Download Imports-------------------
from tqdm.auto import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial


# In[3]:


def absolute_change(a, b):
    """ Find the change in the order of magnitude between two values."""
    return np.abs(np.log10(abs(a)) - np.log10(abs(b)))


# In[4]:


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


# In[5]:


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


# In[6]:


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


# In[7]:


def filter_pixels(file, threshold=1.3, slices=1, min_slices=1, debug_length=10):
    """
    Filters pixels based on their presence in previous and next slices.

    Parameters:
        file (str): Path to the FITS file.
        threshold (float): Threshold for identifying dead pixels.
        slices (int): Number of slices to check before and after the current slice.
        min_slices (int): Minimum number of slices (previous + next) where the pixel must appear.

    Returns:
        list: All clusters of dead pixels.
    """
    all_pixels = []

    with fits.open(file) as hdul:
        data = hdul[1].data
        for i in range(data.shape[0])[:debug_length]:
            dead_pixels = find_dead_pixels(data[i, :, :], threshold)
            clusters = classify_dead_pixels(dead_pixels)
            all_pixels.append(clusters)

    flattened_pixels = [[coord for group in clusters for coord in group] for clusters in all_pixels]

    new_pixels = []

    for i, array in enumerate(flattened_pixels):
        pixel_groups = []
        for pixel in array:
            count_in_prev = sum(pixel in flattened_pixels[i - j] for j in range(1, slices + 1) if i - j >= 0)
            count_in_next = sum(pixel in flattened_pixels[i + j] for j in range(1, slices + 1) if i + j < len(flattened_pixels))
            if count_in_prev + count_in_next >= min_slices:
                pixel_groups.append(pixel)
        new_pixels.append(pixel_groups)

    return new_pixels


# In[8]:


def split_into_segments(points):
    """
    Splits a sorted list of points into continuous segments.

    Parameters
    ----------
    points : list of int
        A sorted list of integer points.

    Returns
    -------
    list of list of int
        A list of continuous segments, where each segment is a list of integers.
    """
    segments = []
    current_segment = [points[0]]
    for i in range(1, len(points)):
        if points[i] == points[i - 1] + 1:
            current_segment.append(points[i])
        else:
            segments.append(current_segment)
            current_segment = [points[i]]
    segments.append(current_segment)
    return segments


# In[ ]:


from collections import deque

def interpolate_dead_pixels(data, dead_pixel_data, confidence_treshold=4, max_radius=1, iterations=1):
    interpolated = data.copy().astype(float)
    initial_dead_pixels = dead_pixel_data.copy()
    #print(initial_dead_pixels)
    for pixel in initial_dead_pixels:
        interpolated[pixel[0], pixel[1]] = np.nan

    low_confidence_pixels, high_confidence_pixels = [], []

    for _ in range(iterations):
        # 1) Detect all dead pixels and normalize to tuple coords
        dead_pixels = initial_dead_pixels
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
            #remove [pix[0], pix[1]] from initial_dead_pixels
            initial_dead_pixels.remove(list(pix))

        #remove duplicates from low confidence pixels
        low_confidence_pixels = list(set(low_confidence_pixels))
#        for pixel in low_confidence_pixels:
#            interpolated[pixel[0], pixel[1]] = np.nan

    # turn it into a dictionary
    high_confidence_pixels = {pixel[0]: pixel[1] for pixel in high_confidence_pixels}
    #print(high_confidence_pixels)
    #print(f"Overall average confidence of {np.mean(high_confidence_pixels)}")

    return interpolated, high_confidence_pixels

