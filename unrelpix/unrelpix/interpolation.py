"""
Interpolation (:mod:`unrelpix.interpolation`)
=====================================================

.. sectionauthor:: Fran Stimac

Function reference
------------------

This module provides functions to interpolate dead pixels in 2D arrays, particularly for astronomical data.]

.. autosummary::
    :toctree: generated/

    interpolate_dead_pixels -- interpolate dead pixels in a 2D array
    interpolate_fits_file -- interpolate dead pixels in a FITS file
"""

import numpy as np
from astropy.io import fits
from scipy.ndimage import label

import copy
import shutil
from collections import deque
from typing import List, Generator
import warnings



def split_into_segments(points: List[int]) -> List[List[int]]:
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

    Notes
    -----
    This function takes a sorted list of integers and splits it into segments where each segment contains
    consecutive integers. For example, the input [1, 2, 3, 5, 6, 8] would yield [[1, 2, 3], [5, 6], [8]].

    .. versionadded:: 0.0.1  
    """

    if not points:
        return []
    segments: List[List[int]] = []
    current_segment: List[int] = [points[0]]
    for i in range(1, len(points)):
        if points[i] == points[i - 1] + 1:
            current_segment.append(points[i])
        else:
            segments.append(current_segment)
            current_segment = [points[i]]
    segments.append(current_segment)
    return segments


def split_into_segments(points: List[int]) -> List[List[int]]:
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

    Notes
    -----
    This function takes a sorted list of integers and splits it into segments where each segment contains
    consecutive integers. For example, the input [1, 2, 3, 5, 6, 8] would yield [[1, 2, 3], [5, 6], [8]].

    .. versionadded:: 0.0.1  
    """

    if not points:
        return []
    segments: List[List[int]] = []
    current_segment: List[int] = [points[0]]
    for i in range(1, len(points)):
        if points[i] == points[i - 1] + 1:
            current_segment.append(points[i])
        else:
            segments.append(current_segment)
            current_segment = [points[i]]
    segments.append(current_segment)
    return segments

def interpolate_dead_pixels(
    data: np.ndarray,
    max_radius: int = 6,
    iterations: int = 24,
    poly_index: int = 1,
    print_errors: bool = False
    ) -> np.ndarray:
    """
    Interpolate only the corner dead pixels of each rectangular cluster of dead pixels.

    This version first identifies corner pixels from clusters, then computes neighbors just for them.

    Parameters
    ----------
    data : 2D np.ndarray
        Input array with "dead" pixels identified by `find_dead_pixels_2d`.
    max_radius : int, optional
        Maximum distance to consider for interpolation. Default is 6.
    iterations : int, optional
        Number of iterations to perform. Default is 24.
    poly_index : int, optional
        Polynomial index for interpolation. Default is 1 (linear interpolation).
    print_errors : bool, optional
        If True, print errors encountered during interpolation. Default is False.

    Returns
    -------
    np.ndarray
        Copy of `data` with only corner pixels of each rectangular dead-pixel cluster interpolated.

    Raises
    ------
    np.linalg.LinAlgError
        If a linear algebra error occurs during interpolation.
    ValueError
        If a value error occurs during interpolation.
    TypeError
        If a type error occurs during interpolation.

    Notes
    -----
    This function performs the following steps:
    1. Detects all dead pixels and normalizes them to tuple coordinates.
    2. Clusters the dead pixels into connected components using 8-connectivity.
    3. Identifies corner pixels for each component.
    4. For each corner pixel, computes its neighbors and performs linear interpolation.
    5. Returns a new array with interpolated values for the corner pixels.
    The function uses a maximum radius to limit the search for neighbors and performs multiple iterations
    to refine the interpolation. It also handles errors gracefully if they occur during the interpolation process.

    .. versionadded:: 0.0.1
    .. versionchanged:: 0.0.4
        Added `print_errors` parameter to control error output.
        Added poly_index parameter to allow polynomial fitting.
    """

    interpolated: np.ndarray = copy.deepcopy(data).astype(float)

    for _ in range(iterations):
        # 1) Detect all dead pixels and normalize to tuple coords
        dead_pixels: List[List[int]] = [[idx[0], idx[1]] for idx in np.argwhere(np.isnan(interpolated))]
        dead_set: set[tuple[int, int]] = {tuple(dp) for dp in dead_pixels}
        if not dead_set:
            break

        # 2) Cluster into connected components (8-connectivity)
        # Optimized connected components using numpy and scipy.ndimage
        dead_mask = np.isnan(interpolated)
        structure = np.ones((3, 3), dtype=bool)  # 8-connectivity
        labeled, num_features = label(dead_mask, structure=structure)
        components: List[List[tuple[int, int]]] = [
            list(map(tuple, np.argwhere(labeled == i)))
            for i in range(1, num_features + 1)
        ]

        # 3) Identify corner pixels for each component (vectorized)
        corner_pixels: set[tuple[int, int]] = set()
        for comp in components:
            comp_arr = np.array(comp)
            # Generate all 8-connected neighbor offsets except (0,0)
            offsets = np.array([(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if not (dr == 0 and dc == 0)])
            neighbors = comp_arr[:, None, :] + offsets[None, :, :]  # shape (N, 8, 2)
            nr = neighbors[..., 0]
            nc = neighbors[..., 1]
            # Mask for valid indices
            valid_mask = (nr >= 0) & (nr < data.shape[0]) & (nc >= 0) & (nc < data.shape[1])
            # For each pixel, count how many neighbors are not in dead_set
            for idx, (row, col) in enumerate(comp):
                valid_nbrs = [(int(nr[idx, j]), int(nc[idx, j])) for j in range(8) if valid_mask[idx, j]]
                non_dead_neighbors = sum((nbr not in dead_set) for nbr in valid_nbrs)
                if non_dead_neighbors >= 4:
                    corner_pixels.add((row, col))

        # Optimized vectorized neighbor search for all corner pixels at once
        def compute_pixels_dict(interpolated: np.ndarray, corner_pixels: set[tuple[int, int]], max_radius: int) -> dict:
            """
            Compute neighbors for corner pixels and their values in the interpolated array.

            Parameters
            ----------
            interpolated : np.ndarray
                The 2D array with interpolated values.
            corner_pixels : set[tuple[int, int]]
                Set of corner pixel coordinates.
            max_radius : int
                Maximum radius to consider for neighbor search.
            
            Returns
            -------
            dict
                Dictionary where keys are corner pixel coordinates and values are lists of tuples
                containing neighbor coordinates and their corresponding values.
            """

            if not corner_pixels:
                return {}

            corner_pixels_arr = np.array(list(corner_pixels))
            rows, cols = interpolated.shape

            # Generate all offsets except (0, 0)
            offsets = np.array([
            (dr, dc)
            for dr in range(-max_radius, max_radius + 1)
                for dc in range(-max_radius, max_radius + 1)
                    if not (dr == 0 and dc == 0)
            ], dtype=int)

            # Compute all neighbor coordinates for all corner pixels
            neighbors = corner_pixels_arr[:, None, :] + offsets[None, :, :]  # (N, M, 2)
            nr = neighbors[..., 0]
            nc = neighbors[..., 1]

            # Mask for valid indices
            valid_mask = (nr >= 0) & (nr < rows) & (nc >= 0) & (nc < cols)
            # Mask for not-NaN values
            vals = np.full(nr.shape, np.nan, dtype=interpolated.dtype)
            vals[valid_mask] = interpolated[nr[valid_mask], nc[valid_mask]]
            not_nan_mask = ~np.isnan(vals)

            # Build result dictionary
            result = {}
            for idx, cp in enumerate(corner_pixels_arr):
                valid_neighbors = np.where(not_nan_mask[idx])
                coords = list(zip(nr[idx][valid_neighbors], nc[idx][valid_neighbors]))
                values = vals[idx][valid_neighbors]
                result[tuple(cp)] = list(zip(coords, values))
            return result
        
        pixels_dict = compute_pixels_dict(interpolated, corner_pixels, max_radius)

        # 5) Interpolate values for corner pixels
        for pix, candidates in pixels_dict.items():
            # Extract x and y neighbors and their values
            x_neighbors = [(coord[1], val) for coord, val in candidates if coord[0] == pix[0]]
            y_neighbors = [(coord[0], val) for coord, val in candidates if coord[1] == pix[1]]

            if not x_neighbors or not y_neighbors:
                interpolated[pix[0], pix[1]] = np.nan
                continue

            # Split into continuous segments and keep only those adjacent to the pixel
            def filter_segments(points: List[tuple[int, float]], pix_val: int) -> tuple[List[int], List[float]]:
                """
                Filters segments of points to keep only those that are adjacent to the pixel value.
                
                Parameters
                ----------
                points : list of tuples
                    List of tuples where each tuple contains a point and its value.
                pix_val : int
                    The pixel value to check adjacency against.
                
                Returns
                -------
                tuple
                    A tuple containing two lists: flat_points and flat_vals.
                    flat_points contains the x or y coordinates of the points,
                    flat_vals contains the corresponding values.
                """


                segments = split_into_segments([pt for pt, _ in points])
                segments = [seg for seg in segments if any(abs(x - pix_val) == 1 for x in seg)]
                flat_points = [x for seg in segments for x in seg]
                flat_vals = [val for pt, val in points if pt in flat_points]
                return flat_points, flat_vals

            x_points, x_points_vals = filter_segments(x_neighbors, pix[1])
            y_points, y_points_vals = filter_segments(y_neighbors, pix[0])
            
            try:
                with warnings.catch_warnings():
                    #warnings.simplefilter('ignore', np.RankWarning)
                    warnings.simplefilter('ignore', RuntimeWarning)

                    popt_x: np.ndarray
                    # Perform fit for x-coordinates
                    popt_x, _ = np.polyfit(x_points, x_points_vals, poly_index, cov=True)
                    popt_y: np.ndarray
                    # Perform fit for y-coordinates
                    popt_y, _ = np.polyfit(y_points, y_points_vals, poly_index, cov=True)
            except np.linalg.LinAlgError as e:
                if print_errors:
                    print(f"Error during linear fit for pixel {pix}: {e}")
                interpolated[pix[0], pix[1]] = np.nan
                continue
            except ValueError as e:
                if print_errors:
                    print(f"ValueError during linear fit for pixel {pix}: {e}")
                interpolated[pix[0], pix[1]] = np.nan
                continue
            except TypeError as e:
                if print_errors:
                    print(f"TypeError during linear fit for pixel {pix}: {e}")
                interpolated[pix[0], pix[1]] = np.nan
                continue

            # Find the point value
            new_x: float = float(np.polyval(popt_x, pix[1]))
            new_y: float = float(np.polyval(popt_y, pix[0]))

            # Ensure correlation values are non-negative for weights
            weight_x: float = abs(np.std([x - (popt_x[0] * x + popt_x[1]) for x in x_points_vals]))
            weight_y: float = abs(np.std([y - (popt_y[0] * y + popt_y[1]) for y in y_points_vals]))

            # Normalize weights
            total_weight: float = 1 / weight_x + 1 / weight_y
            weight_x = (1 / weight_x) / total_weight
            weight_y = (1 / weight_y) / total_weight

            # Weighted average
            interpolated_value: float = weight_x * new_x + weight_y * new_y

            # Assign the interpolated value to the pixel
            interpolated[pix[0], pix[1]] = interpolated_value

    return interpolated


def interpolate_fits_file(
    ramp_file_path: str,
    max_radius: int = 6,
    iterations: int = 24,
    poly_index: int = 1,
    print_errors: bool = False
):
    """
    Interpolate dead pixels in a single FITS file.

    Parameters
    ----------
    ramp_file_path : str
        Path to a FITS file.
    max_radius : int, optional
        Maximum distance to consider for interpolation. Default is 6.
    iterations : int, optional
        Number of iterations to perform. Default is 24.
    poly_index : int, optional
        Polynomial index for interpolation. Default is 1 (linear interpolation).
    print_errors : bool, optional
        If True, print errors encountered during interpolation. Default is False.

    Returns
    -------
    str
        Path to the new FITS file with interpolated dead pixels.

    Raises
    ------
    ValueError
        If the input file is not a FITS file or does not contain "_rampfitted.fits" in its name.
    TypeError
        If the input file path is not a string.

    Notes
    -----
    This function creates a new FITS file with interpolated dead pixels. 
    It reads the input file, processes the data to interpolate dead pixels, and saves the result to a new file.

    .. versionadded:: 0.0.1
    .. versionchanged:: 0.0.2
    """

    if not ramp_file_path.endswith(".fits"):
        raise ValueError("Input file must be a FITS file.")

    if "_rampfitted.fits" not in ramp_file_path:
        raise ValueError("Input file be a ramp-fitted file.")
    
    file_path_interpolated: str = ramp_file_path.replace(
        "_rampfitted.fits", "_interpolated.fits"
    )
    shutil.copyfile(ramp_file_path, file_path_interpolated)

    with fits.open(file_path_interpolated, mode='update') as hdul:
        data: np.ndarray = hdul[1].data
        interpolated_data: np.ndarray = np.empty_like(data, dtype=np.float32)

        for i in range(data.shape[0]):
            interpolated_data[i, :, :] = interpolate_dead_pixels(
                data[i, :, :],
                max_radius,
                iterations,
                poly_index,
                print_errors
            )

        hdul[1].data = interpolated_data
        hdul.flush()

    print(f"Interpolated data saved to {file_path_interpolated}.")

    return file_path_interpolated
