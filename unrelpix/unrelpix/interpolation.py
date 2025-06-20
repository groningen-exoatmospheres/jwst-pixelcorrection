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

import copy
import shutil
from collections import deque
from typing import List, Generator


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
    max_radius: int = 5,
    iterations: int = 5,
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
        Maximum distance to consider for interpolation. Default is 5.
    iterations : int, optional
        Number of iterations to perform. Default is 5.
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
    .. versionchanged:: 0.0.2
        Added `print_errors` parameter to control error output.
    """

    interpolated: np.ndarray = copy.deepcopy(data).astype(float)

    for _ in range(iterations):
        # 1) Detect all dead pixels and normalize to tuple coords
        dead_pixels: List[List[int]] = [[idx[0], idx[1]] for idx in np.argwhere(np.isnan(interpolated))]
        dead_set: set[tuple[int, int]] = {tuple(dp) for dp in dead_pixels}
        if not dead_set:
            break

        # 2) Cluster into connected components (8-connectivity)
        def get_neighbors(pt: tuple[int, int]) -> 'Generator[tuple[int, int], None, None]':
            """
            Get 8-connected neighbors of a pixel.
            Parameters
            ----------
            pt : tuple of int
                Pixel coordinates as (row, column).
            
            Yields
            ------
            tuple of int
                Neighbor pixel coordinates as (row, column).
            
            Notes
            -----
            This function yields the coordinates of all 8-connected neighbors of a given pixel.
            It checks all surrounding pixels (including diagonals) and yields those that are in the dead set.

            .. versionadded:: 0.0.1
            """
            r, c = pt
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nbr = (r + dr, c + dc)
                    if nbr in dead_set:
                        yield nbr

        components: List[List[tuple[int, int]]] = []
        visited: set[tuple[int, int]] = set()
        for pix in dead_set:
            if pix in visited:
                continue
            queue: deque[tuple[int, int]] = deque([pix])
            comp: List[tuple[int, int]] = []
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
        corner_pixels: set[tuple[int, int]] = set()
        for comp in components:
            for r, c in comp:
                non_dead_neighbors: int = 0
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < data.shape[0] and 0 <= nc < data.shape[1] and (nr, nc) not in dead_set:
                            non_dead_neighbors += 1
                if non_dead_neighbors >= 4:
                    corner_pixels.add((r, c))
        
        pixels_dict: dict[tuple[int, int], list[tuple[tuple[int, int], float, float]]] = {}
        # 4) For each corner pixel, compute neighbors and interpolate
        for r, c in corner_pixels:
            candidates: list[tuple[tuple[int, int], float, float]] = []
            for dr in range(-max_radius, max_radius + 1):
                for dc in range(-max_radius, max_radius + 1):
                    nr, nc = r + dr, c + dc
                    if (dr == 0 and dc == 0) or not (0 <= nr < interpolated.shape[0] and 0 <= nc < interpolated.shape[1]):
                        continue
                    val: float = interpolated[nr, nc]
                    if not np.isnan(val):
                        dist: float = float(np.hypot(dr, dc))
                        candidates.append(((nr, nc), val, dist))
            pixels_dict[(r, c)] = candidates

            if not candidates:
                continue

        # 5) Interpolate values for corner pixels
        for pix, candidates in pixels_dict.items():
            x_points: List[int] = [neighbor_coordinates[1] for neighbor_coordinates, val, dist in candidates if neighbor_coordinates[0] == pix[0]]
            x_points_vals: List[float] = [val for neighbor_coordinates, val, dist in candidates if neighbor_coordinates[0] == pix[0]]

            y_points: List[int] = [neighbor_coordinates[0] for neighbor_coordinates, val, dist in candidates if neighbor_coordinates[1] == pix[1]]
            y_points_vals: List[float] = [val for neighbor_coordinates, val, dist in candidates if neighbor_coordinates[1] == pix[1]]

            if not x_points or not y_points:
                interpolated[pix[0], pix[1]] = np.nan
                continue
        
            # split the x_points into n lists with the break at the point where the list is not continuous
            x_segments: List[List[int]] = split_into_segments(x_points)
            x_segments = [segment for segment in x_segments if any(abs(x - pix[1]) == 1 for x in segment)]
            x_points_vals = [x_points_vals[x_points.index(x)] for segment in x_segments for x in segment]

            y_segments: List[List[int]] = split_into_segments(y_points)
            y_segments = [segment for segment in y_segments if any(abs(y - pix[0]) == 1 for y in segment)]
            y_points_vals = [y_points_vals[y_points.index(y)] for segment in y_segments for y in segment]

            x_points = [x for segment in x_segments for x in segment]
            y_points = [y for segment in y_segments for y in segment]
            
            try:
                popt_x: np.ndarray
                # Perform linear fit for x-coordinates
                popt_x, _ = np.polyfit(x_points, x_points_vals, 1, cov=True)
                popt_y: np.ndarray
                # Perform linear fit for y-coordinates
                popt_y, _ = np.polyfit(y_points, y_points_vals, 1, cov=True)
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
    max_radius: int = 3,
    iterations: int = 5,
    print_errors: bool = False
):
    """
    Interpolate dead pixels in a single FITS file.

    Parameters
    ----------
    ramp_file_path : str
        Path to a FITS file.
    max_radius : int, optional
        Maximum distance to consider for interpolation. Default is 5.
    iterations : int, optional
        Number of iterations to perform. Default is 5.
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
                print_errors
            )

        hdul[1].data = interpolated_data
        hdul.flush()

    print(f"Interpolated data saved to {file_path_interpolated}.")

    return file_path_interpolated
