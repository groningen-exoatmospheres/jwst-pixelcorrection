"""
Unreliable pixel classification (:mod:`unrelpix.classification`)
=====================================================

.. sectionauthor:: Fran Stimac

Function reference
------------------

This module provides functions for classifying unreliable pixels in astronomical data.
It includes methods to classify dead pixels into groups based on connectivity,
process specific integration and group indices from FITS files, and process entire FITS files
to detect and classify extra pixels.

.. autosummary::
    :toctree: generated/

    classify_dead_pixels -- classify dead pixels into groups based on connectivity
    process_integration_and_group_index -- process a specific integration and group index
    process_fits_file -- process a FITS file to detect and classify extra pixels
"""

import numpy as np
from astropy.io import fits
from scipy.ndimage import label
from typing import List, Tuple

import multiprocessing
from joblib import Parallel, delayed
import shutil

from unrelpix.identification import find_unreliable_pixels
import os


def classify_dead_pixels(pixels: List[Tuple[int, int]]) -> List[List[Tuple[int, int]]]:
    """
    Classify dead pixels into groups based on connectivity.
    Pixels are grouped if they are connected in the 2D grid, and the groups are
    sorted by size in descending order.

    Parameters
    ----------
    pixels : list of tuples
        List of pixel coordinates (x, y) that are considered dead pixels.

    Returns
    -------
    list of lists
        A list of groups, where each group is a list of pixel coordinates.
        Groups are sorted by size in descending order.

    Notes
    -----
    This function uses a binary mask to label connected components in the pixel grid.
    It returns a list of groups of connected dead pixels, sorted by the size of each group.

    .. versionadded:: 0.0.2
    """

    if not pixels:
        return []

    # Create a binary mask for the dead pixels
    mask = np.zeros((max(p[0] for p in pixels) + 1, max(p[1] for p in pixels) + 1), dtype=int)
    for p in pixels:
        mask[p[0], p[1]] = 1

    # Label connected components
    labeled_array, _ = label(mask)

    # Group pixels by connected components
    groups: dict[int, List[Tuple[int, int]]] = {}
    for p in pixels:
        group_id = labeled_array[p[0], p[1]]
        if group_id not in groups:
            groups[group_id] = []
        groups[group_id].append(p)

    # Sort groups by size in descending order
    sorted_groups = sorted(groups.values(), key=len, reverse=True)

    return sorted_groups

def process_integration_and_group_index(
        file_path: str, 
        integration_idx: int, 
        group_idx: int,
        search_radius: int = 15, 
        window_radius: int = 3, 
        var_min: float = 0.5, 
        n_jobs: int = -1
        ) -> Tuple[Tuple[int, int], List[Tuple[int, int]], np.ndarray]:
    """
    Process a specific integration and group index from a FITS file to identify
    unreliable pixels and classify them.
    
    Parameters
    ----------
    file_path : str
        Path to the FITS file containing the data.
    integration_idx : int
        Index of the integration to process.
    group_idx : int
        Index of the group within the integration to process.
    search_radius : int, optional
        Radius around each candidate pixel for Gaussian fitting, by default 15.
    window_radius : int, optional
        Radius for local mean calculation in prefiltering, by default 3.
    var_min : float, optional
        Minimum variance threshold for candidate selection, by default 0.5.
    n_jobs : int, optional
        Number of parallel jobs to run, by default -1 (all processors).
    
    Returns
    -------
    Tuple[Tuple[int, int], List[Tuple[int, int]], np.ndarray]
        A tuple containing:
        - (integration_idx, group_idx): Indices of the processed integration and group.
        - List of tuples representing the coordinates of identified unreliable pixels.
        - A 2D numpy array representing the classification grid for the group.
    Notes
    -----
    This function reads the specified integration and group from the FITS file,
    identifies unreliable pixels using the `find_unreliable_pixels` function, and
    classifies them into categories such as dead pixels, low QE, hot pixels, and others.

    .. versionadded:: 0.0.2  
    """

    with fits.open(file_path, mode='update') as hdul1:
        flagged_pixels: np.ndarray = hdul1[3].data[integration_idx, group_idx, :, :].flatten()
        flagged_pixels = np.log2(flagged_pixels)
        flagged_pixels_indexes: np.ndarray = np.where(~np.isinf(flagged_pixels))[0]

        dead_pixels: np.ndarray = hdul1[2].data[:, :].flatten()
        dead_pixels = np.log2(dead_pixels)
        dead_pixels_indexes: np.ndarray = np.where(~np.isinf(dead_pixels))[0]

        indexes: np.ndarray = np.union1d(dead_pixels_indexes, flagged_pixels_indexes)

        group_data: np.ndarray = hdul1[1].data[integration_idx, group_idx, :, :]
        group_shape: Tuple[int, int] = group_data.shape
        group_data_flat: np.ndarray = group_data.flatten()
        group_data_flat[indexes] = np.nan
        group_data = group_data_flat.reshape(group_shape)

        # Find extra pixels
        extra_pixels: List[Tuple[int, int]] = find_unreliable_pixels(
            group_data,
            search_radius,
            window_radius,
            var_min,
            n_jobs
        )

        if len(extra_pixels) == 0:
            return None

        classified: List[List[Tuple[int, int]]] = classify_dead_pixels(extra_pixels)
        grid: np.ndarray = np.zeros_like(group_data, dtype=np.uint8)

        for i in classified:
            if len(i) == 1:
                x, y = i[0]
                grid[x, y] = 13 if group_data[x, y] < 0 else 11  # LOW_QE or HOT
            elif len(i) == 9 and np.isnan(group_data[i[4][0], i[4][1]]):
                for x, y in i:
                    grid[x, y] = 27  # ADJ_OPEN
                cx, cy = i[4]
                grid[cx, cy] = 26  # OPEN
            elif len(i) == 9 and group_data[i[4][0], i[4][1]] == max(group_data[x, y] for x, y in i):
                for x, y in i:
                    grid[x, y] = 27  # ADJ_OPEN
                cx, cy = i[4]
                grid[cx, cy] = 26  # OPEN
            elif len(i) == 4:
                max_pixel: Tuple[int, int] = max(i, key=lambda pixel: group_data[pixel[0], pixel[1]])
                for pixel in i:
                    x, y = pixel
                    grid[x, y] = 26 if pixel == max_pixel else 27
            else:
                for x, y in i:
                    grid[x, y] = 30  # OTHER_BAD_PIXEL

        flagged_pixels_grid: np.ndarray = hdul1[3].data[integration_idx, group_idx, :, :]
        merged_grid: np.ndarray = np.where(grid != 0, grid, flagged_pixels_grid)

        return (integration_idx, group_idx), extra_pixels, merged_grid



def process_fits_file(
    file_path: str,
    search_radius: int = 15, 
    window_radius: int = 3, 
    var_min: float = 0.5, 
    n_jobs: int = -1,
    output_dir: str = "./"
    ) -> str: 
    """
    Process a FITS file to detect and classify extra (bad) pixels.
    
    Parameters
    ----------
    file_path : str
        Path to the FITS file containing the data.
    search_radius : int, optional
        Radius around each candidate pixel for Gaussian fitting, by default 15.
    window_radius : int, optional
        Radius for local mean calculation in prefiltering, by default 3.
    var_min : float, optional
        Minimum variance threshold for candidate selection, by default 0.5.
    n_jobs : int, optional
        Number of parallel jobs to run, by default -1 (all processors).
    output_dir : str, optional
        Directory to save the modified FITS file, by default "./"

    Returns
    -------
    str
        Path to the modified FITS file with classified pixels.

    Notes
    -----
    This function reads a FITS file, processes each integration and group to identify
    unreliable pixels, and classifies them into categories such as dead pixels, low QE,
    hot pixels, and others. The results are saved back to the FITS file in a
    new extension. The original FITS file is copied to avoid overwriting it.

    .. versionadded:: 0.0.2
    .. versionchanged:: 0.0.3
    """
    

    # Copy the original FITS file to avoid overwriting
    base_filename = os.path.basename(file_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    copied_file_path: str = os.path.join(output_dir, base_filename.replace('.fits', '_modified.fits'))
    shutil.copyfile(file_path, copied_file_path)
    file_path: str = copied_file_path

    # Determine processing indices
    with fits.open(file_path) as hdul1:
        integration_indices: range = range(hdul1[1].data.shape[0])
        group_indices: range = range(hdul1[1].data.shape[1])
    
    all_tasks: list[tuple[int, int]] = [(i, g) for i in integration_indices for g in group_indices]

    def process_task(
        file_path: str, 
        integration_idx: int, 
        group_idx: int,
        search_radius: int, 
        window_radius: int,
        var_min: float,
        n_jobs: int,
    ) -> Tuple[Tuple[int, int], List[Tuple[int, int]], np.ndarray]:
        """
        Process a single integration and group index to identify and classify extra pixels.
        This function is designed to be run in parallel for each integration and group.

        Parameters
        ----------
        file_path : str
            Path to the FITS file containing the data.
        integration_idx : int
            Index of the integration to process.
        group_idx : int
            Index of the group within the integration to process.
        search_radius : int, optional
            Radius around each candidate pixel for Gaussian fitting, by default 15.
        window_radius : int, optional
            Radius for local mean calculation in prefiltering, by default 3.
        var_min : float, optional
            Minimum variance threshold for candidate selection, by default 0.5.
        n_jobs : int, optional
            Number of parallel jobs to run, by default -1 (all processors).
            
        Returns
        -------
        Tuple[Tuple[int, int], List[Tuple[int, int]], np.ndarray]
            A tuple containing:
            - (integration_idx, group_idx): Indices of the processed integration and group.
            - List of tuples representing the coordinates of identified unreliable pixels.
            - A 2D numpy array representing the classification grid for the group.

        Raises
        ------
        Exception
            If an error occurs during processing, it will be caught and printed.

        Notes
        -----
        This function reads the specified integration and group from the FITS file,
        identifies unreliable pixels using the `find_unreliable_pixels` function, and
        classifies them into categories such as dead pixels, low QE, hot pixels, and others.
        It is designed to be run in parallel for efficiency.

        .. versionadded:: 0.0.2
        """

        try:
            return process_integration_and_group_index(
            file_path, integration_idx, group_idx, search_radius, window_radius, var_min, n_jobs
            )
        except Exception as e:
            print(f"Error processing ({integration_idx}, {group_idx}): {e}")
            return None
    
    # Process all tasks in parallel
    results: list[tuple[tuple[int, int], list[tuple[int, int]], np.ndarray]] = Parallel(n_jobs=multiprocessing.cpu_count())(
    delayed(process_task)(
        file_path, i, g, search_radius, window_radius, var_min, n_jobs
    ) for i, g in all_tasks
    )

    results = [res for res in results if res is not None]

    with fits.open(file_path, mode='update') as hdul1:
        for (integration_idx, group_idx), _, merged_grid in results:
            hdul1[3].data[integration_idx, group_idx, :, :] = merged_grid
        hdul1.flush()
        
    print(f"Processing complete. Output file: {copied_file_path}")

    return copied_file_path
