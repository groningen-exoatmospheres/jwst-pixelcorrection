"""
Unreliable pixel identification (:mod:`unrelpix.identification`)
=====================================================

.. sectionauthor:: Fran Stimac

Function reference
------------------

This module provides functions for identifying unreliable pixels in 2D data arrays,
such as those found in astronomical images. It uses Gaussian fitting to analyze
local data profiles and determine if pixels are unreliable based on their
statistical properties. The main function, `find_unreliable_pixels`, processes the
data to identify pixels that deviate significantly from their local neighborhood,
and returns a list of indices for these unreliable pixels.

.. autosummary::
    :toctree: generated/
    
    find_unreliable_pixels -- identify unreliable pixels in a 2D array
    process_pixel          -- process a single pixel to identify unreliable pixels
    prefilter_candidates   -- prefilter candidates for unreliable pixels based on local variance
    interpolate_nans_2d    -- interpolate NaN values in a 2D array
    fit_gaussian_1d        -- fit a Gaussian function to 1D data and calculate R-squared value
    gaussian               -- compute the Gaussian function for given parameters
"""

import numpy as np

from scipy.ndimage import uniform_filter
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

import warnings

from joblib import Parallel, delayed


def gaussian(
        x: np.ndarray, 
        a: float, 
        c: float, 
        d: float
        ) -> np.ndarray:
    """
    Compute the Gaussian function for a given set of parameters.

    .. math::

        f(x) = a \\cdot e^{-\\frac{(x - \\mu)^2}{2c^2}} + d

    Parameters
    ----------
    x : array_like
        Input values where the Gaussian function is evaluated.
    a : float
        Amplitude of the Gaussian peak.
    c : float
        Standard deviation (sigma) of the Gaussian, which controls the width of the peak.
    d : float
        Offset value, which shifts the Gaussian vertically.

    Returns
    -------
    array_like
        The computed Gaussian values at each point in `x`.

    Raises
    ------
    ValueError
        If `x` is empty, as there are no values to compute the Gaussian for.
    ValueError
        If `c` is zero, as this would lead to division by zero in the Gaussian
        formula.

    Notes
    -----
    This function computes the Gaussian function based on the provided parameters.
    The mean of `x` is used as the center of the Gaussian peak. 

    .. versionadded:: 0.0.2
    """

    if len(x) == 0:
        raise ValueError("Input array x must not be empty.")
    if not isinstance(x, np.ndarray):
        x = np.asarray(x)
    if c == 0:
        raise ValueError("Standard deviation c must be non-zero.")

    return a * np.exp(-((x - np.mean(x)) ** 2) / (2 * c ** 2)) + d


def fit_gaussian_1d(data: np.ndarray) -> tuple[float, np.ndarray]:
    """
    Fit a Gaussian function to a 1D data profile and calculate the R-squared value.

    Parameters
    ----------
    data : array_like
        1D array of data points to fit the Gaussian to.

    Returns
    -------
    tuple
        A tuple containing the R-squared value of the fit and the parameters of the fitted Gaussian
        (amplitude, standard deviation, offset).

    Raises
    ------
    RuntimeError
        If the fitting process fails, returns (0, np.zeros(3)).

    Notes
    -----
    This function uses `scipy.optimize.curve_fit` to fit a Gaussian function to the provided data.
    where `a` is the amplitude, `c` is the standard deviation, and `d` is the offset.
    The R-squared value is computed to assess the goodness of fit defined as:
    .. math::
        R^2 = 1 - \\frac{\\sum (y_i - f(x_i))^2}{\\sum (y_i - \\bar{y})^2} 
    where :math:`y_i` are the observed data points, :math:`f(x_i)` are the fitted values, and :math:`\\bar{y}` is the mean of the observed data.
    If the fitting fails, it returns an R-squared value of 0 and a zero array of parameters.

    .. versionadded:: 0.0.2
    """

    x = np.arange(len(data))
    try:
        # Suppress warnings during curve fitting
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Initial guess: amplitude=max(data), stddev=1, offset=1
            initial_guess = [np.max(data), 1, 1]
            popt, _ = curve_fit(gaussian, x, data, p0 = initial_guess)
        
        # Compute fitted values and R-squared
        fitted_values = gaussian(x, *popt)
        r_squared = r2_score(data, fitted_values)
        return r_squared, popt
    except Exception:
        # Return default values if fitting fails
        return 0, np.zeros(3)

def process_pixel(
        r: int, 
        c: int, 
        group_data: np.ndarray, 
        search_radius: int
        ) -> list[tuple[int, int]] | None:
    """
    Process a single pixel to identify unreliable pixels based on Gaussian fitting.

    Parameters
    ----------
    r : int
        Row index of the pixel in the group data.
    c : int
        Column index of the pixel in the group data.
    group_data : np.ndarray
        2D array containing the data for the group.
    search_radius : int
        Radius around the pixel to search for neighbors.

    Returns
    -------
    list[tuple[int, int]] or None
        A list of tuples representing the indices of unreliable pixels if identified,
        otherwise None.

    Notes
    -----
    This function extracts a neighborhood around the specified pixel, fits a Gaussian
    function to the data, and checks if the fit is reliable based on the R-squared value.
    If the fit is reliable, it calculates the full width at half maximum (FWHM) and
    identifies the indices of the pixels that fall within this range. The function
    returns a list of tuples representing the indices of these unreliable pixels.   

    .. versionadded:: 0.0.2 
    """

    row, col = group_data.shape
    x_min: int = max(0, c - search_radius)
    x_max: int = min(col, c + search_radius + 1)

    if (c - x_min) < (x_max - c - 1):
        x_max = c + (c - x_min) + 1
    elif (c - x_min) > (x_max - c - 1):
        x_min = c - (x_max - c - 1)

    # Ensure the pixel is not too close to the edges
    if (x_max - x_min) < 7:
        return None

    neighbors_x: np.ndarray = group_data[r, x_min:x_max]
    linspace_values: np.ndarray = np.arange(x_min, x_max)

    r2_x: float
    popt: np.ndarray
    r2_x, popt = fit_gaussian_1d(neighbors_x)

    if r2_x > 0.7:
        fwhm_constant: float = 2.3548200450 # Constant for converting standard deviation to FWHM (2 * sqrt(2 * ln(2)))
        fwhm: float = fwhm_constant * popt[1]
        center: float = np.mean(linspace_values)
        half_max_left: float = center - fwhm
        half_max_right: float = center + fwhm

        y_fwhm: np.ndarray = (lambda x: popt[0] * np.exp(-((x - center) ** 2) / (2 * popt[1] ** 2)) + popt[2])(linspace_values)

        indices: np.ndarray
        if popt[0] < 0:
            indices = linspace_values[np.where(neighbors_x < y_fwhm)[0]]
        else:
            indices = linspace_values[np.where(neighbors_x > y_fwhm)[0]]

        closest_left: int = linspace_values[np.abs(linspace_values - half_max_left).argmin()]
        closest_right: int = linspace_values[np.abs(linspace_values - half_max_right).argmin()]

        if closest_left < closest_right:
            indices = indices[(indices >= closest_left) & (indices <= closest_right)]
        else:
            indices = indices[(indices >= closest_right) & (indices <= closest_left)]

        if popt[0] > 0:
            if not any(neighbors_x < y_fwhm):
                return None
        else:
            if not any(neighbors_x > y_fwhm):
                return None

        if np.log10(abs(np.max(neighbors_x))) - np.log10(abs(np.min(neighbors_x))) < 0.8 and np.min(neighbors_x) > -200:
            return None
        else:
            return [(r, int(i)) for i in indices]
    return None

def interpolate_nans_2d(data: np.ndarray) -> np.ndarray:
    """
    Interpolate NaN values in a 2D array along each row.

    Parameters
    ----------
    data : np.ndarray
        2D array with NaN values to be interpolated.

    Returns
    -------
    np.ndarray
        2D array with NaN values replaced by interpolated values.

    Notes
    -----
    This function iterates over each row of the 2D array and performs linear interpolation
    to fill NaN values. It uses `np.interp` to interpolate the missing values based on
    the non-NaN values in the same row. The x-coordinates for interpolation are generated
    as a range from 0 to the number of columns in the row.

    .. versionadded:: 0.0.2
    """

    x: np.ndarray = np.arange(data.shape[1])
    for row in range(data.shape[0]):
        nan_mask: np.ndarray = np.isnan(data[row])
        if nan_mask.any():
            data[row, nan_mask] = np.interp(x[nan_mask], x[~nan_mask], data[row, ~nan_mask])
    return data

def prefilter_candidates(
        data: np.ndarray, 
        window_radius: int = 3, 
        var_min: float = 0.5
        ) -> list[tuple[int, int]]:
    """
    Prefilter candidates for unreliable pixels based on local variance.

    Parameters
    ----------
    data : np.ndarray
        2D array of data to analyze.
    window_radius : int, optional
        Radius for local mean calculation, by default 3.
    var_min : float, optional
        Minimum variance threshold for candidate selection, by default 0.5.

    Returns
    -------
    list of tuples
        List of tuples representing the indices of candidate pixels that may be unreliable.

    Notes
    -----
    This function computes the local mean of the data using a uniform filter and then
    identifies candidates where the logarithmic difference between the data and local mean
    exceeds a specified threshold. The candidates are returned as a list of tuples
    representing the indices of the pixels that meet the criteria.

    .. versionadded:: 0.0.2
    """

    local_mean: np.ndarray = uniform_filter(data, size = 2 * window_radius + 1, mode = 'reflect')
    safe_data: np.ndarray = np.where(np.abs(data) > 1e-8, np.abs(data), 1e-8)
    safe_mean: np.ndarray = np.where(np.abs(local_mean) > 1e-8, np.abs(local_mean), 1e-8)
    log_diff: np.ndarray = np.log10(safe_data) - np.log10(safe_mean)

    candidate_indices: np.ndarray = np.argwhere((log_diff >= var_min))

    return [tuple(idx) for idx in candidate_indices]

def find_unreliable_pixels(
        group_data: np.ndarray, 
        search_radius: int = 15, 
        window_radius: int = 3, 
        var_min: float = 0.5, 
        n_jobs: int = -1
        ) -> list[tuple[int, int]]:
    """
    Identify unreliable pixels in a 2D array using Gaussian fitting and local variance.

    Parameters
    ----------
    group_data : np.ndarray
        2D array of data to analyze.
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
    list[tuple[int, int]]
        Sorted list of unique (row, column) indices of unreliable pixels.

    Notes
    ------
    This function first interpolates NaN values in the input data, then prefilters candidates
    based on local variance. It processes each candidate pixel in parallel to fit a Gaussian
    function to the data in a specified search radius. If the fit is reliable, it identifies
    the indices of unreliable pixels based on the full width at half maximum (FWHM) of the
    fitted Gaussian. The function returns a sorted list of unique pixel indices that are
    considered unreliable.

    .. versionadded:: 0.0.2
    """
    group_data = interpolate_nans_2d(group_data.copy())
    candidates: list[tuple[int, int]] = prefilter_candidates(group_data, window_radius, var_min)
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_pixel)(r, c, group_data, search_radius)
        for r, c in candidates
    )

    results_list: list[list[tuple[int, int]]] = [res for res in results if res is not None]
    flattened: list[tuple[int, int]] = [(int(x), int(y)) for sublist in results_list for (x, y) in sublist]
    return sorted(set(flattened))