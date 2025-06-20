"""
Ramp fitting on modified files (:mod:`unrelpix.rampfitting`)
=====================================================

.. sectionauthor:: Fran Stimac

Function reference
------------------

This module provides functionality to run the JWST Ramp Fitting step on modified ramp files.

.. autosummary::
    :toctree: generated/

    run_ramp_fitting -- run the JWST Ramp Fitting step on modified ramp files
"""

import os
import warnings
import requests

warnings.simplefilter("ignore", RuntimeWarning)

# Set CRDS cache directory to user home if not already set.
if os.getenv('CRDS_PATH') is None:
    os.environ['CRDS_PATH'] = os.path.join(os.path.expanduser('~'), 'crds_cache')

# Check whether the CRDS server URL has been set. If not, set it.
if os.getenv('CRDS_SERVER_URL') is None:
    os.environ['CRDS_SERVER_URL'] = 'https://jwst-crds.stsci.edu'

# Output the current CRDS path and server URL in use.
print('CRDS local filepath:', os.environ['CRDS_PATH'])
print('CRDS file server:', os.environ['CRDS_SERVER_URL'])

import os
from jwst.ramp_fitting import RampFitStep


def run_ramp_fitting(
        modified_file: str, 
        output_dir: str = "./"
        ) -> str:
    """
    Run the JWST Ramp Fitting step on a list of modified ramp fils.

    Parameters
    ----------
    modified_file : str
        File paths to the modified ramp file to be processed.
    output_dir : str, optional
        Directory where the output file will be saved. Default is the current directory.

    Returns
    -------
    str
        The output directory where the ramp-fitted file are saved.

    Notes
    -----
    This function initializes the `RampFitStep` from the JWST pipeline, processes each input file,
    and saves the resulting integration cubes to the specified output directory.

    .. versionadded:: 0.0.2
    """

    os.makedirs(output_dir, exist_ok=True)

    # Initialize the RampFitStep
    ramp_fit = RampFitStep()

    # Optional: Configure parameters
    ramp_fit.save_results = True  # Save the output to a file
    ramp_fit.maximum_cores = 'all'  # Utilize all available cores for multiprocessing

    # Execute the ramp fitting process
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    result = ramp_fit.call(modified_file, output_dir=output_dir)

    # The result is a tuple: (ImageModel, CubeModel)
    _, integration_cube = result

    # Save the outputs if needed
    integration_cube.save(
        os.path.join(
            output_dir,
            f'{os.path.basename(modified_file).replace(".fits", "_rampfitted.fits")}'
        )
    )

    print(f"Ramp fitting completed for {modified_file}. Output saved to {output_dir}.")

    return output_dir     