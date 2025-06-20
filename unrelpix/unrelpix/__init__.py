"""
UnrelPix: A package for Python to analyze and process unreliable pixel data
================================================

Subpackages
-----------
::

 identification               --- Identification algorithm
 classification               --- Classification algorithm
 rampfitting                  --- Ramp fitting of modified files
 interpolation                --- Interpolation of ramp data

::
__version__ = "0.0.2"
__author__ = "Fran Stimac"
__all__ = ["identification", "classification", "rampfitting", "interpolation"]

"""

from . import identification
from . import classification
from . import rampfitting
from . import interpolation