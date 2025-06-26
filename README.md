## unrelpix: Python module to identify, classify, and interpolate unreliable pixels in NIRSpec data
This module is designed to work with NIRSpec data, specifically to identify and handle unreliable pixels. It provides functionality to classify these pixels based on their reliability and to interpolate over them as needed. 


### Installation
----
To install the `unrelpix` module, you can use pip. The recommended way is to install it in editable mode, which allows you to make changes to the code without needing to reinstall the package each time.

```bash
pip install git+https://github.com/groningen-exoatmospheres/jwst-pixelcorrection.git
```
or
```bash
pip install -e git+https://github.com/groningen-exoatmospheres/jwst-pixelcorrection.git
```
### Usage and purpose
----
The `unrelpix` module is intended for use in the analysis of NIRSpec data, particularly in identifying and handling unreliable pixels.

Taking the `ramp` files, the module identifies additional unreliable pixels missed by the default `jwst.pipeline` settings. The classification of additional pixels is taken from the JWST pipeline documentation (https://jwst-pipeline.readthedocs.io/en/latest/jwst/references_general/references_general.html#data-quality-flags). Running the `ramp` files again through the `RampFitStep` with the additional unreliable pixels identified, the new `rateints` files are then interpolated using `unrelpix.interpolation.interpolate_fits_files` to fill in the gaps left by the unreliable pixels. See example Jupyter notebook in `notebook_examples/` directory for a demonstration of how to use the module to improve an exoplanet lightcurve.

----
### Requirements
- `jwst` locally installed (e.g., via `pip install jwst`)

Tested on Python 3.13, should work on 3.10, 3.11, 3.12.
