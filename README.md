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
---
Tested on Python 3.13, should work on 3.10, 3.11, 3.12.
