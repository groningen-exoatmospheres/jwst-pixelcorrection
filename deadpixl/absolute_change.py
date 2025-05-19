import numpy as np

def absolute_change(a, b):
    """ Find the change in the order of magnitude between two values."""
    return np.abs(np.log10(abs(a)) - np.log10(abs(b)))
