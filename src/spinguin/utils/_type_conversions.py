# Imports
import numpy as np
from numpy.typing import ArrayLike

def arraylike_to_array(A: ArrayLike) -> np.ndarray:
    """
    Converts an `ArrayLike` object into a NumPy array while ensuring
    that at least one dimension is created.

    Parameters
    ----------
    A : ArrayLike
        An object that can be converted into NumPy array.

    Returns
    -------
    A : ndarray
        The original object converted into a NumPy array.
    """

    # Convert to NumPy array and ensure at least one dimension
    A = np.asarray(A)
    A = np.atleast_1d(A)

    return A

def arraylike_to_tuple(A: ArrayLike) -> tuple:
    """
    Converts a 1-dimensional `ArrayLike` object into a Python tuple.

    Parameters
    ----------
    A : ArrayLike
        An object that can be converted into NumPy array.

    Returns
    -------
    A : tuple
        The original object represented as Python tuple.
    """

    # Convert to tuple
    A = np.asarray(A)
    if A.ndim == 0:
        A = tuple([A.item()])
    elif A.ndim == 1:
        A = tuple(A)
    else:
        raise ValueError(f"Cannot convert {A.ndim}-dimensional array into "
                         "tuple.")
    
    return A