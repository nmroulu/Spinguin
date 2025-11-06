# Type hints
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import scipy.sparse as sp

# Imports
import numpy as np
from io import BytesIO
from numpy.typing import ArrayLike
from scipy.io import mmwrite, mmread

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

def bytes_to_sparse(A_bytes: bytes) -> sp.csc_array:
    """
    Converts a byte representation back to a SciPy sparse array.

    Parameters
    ----------
    A_bytes : bytes
        Byte representation of a SciPy sparse array.

    Returns
    -------
    A : csc_array
        Sparse array reconstructed from the byte representation.
    """

    # Initialize a BytesIO object
    bytes_io = BytesIO(A_bytes)

    # Read the SciPy sparse array from bytes
    A = mmread(bytes_io)

    return A

def sparse_to_bytes(A: sp.csc_array) -> bytes:
    """
    Converts the given SciPy sparse array into a byte representation.

    Parameters
    ----------
    A : csc_array
        Sparse matrix to be converted into bytes.

    Returns
    -------
    A_bytes : bytes
        Byte representation of the input matrix.
    """
    
    # Initialize a BytesIO object
    bytes_io = BytesIO()

    # Write the matrix A to bytes
    mmwrite(bytes_io, A)

    # Retrieve the bytes
    A_bytes = bytes_io.getvalue()

    return A_bytes