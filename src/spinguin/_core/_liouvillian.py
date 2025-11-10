"""
This module provides a function for calculating the Liouvillian.
"""

# Imports
import numpy as np
import scipy.sparse as sp

def liouvillian(
    H: np.ndarray | sp.csc_array = None,
    R: np.ndarray | sp.csc_array = None,
    K: np.ndarray | sp.csc_array = None
) -> np.ndarray | sp.csc_array:
    """
    Constructs the Liouvillian superoperator from the Hamiltonian, relaxation
    superoperator, and exchange superoperator.

    Parameters
    ----------
    H : ndarray or csc_array
        Hamiltonian superoperator.
    R : ndarray or csc_array
        Relaxation superoperator
    K : ndarray or csc_array
        Exchange superoperator.

    Returns
    -------
    L : ndarray or csc_array
        Liouvillian superoperator.
    """

    # Check for totally empty input
    if H is None and R is None and K is None:
        raise ValueError("H, R and K cannot all be None simultaneously.")

    # Assign zeroes if None
    if H is None:
        H = 0
    if R is None:
        R = 0
    if K is None:
        K = 0

    # Construct the Liouvillian
    L = -1j*H - R + K

    return L