"""
This module provides a function for calculating the Liouvillian.
"""

# Imports
import numpy as np
import scipy.sparse as sp
import math
from spinguin.la import auxiliary_matrix_rotframe_expm, issparse

def sop_L_to_rotframe(
    L0: np.ndarray | sp.csc_array,
    L1: np.ndarray | sp.csc_array,
    T: float,
    order: int,
    zero_value: float
) -> np.ndarray | sp.csc_array:
    """
    Converts the input Liouvillian `L = L0 + L1` into a rotating frame defined
    by `L0`.

    Based on Goodwin and Kuprov (Eq. 16): https://doi.org/10.1063/1.4928978

    Parameters
    ----------
    L0 : ndarray or csc_array
        Liouvillian containing the largest interaction (must be a single
        frequency).
    L1 : ndarray or csc_array
        Rest of the Liouvillian. Must be a perturbation: `||L1|| < ||L0||`.
    T : float
        Period of the Liouvillian: `expm(L0*T) = I`.
    order : int
        Order to which the series is truncated.
    zero_value : float
        Numerical accuracy of the auxiliary matrix method.

    Returns
    -------
    L_rot : ndarray or csc_array
        Total Liouvillian in the rotating frame.
    """
    # Based on the input type, choose whether to use sparses
    sparse = issparse(L0)

    # Obtain the auxiliary matrix
    dim = order+1
    aux = auxiliary_matrix_rotframe_expm(L0, L1, T, dim, zero_value)

    # Extract the derivatives
    D = []
    for i in range(dim):
        aux_element = aux[:L0.shape[0], i*L0.shape[0]:(i+1)*L0.shape[0]]
        if not sparse:
            aux_element = aux_element.toarray()
        D.append(math.factorial(i)*aux_element)

    # Initialise the result
    if sparse:
        L_rot = sp.csc_array((L0.shape[0], L0.shape[0]))
    else:
        L_rot = np.zeros(shape=L0.shape)

    # Build the rotating frame Liouvillian
    for n in range(order):
        for k in range(n+1):
            denominator = T * (n+1) * math.factorial(k) * math.factorial(n-k)
            L_rot = L_rot + D[n-k].conj().T @ D[k+1] / denominator

    return L_rot


def sop_L(H: np.ndarray | sp.csc_array = None,
          R: np.ndarray | sp.csc_array = None,
          K: np.ndarray | sp.csc_array = None) -> np.ndarray | sp.csc_array:
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