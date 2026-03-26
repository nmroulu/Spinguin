"""
Construction of the Liouvillian superoperator.

This module provides a small helper for combining Hamiltonian, relaxation, and
exchange contributions into a single Liouvillian.
"""

# Imports
import numpy as np
from scipy.sparse import csc_array

# Define a type alias for the allowed input types of the superoperators.
DenseOrSparse = np.ndarray | csc_array

def liouvillian(
    H: DenseOrSparse=None,
    R: DenseOrSparse=None,
    K: DenseOrSparse=None,
) -> DenseOrSparse:
    """
    Construct the Liouvillian superoperator.

    The Liouvillian is assembled according to

    $$
    L = -iH - R + K,
    $$

    where `H` is the Hamiltonian superoperator, `R` is the relaxation
    superoperator, and `K` is the exchange superoperator.

    Parameters
    ----------
    H : ndarray or csc_array, optional
        Hamiltonian superoperator.
    R : ndarray or csc_array, optional
        Relaxation superoperator.
    K : ndarray or csc_array, optional
        Exchange superoperator.

    Returns
    -------
    L : ndarray or csc_array
        Liouvillian superoperator.

    Raises
    ------
    ValueError
        Raised if `H`, `R`, and `K` are all `None`.
    """

    # Require at least one physical contribution for the Liouvillian.
    if H is None and R is None and K is None:
        raise ValueError("H, R and K cannot all be None simultaneously.")

    # Replace omitted contributions by zero so that the algebra below remains
    # uniform.
    if H is None:
        H = 0
    if R is None:
        R = 0
    if K is None:
        K = 0

    # Combine the three superoperator contributions into the Liouvillian.
    L = -1j * H - R + K
    return L