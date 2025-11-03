"""
This module contains various utility functions required for the relaxation
theory.
"""
# Imports
import numpy as np
import scipy.sparse as sp
from spinguin.la import expm
from spinguin.utils import HidePrints

def auxiliary_matrix_expm(A: np.ndarray | sp.csc_array,
                          B: np.ndarray | sp.csc_array,
                          C: np.ndarray | sp.csc_array,
                          t: float,
                          zero_value: float) -> sp.csc_array:   
    """
    Computes the matrix exponential of an auxiliary matrix. This is used to 
    calculate the Redfield integral.

    Based on Goodwin and Kuprov (Eq. 3): https://doi.org/10.1063/1.4928978
    
    Parameters
    ----------
    A : ndarray or csc_array
        Top-left block of the auxiliary matrix.
    B : ndarray or csc_array
        Top-right block of the auxiliary matrix.
    C : ndarray or csc_array
        Bottom-right block of the auxiliary matrix.
    t : float
        Integration time.
    zero_value : float
        Threshold below which values are considered zero when exponentiating the
        auxiliary matrix using the Taylor series. This significantly impacts
        performance. Use the largest value that still provides correct results.
    
    Returns
    -------
    expm_aux : ndarray or csc_array
        Matrix exponential of the auxiliary matrix. The output is sparse or
        dense matching the sparsity of the input.
    """

    # Ensure that the input arrays are all either sparse or dense
    if not (sp.issparse(A) == sp.issparse(B) == sp.issparse(C)):
        raise ValueError(f"All arrays A, B and C must be of same type.")

    # Are we using sparse?
    sparse = sp.issparse(A)

    # Construct the auxiliary matrix
    if sparse:
        empty_array = sp.csc_array(A.shape)
        aux = sp.block_array([[A, B],
                        [empty_array, C]], format='csc')
    else:
        empty_array = np.zeros(A.shape)
        aux = np.block([[A, B],
                        [empty_array, C]])

    # Compute the matrix exponential of the auxiliary matrix
    with HidePrints():
        expm_aux = expm(aux * t, zero_value)

    return expm_aux