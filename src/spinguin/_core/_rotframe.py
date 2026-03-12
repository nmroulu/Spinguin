"""
This module contains functions that are used to transform the Liouvillian into
a rotating frame.
"""
# Type checking
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from spinguin._core._spin_system import SpinSystem

# Imports
import math
import numpy as np
import scipy.sparse as sp
import time
from spinguin._core._hide_prints import HidePrints
from spinguin._core._la import norm_1, expm
from spinguin._core._specutils import resonance_frequency
from spinguin._core._parameters import parameters
from spinguin._core._superoperators import superoperator

def _auxiliary_matrix_rotframe_expm(
    A: np.ndarray | sp.csc_array,
    B: np.ndarray | sp.csc_array,
    T: float,
    order: int
) -> sp.csc_array:
    """
    Computes the matrix exponential of an auxiliary matrix that is used to
    calculate the interaction frame Hamiltonian.

    Based on Goodwin and Kuprov (Eq. 18): https://doi.org/10.1063/1.4928978

    Parameters
    ----------
    A : ndarray or csc_array
        Diagonal elements of the auxiliary matrix (L0)
    B : ndarray or csc_array
        Superdiagonal elements of the auxiliary matrix (L1)
    T : float
        Time
    order : int
        Rotating frame correction order.

    Returns
    -------
    expm_aux : csc_array
        Matrix exponential of the auxiliary matrix. The output is sparse
        regardless of the input array.
    """
    # Convert input arrays to sparse if not already
    if not sp.issparse(A):
        A = sp.csc_array(A)
    if not sp.issparse(B):
        B = sp.csc_array(B)

    # Create a sparse zero-array
    Z = sp.csc_array(A.shape)

    # Construct the auxiliary matrix
    aux = []
    for i in range(order+1):
        row = []
        for j in range(order+1):
            if i == j:
                row.append(A)
            elif i == j-1:
                row.append(B)
            else:
                row.append(Z)
        aux.append(row)
    aux = sp.block_array(aux, format='csc')

    # Compute the matrix exponential of the auxiliary matrix
    with HidePrints():
        expm_aux = expm(T*aux, parameters.zero_aux)

    return expm_aux

def rotating_frame(
    spin_system: SpinSystem,
    L: np.ndarray | sp.csc_array,
    isotopes: list,
    center_frequencies: list = [],
    orders: list = []
) -> np.ndarray | sp.csc_array:
    """
    Transforms the Liouvillian into the rotating frame.

    Parameters
    ----------
    spin_system : SpinSystem
        Spin system whose Liouvillian is going to be transformed.
    L : ndarray or csc_array
        Liouvillian superoperator in the laboratory frame.
    isotopes : list
        List of isotopes whose rotating frames are applied.
    center_frequencies : list, default=[]
        List of center frequencies (in ppm) for each isotope. If empty, zero is
        used for all isotopes.
    orders : list, default=[]
        List of integers that define the order of the rotating frame for each
        isotope. If empty, the default value defined in
        `parameters.rotating_frame_order` is used for all isotopes.

    Returns
    -------
    L_rot : ndarray or csc_array
        Liouvillian superoperator in the rotating frame.
    """
    print("Transforming Liouvillian to the rotating frame...")
    time_start = time.time()

    # Check list lengths
    if len(isotopes) == 0:
        raise ValueError("must specify at least one isotope")
    if not (len(orders) == len(isotopes) or len(orders) == 0):
        raise ValueError("lengths of orders and isotopes must match")
    if not (
        len(center_frequencies) == len(isotopes) or
        len(center_frequencies) == 0
    ):
        raise ValueError("center_frequencies and isotopes must have same length")
    
    # Check that the isotopes exist in the spin system
    for isotope in isotopes:
        if isotope not in spin_system.isotopes:
            raise ValueError(f"isotope {isotope} is not in the spin system")
        
    # Check that each given isotope is unique
    if not len(isotopes) == len(set(isotopes)):
        raise ValueError("given isotopes must be unique")
    
    # Fill input lists with default values
    if len(orders) == 0:
        orders = [parameters.rotating_frame_order for _ in range(len(isotopes))]
    if len(center_frequencies) == 0:
        center_frequencies = [0 for _ in range(len(isotopes))]

    # Frequencies for the interaction frames
    freqs = []
    for i in range(len(isotopes)):
        freq = resonance_frequency(
            isotope = isotopes[i], 
            delta = center_frequencies[i],
            unit = "rad/s"
        )
        freqs.append(freq)

    # Corresponding Hamiltonians
    H0s = []
    for i in range(len(isotopes)):
        spins = np.where(spin_system.isotopes == isotopes[i])[0]
        dim = spin_system.basis.dim
        if parameters.sparse_superoperator:
            H = sp.csc_array((dim, dim))
        else:
            H = np.zeros((dim, dim))
        for n in spins:
            Iz_n = superoperator(spin_system, f"I(z, {n})")
            H = H + freqs[i] * Iz_n
        H0s.append(H)

    # Corresponding Liouvillians
    L0s = []
    for H in H0s:
        L0s.append(-1j*H)

    # Norms of the Liouvillians
    norms = []
    for L0 in L0s:
        norms.append(norm_1(L0))

    # Re-order based on the norms
    sort = np.flip(np.argsort(norms))
    freqs = [freqs[i] for i in sort]
    orders = [orders[i] for i in sort]
    L0s = [L0s[i] for i in sort]
    isotopes = [isotopes[i] for i in sort]

    # Calculate the periods
    Ts = []
    for freq in freqs:
        Ts.append(2*np.pi / freq)

    # Apply each rotating frame
    for i in range(len(L0s)):
        print(f"\tApplying rotating frame for {isotopes[i]}...")
        L0 = L0s[i]
        L1 = L - L0
        T = Ts[i]
        order = orders[i]
        L = _sop_L_to_rotframe(L0, L1, T, order)

    print(f"Completed in {time.time() - time_start:.4f} seconds.")

    return L

def _sop_L_to_rotframe(
    L0: np.ndarray | sp.csc_array,
    L1: np.ndarray | sp.csc_array,
    T: float,
    order: int
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

    Returns
    -------
    L_rot : ndarray or csc_array
        Total Liouvillian in the rotating frame.
    """
    # Obtain the auxiliary matrix (returns a sparse array)
    aux = _auxiliary_matrix_rotframe_expm(L0, L1, T, order)

    # Extract the derivatives
    D = []
    for i in range(order+1):
        aux_element = aux[:L0.shape[0], i*L0.shape[0]:(i+1)*L0.shape[0]]
        if not parameters.sparse_superoperator:
            aux_element = aux_element.toarray()
        D.append(math.factorial(i)*aux_element)

    # Initialise the result
    if parameters.sparse_superoperator:
        L_rot = sp.csc_array(L0.shape)
    else:
        L_rot = np.zeros(shape=L0.shape)

    # Build the rotating frame Liouvillian (Eq. 16)
    for n in range(1, order+1):
        for k in range(1, n+1):

            # Get the denominator in a numerically stable manner
            denom = T * n * math.factorial(k - 1) * math.factorial(n - k)

            # Add to the existing term
            L_rot = L_rot + D[n-k].conj().T @ D[k] / denom

    return L_rot