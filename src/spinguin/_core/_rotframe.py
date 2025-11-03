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
from spinguin.la import norm_1, expm
from spinguin.utils import HidePrints
from spinguin._core._config import config
from spinguin._core._parameters import parameters
from spinguin._core._superoperators import superoperator

def _auxiliary_matrix_rotframe_expm(
    A: np.ndarray | sp.csc_array,
    B: np.ndarray | sp.csc_array,
    T: float,
    dim: int,
    zero_value: float
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
    dim : int
        Dimension of the auxiliary matrix.
    zero_value : float
        Threshold below which values are considered zero when exponentiating the
        auxiliary matrix using the Taylor series. This significantly impacts
        performance. Use the largest value that still provides correct results.

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
    Z = sp.csc_array((A.shape[0], B.shape[0]))

    # Construct the auxiliary matrix
    aux = []
    for i in range(dim):
        row = []
        for j in range(dim):
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
        expm_aux = expm(T*aux, zero_value)

    return expm_aux

def rotating_frame(
    spin_system: SpinSystem,
    L: np.ndarray | sp.csc_array,
    isotopes: list,
    orders: list = [],
    center_frequencies: list = [],
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
    orders : list, default=[]
        List of integers that define the order of the rotating frame for each
        isotope. If empty, the default value defined in
        `parameters.rotating_frame_order` is used for all isotopes.
    center_frequencies : list, default=[]
        List of center frequencies (in ppm) for each isotope. If empty, zero is
        used for all isotopes.

    Returns
    -------
    L_rot : ndarray or csc_array
        Liouvillian superoperator in the rotating frame.
    """
    # Check input types
    if not isinstance(spin_system, SpinSystem):
        raise ValueError("spin_system must be a SpinSystem object")
    if not isinstance(L, (np.ndarray, sp.csc_array)):
        raise ValueError("L must be NumPy array or SciPy CSC array")
    if not isinstance(isotopes, list):
        raise ValueError("isotopes must be a list")
    if not isinstance(orders, list):
        raise ValueError("orders must be a list")
    if not isinstance(center_frequencies, list):
        raise ValueError("center_frequencies must be a list")
    
    # Check list lengths
    if len(isotopes) == 0:
        raise ValueError("isotopes cannot be an empty list")
    if not (len(orders) == len(isotopes) or len(orders) == 0):
        raise ValueError(
            "orders must have the same length as isotopes or be empty")
    if not (len(center_frequencies) == len(isotopes) or
            len(center_frequencies) == 0):
        raise ValueError("center_frequencies must have the same length as "
                         "isotopes or be empty")
    
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
        freq = resonance_frequency(isotopes[i], center_frequencies[i], "rad/s")
        freqs.append(freq)

    # Corresponding Hamiltonians
    hamiltonians = []
    for i in range(len(isotopes)):
        spins = np.where(spin_system.isotopes == isotopes[i])[0]
        dim = spin_system.basis.dim
        hamiltonian = sp.csc_array((dim, dim))
        for n in spins:
            Iz_n = superoperator(spin_system, f"I(z, {n})")
            hamiltonian = hamiltonian + freqs[i] * Iz_n
        hamiltonians.append(hamiltonian)

    # Corresponding Liouvillians
    L0s = []
    for hamiltonian in hamiltonians:
        L0s.append(-1j*hamiltonian)

    # Norms of the Liouvillians
    norms = []
    for L0 in L0s:
        norms.append(norm_1(L0))

    # Re-order based on the norms
    sort = np.argsort(norms)
    freqs = [freqs[i] for i in sort]
    orders = [orders[i] for i in sort]
    L0s = [L0s[i] for i in sort]

    # Calculate the periods
    Ts = []
    for freq in freqs:
        Ts.append(2*np.pi / freq)

    # Apply each rotating frame
    for i in range(len(L0s)):
        L0 = L0s[i]
        L1 = L - L0
        T = Ts[i]
        order = orders[i]
        L = _sop_L_to_rotframe(L0, L1, T, order, config.zero_aux_rotframe)

    return L

def _sop_L_to_rotframe(
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
    sparse = sp.issparse(L0)

    # Obtain the auxiliary matrix
    dim = order+1
    aux = _auxiliary_matrix_rotframe_expm(L0, L1, T, dim, zero_value)

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