"""
Rotating-frame transformation utilities.

The module contains helper functions for transforming Liouvillian
superoperators into one or more rotating frames.
"""

from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING

import numpy as np
import scipy.sparse as sp

from spinguin._core._hide_prints import HidePrints
from spinguin._core._la import expm, norm_1
from spinguin._core._parameters import parameters
from spinguin._core._specutils import resonance_frequency
from spinguin._core._status import status
from spinguin._core._superoperators import superoperator
from spinguin._core._validation import require

if TYPE_CHECKING:
    from spinguin._core._spin_system import SpinSystem


def _auxiliary_matrix_rotframe_expm(
    A: np.ndarray | sp.csc_array,
    B: np.ndarray | sp.csc_array,
    T: float,
    order: int,
) -> sp.csc_array:
    """
    Compute the auxiliary-matrix exponential used in rotating-frame
    transformations.

    Based on Goodwin and Kuprov (Eq. 18): https://doi.org/10.1063/1.4928978

    Parameters
    ----------
    A : ndarray or csc_array
        Diagonal elements of the auxiliary matrix ``L0``.
    B : ndarray or csc_array
        Superdiagonal elements of the auxiliary matrix ``L1``.
    T : float
        Interaction-frame period.
    order : int
        Rotating frame correction order.

    Returns
    -------
    expm_aux : csc_array
        Matrix exponential of the auxiliary matrix. The output is sparse
        regardless of the input array.
    """

    # Convert the input blocks to sparse format if needed.
    if not sp.issparse(A):
        A = sp.csc_array(A)
    if not sp.issparse(B):
        B = sp.csc_array(B)

    # Construct the sparse zero block used in the auxiliary matrix.
    Z = sp.csc_array(A.shape)

    # Assemble the block auxiliary matrix.
    aux = []
    for i in range(order + 1):
        row = []
        for j in range(order + 1):
            if i == j:
                row.append(A)
            elif i == j - 1:
                row.append(B)
            else:
                row.append(Z)
        aux.append(row)
    aux = sp.block_array(aux, format="csc")

    # Evaluate the auxiliary-matrix exponential while silencing nested output.
    with HidePrints():
        expm_aux = expm(T * aux, parameters.zero_aux)

    return expm_aux

def rotating_frame(
    spin_system: SpinSystem,
    L: np.ndarray | sp.csc_array,
    isotopes: list[str],
    center_frequencies: list[float]=[],
    orders: list[int]=[],
) -> np.ndarray | sp.csc_array:
    """
    Transform a Liouvillian into one or more rotating frames.

    Parameters
    ----------
    spin_system : SpinSystem
        Spin system whose Liouvillian is going to be transformed.
    L : ndarray or csc_array
        Liouvillian superoperator in the laboratory frame.
    isotopes : list of str
        List of isotopes whose rotating frames are applied.
    center_frequencies : list of float, default=[]
        List of centre frequencies in ppm for each isotope. If empty,
        zero is used for all isotopes.
    orders : list of int, default=[]
        List of integers that define the order of the rotating frame for each
        isotope. If empty, the default value defined in
        ``parameters.rotating_frame_order`` is used for all isotopes.

    Returns
    -------
    L_rot : ndarray or csc_array
        Liouvillian superoperator in the rotating frame.
    """

    # Report the start of the rotating-frame transformation.
    status("Transforming Liouvillian to the rotating frame...")
    time_start = time.time()

    # Ensure that the working basis has been built.
    require(spin_system, "basis.basis", "transforming to the rotating frame")

    # Validate the basic input list lengths.
    if len(isotopes) == 0:
        raise ValueError("Must specify at least one isotope.")
    if len(orders) not in (0, len(isotopes)):
        raise ValueError("Lengths of orders and isotopes must match.")
    if not (
        len(center_frequencies) == len(isotopes) or
        len(center_frequencies) == 0
    ):
        raise ValueError(
            "Lengths of centre frequencies and isotopes must match."
        )
    
    # Check that each requested isotope exists in the spin system.
    for isotope in isotopes:
        if isotope not in spin_system.isotopes:
            raise ValueError(
                f"Isotope {isotope} is not present in the spin system."
            )
        
    # Check that the requested isotope labels are unique.
    if len(isotopes) != len(set(isotopes)):
        raise ValueError("Given isotopes must be unique.")

    # Fill optional inputs with default values when needed.
    if len(orders) == 0:
        orders = [parameters.rotating_frame_order for _ in range(len(isotopes))]
    if len(center_frequencies) == 0:
        center_frequencies = [0 for _ in range(len(isotopes))]

    # Calculate the resonance frequencies that define the interaction frames.
    freqs = [
        resonance_frequency(
            isotope=isotope,
            delta=center_frequency,
            unit="rad/s",
        )
        for isotope, center_frequency in zip(isotopes, center_frequencies)
    ]

    # Build the Hamiltonians associated with the requested rotating frames.
    H0s = []
    dim = spin_system.basis.dim
    for isotope, freq in zip(isotopes, freqs):

        # Identify the spins that belong to the current isotope.
        spins = np.where(spin_system.isotopes == isotope)[0]

        # Allocate the Hamiltonian contribution for the current frame.
        if parameters.sparse_superoperator:
            H = sp.csc_array((dim, dim))
        else:
            H = np.zeros((dim, dim))

        # Add the z-Hamiltonian term of each spin of the current isotope.
        for n in spins:
            Iz_n = superoperator(spin_system, f"I(z, {n})")
            H = H + freq * Iz_n
        H0s.append(H)

    # Convert the frame Hamiltonians to Liouvillians.
    L0s = [-1j * H for H in H0s]

    # Calculate the norms used to determine the transformation order.
    norms = [norm_1(L0) for L0 in L0s]

    # Reorder the transformations from largest to smallest norm.
    sort = np.flip(np.argsort(norms))
    freqs = [freqs[i] for i in sort]
    orders = [orders[i] for i in sort]
    L0s = [L0s[i] for i in sort]
    isotopes = [isotopes[i] for i in sort]

    # Convert the frequencies to periods.
    Ts = [2 * np.pi / freq for freq in freqs]

    # Apply the requested rotating-frame transformations sequentially.
    for isotope, L0, T, order in zip(isotopes, L0s, Ts, orders):
        status(f"Applying rotating frame for {isotope}...")
        L1 = L - L0
        L = _sop_L_to_rotframe(L0, L1, T, order)

    # Report the completion of the rotating-frame transformation.
    status(f"Completed in {time.time() - time_start:.4f} seconds.\n")

    return L

def _sop_L_to_rotframe(
    L0: np.ndarray | sp.csc_array,
    L1: np.ndarray | sp.csc_array,
    T: float,
    order: int,
) -> np.ndarray | sp.csc_array:
    """
    Convert ``L = L0 + L1`` into the rotating frame defined by ``L0``.

    Based on Goodwin and Kuprov (Eq. 16): https://doi.org/10.1063/1.4928978

    Parameters
    ----------
    L0 : ndarray or csc_array
        Liouvillian containing the largest interaction (must be a single
        frequency).
    L1 : ndarray or csc_array
        Rest of the Liouvillian. Must be a perturbation:
        ``||L1|| < ||L0||``.
    T : float
        Period of the Liouvillian, for which ``expm(L0*T) = I``.
    order : int
        Order to which the series is truncated.

    Returns
    -------
    L_rot : ndarray or csc_array
        Total Liouvillian in the rotating frame.
    """

    # Evaluate the auxiliary-matrix exponential used in the expansion.
    aux = _auxiliary_matrix_rotframe_expm(L0, L1, T, order)

    # Extract the derivative-like blocks from the auxiliary matrix.
    D = []
    for i in range(order + 1):
        aux_element = aux[
            :L0.shape[0],
            i * L0.shape[0]:(i + 1) * L0.shape[0],
        ]
        if not parameters.sparse_superoperator:
            aux_element = aux_element.toarray()
        D.append(math.factorial(i) * aux_element)

    # Allocate the rotating-frame Liouvillian.
    if parameters.sparse_superoperator:
        L_rot = sp.csc_array(L0.shape)
    else:
        L_rot = np.zeros(shape=L0.shape)

    # Assemble the rotating-frame Liouvillian according to Eq. 16.
    for n in range(1, order + 1):
        for k in range(1, n + 1):

            # Build the denominator of the current series contribution.
            denom = T * n * math.factorial(k - 1) * math.factorial(n - k)

            # Accumulate the current contribution to the rotating-frame series.
            L_rot = L_rot + D[n-k].conj().T @ D[k] / denom

    return L_rot