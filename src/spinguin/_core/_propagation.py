"""
This module is responsible for calculating time propagators.
"""

# Imports
import time
import numpy as np
import scipy.sparse as sp
import warnings
from spinguin._core._la import expm
from spinguin._core._superoperators import sop_from_string
from spinguin._core._hide_prints import HidePrints

def sop_propagator(L: np.ndarray | sp.csc_array,
                   t: float,
                   zero_value: float=1e-18,
                   density_threshold: float=0.5) -> sp.csc_array | np.ndarray:
    """
    Constructs the time propagator exp(L*t).

    Parameters
    ----------
    L : ndarray or csc_array
        Liouvillian superoperator, L = -iH - R + K.
    t : float
        Time step of the simulation in seconds.
    zero_value : float, default=1e-18
        Calculating the propagator involves a matrix exponential, which is
        calculated using the scaling and squaring method together with Taylor
        series. This threshold is used to estimate the convergence of the Taylor
        series and to eliminate small values during the squaring step.
    density_threshold : float, default=0.5
        Sparse matrix is returned if the density is less than this threshold.
        Otherwise dense matrix is returned.

    Returns
    -------
    expm_Lt : csc_array or ndarray
        Time propagator exp(L*t).
    """

    print("Constructing propagator...")
    time_start = time.time()

    # Compute the matrix exponential
    expm_Lt = expm(L * t, zero_value)

    # Calculate the density of the propagator
    density = expm_Lt.nnz / (expm_Lt.shape[0] ** 2)
    print(f"Propagator density: {density:.4f}")

    # Convert to NumPy array if density exceeds the threshold
    if density > density_threshold:
        print("Density exceeds threshold. Converting to NumPy array.")
        expm_Lt = expm_Lt.toarray()

    print(f'Propagator constructed in {time.time() - time_start:.4f} seconds.')
    print()

    return expm_Lt

def propagator_to_rotframe(
        sop_P: np.ndarray | sp.csc_array,
        sop_H0: np.ndarray | sp.csc_array,
        t: float,
        zero_value: float=1e-18
) -> np.ndarray | sp.csc_array:
    """
    Transforms the time propagator to the rotating frame.

    Parameters
    ----------
    sop_P : ndarray or csc_array
        Time propagator.
    sop_H0 : ndarray or csc_array
        Hamiltonian superoperator representing the interaction used
        to define the rotating frame transformation.
    t : float
        Time step of the simulation in seconds.
    zero_value : float, default=1e-18
        Calculating the rotating frame transformation involves a matrix
        exponential, which is calculated using the scaling and squaring method
        together with Taylor series. This threshold is used to estimate the
        convergence of the Taylor series and to eliminate small values during
        the squaring step.

    Returns
    -------
    sop_P : ndarray or csc_array
        The time propagator transformed into the rotating frame.
    """

    print("Applying rotating frame transformation...")
    time_start = time.time()

    # Acquire matrix exponential from the Hamiltonian
    with HidePrints():
        expm_H0t = expm(1j * sop_H0 * t, zero_value)

    # Convert the time propagator to rotating frame
    sop_P = expm_H0t @ sop_P

    print("Rotating frame transformation applied in "
          f"{time.time() - time_start:.4f} seconds.")
    print()

    return sop_P

def sop_pulse(basis: np.ndarray,
              spins: np.ndarray,
              operator: str,
              angle: float,
              sparse: bool=True,
              zero_value: float=1e-18) -> np.ndarray | sp.csc_array:
    """
    Generates a superoperator corresponding to the pulse described
    by the given operator and angle.

    Parameters
    ----------
    basis : ndarray
        A 2-dimensional array containing the basis set that contains sequences
        of integers describing the Kronecker products of irreducible spherical
        tensors.
    spins : ndarray
        A 1-dimensional array containing the spin quantum numbers of each spin.
    operator : str
        Defines the pulse to be generated. The operator string must
        follow the rules below:

        - Cartesian and ladder operators: `I(component,index)` or
          `I(component)`. Examples:

            - `I(x,4)` --> Creates x-operator for spin at index 4.
            - `I(x)`--> Creates x-operator for all spins.

        - Spherical tensor operators: `T(l,q,index)` or `T(l,q)`. Examples:

            - `T(1,-1,3)` --> \
              Creates operator with `l=1`, `q=-1` for spin at index 3.
            - `T(1, -1)` --> \
              Creates operator with `l=1`, `q=-1` for all spins.
            
        - Product operators have `*` in between the single-spin operators:
          `I(z,0) * I(z,1)`
        - Sums of operators have `+` in between the operators:
          `I(x,0) + I(x,1)`
        - Unit operators are ignored in the input. Interpretation of these
          two is identical: `E * I(z,1)`, `I(z,1)`
        
        Special case: An empty `operator` string is considered as unit operator.

        Whitespace will be ignored in the input.

        NOTE: Indexing starts from 0!
    angle : float
        Pulse angle in degrees.
    sparse : bool, default=True
        Specifies whether to construct the pulse superoperator as sparse or
        dense array.
    zero_value : float, default=1e-18
        Calculating the pulse superoperator involves a matrix exponential, which
        is calculated using the scaling and squaring method together with Taylor
        series. This threshold is used to estimate the convergence of the Taylor
        series and to eliminate small values during the squaring step.

    Returns
    -------
    pul : ndarray or csc_array
        Superoperator corresponding to the applied pulse.
    """

    time_start = time.time()
    print("Creating a pulse superoperator...")

    # Show a warning if pulse is generated using a product operator
    if '*' in operator:
        warnings.warn("Applying a pulse using a product operator does not have "
                      "a well-defined angle.")

    # Generate the operator
    op = sop_from_string(operator, basis, spins, side="comm", sparse=sparse)

    # Convert the angle to radians
    angle = angle / 180 * np.pi

    # Construct the pulse propagator
    with HidePrints():
        pul = expm(-1j * angle * op, zero_value)

    print(f'Pulse constructed in {time.time() - time_start:.4f} seconds.\n')

    return pul