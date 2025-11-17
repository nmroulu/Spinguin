"""
This module is responsible for calculating time propagators.
"""
# Referencing SpinSystem class
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from spinguin._core._spin_system import SpinSystem

# Imports
import time
import numpy as np
import scipy.sparse as sp
import warnings
from spinguin._core._la import expm
from spinguin._core._superoperators import sop_from_string
from spinguin._core._hide_prints import HidePrints
from spinguin._core._hamiltonian import sop_H
from spinguin._core._parameters import parameters

def propagator(
    L: np.ndarray | sp.csc_array,
    t: float
) -> np.ndarray | sp.csc_array:
    """
    Constructs the time propagator exp(L*t).

    Parameters
    ----------
    L : ndarray or csc_array
        Liouvillian superoperator, L = -iH - R + K.
    t : float
        Time step of the simulation in seconds.

    Returns
    -------
    P : csc_array or ndarray
        Time propagator exp(L*t).
    """

    print("Constructing propagator...")
    time_start = time.time()

    # Compute the matrix exponential
    P = expm(L * t, parameters.zero_propagator)

    # Calculate the density of the propagator
    density = P.nnz / (P.shape[0] ** 2)
    print(f"Propagator density: {density:.4f}")

    # Convert to NumPy array if density exceeds the threshold
    if density > parameters.propagator_density:
        print("Density exceeds threshold. Converting to NumPy array.")
        P = P.toarray()

    print(f'Propagator constructed in {time.time() - time_start:.4f} seconds.')
    print()

    return P

def pulse(
    spin_system: SpinSystem,
    operator: str,
    angle: float
) -> np.ndarray | sp.csc_array:
    """
    Creates a pulse superoperator that is applied to a state by multiplying
    from the left.

    Parameters
    ----------
    spin_system : SpinSystem
        Spin system for which the pulse superoperator is going to be created.
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

    Returns
    -------
    P : ndarray or csc_array
        Pulse superoperator.
    """

    # Check that the required attributes have been set
    if spin_system.basis.basis is None:
        raise ValueError("Please build the basis before constructing pulse "
                         "superoperators.")

    time_start = time.time()
    print("Creating a pulse superoperator...")

    # Show a warning if pulse is generated using a product operator
    if '*' in operator:
        warnings.warn("Applying a pulse using a product operator does not have "
                      "a well-defined angle.")

    # Generate the operator
    op = sop_from_string(
        operator,
        spin_system.basis.basis,
        spin_system.spins,
        side="comm"
    )

    # Convert the angle to radians
    angle = angle / 180 * np.pi

    # Construct the pulse propagator
    with HidePrints():
        P = expm(-1j * angle * op, parameters.zero_pulse)

    print(f'Pulse constructed in {time.time() - time_start:.4f} seconds.\n')

    return P

def propagator_to_rotframe(
    spin_system: SpinSystem,
    P: np.ndarray | sp.csc_array,
    t: float,
    center_frequencies: dict=None
) -> np.ndarray | sp.csc_array:
    """
    Transforms the time propagator to the rotating frame.

    Parameters
    ----------
    spin_system : SpinSystem
        Spin system whose time propagator is going to be transformed.
    P : ndarray or csc_array
        Time propagator in the laboratory frame.
    t : float
        Time step of the propagator in seconds.
    center_frequencies : dict
        Dictionary that describes the center frequencies for each isotope in the
        units of ppm.

    Returns
    -------
    P_rot : ndarray or csc_array
        The time propagator transformed into the rotating frame.
    """
    print("Applying rotating frame transformation...")
    time_start = time.time()

    # Obtain an array of center frequencies for each spin
    center = np.zeros(spin_system.nspins)
    for spin in range(spin_system.nspins):
        if spin_system.isotopes[spin] in center_frequencies:
            center[spin] = center_frequencies[spin_system.isotopes[spin]]

    # Construct Hamiltonian that specifies the interaction frame
    with HidePrints():
        H_frame = sop_H(
            basis = spin_system.basis.basis,
            spins = spin_system.spins,
            gammas = spin_system.gammas,
            B = parameters.magnetic_field,
            chemical_shifts = center,
            interactions = ["zeeman", "chemical_shift"],
            side = "comm",
            zero_value = parameters.zero_hamiltonian
        )

    # Acquire matrix exponential from the Hamiltonian
    with HidePrints():
        expm_H0t = expm(1j * H_frame * t, parameters.zero_propagator)

    # Convert the time propagator to rotating frame
    P_rot = expm_H0t @ P

    print("Rotating frame transformation applied in "
          f"{time.time() - time_start:.4f} seconds.")
    print()
    
    return P_rot