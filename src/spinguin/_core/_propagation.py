"""
Time-propagation utilities for Hilbert-space and Liouville-space simulations.

The module contains helpers for constructing time propagators, pulse
superoperators, and rotating-frame propagators used in spin-dynamics
simulations.
"""

from __future__ import annotations

import time
import warnings
from copy import deepcopy
from typing import TYPE_CHECKING

import numpy as np
import scipy.sparse as sp

from spinguin._core._hamiltonian import hamiltonian
from spinguin._core._hide_prints import HidePrints
from spinguin._core._la import expm
from spinguin._core._parameters import parameters
from spinguin._core._status import status
from spinguin._core._superoperators import superoperator

if TYPE_CHECKING:
    from spinguin._core._spin_system import SpinSystem


def propagator(
    L: np.ndarray | sp.csc_array,
    t: float,
) -> np.ndarray | sp.csc_array:
    """
    Construct the time propagator ``exp(L*t)``.

    Usage: ``propagator(L, t)``.

    Parameters
    ----------
    L : ndarray or csc_array
        Liouvillian superoperator, ``L = -iH - R + K``.
    t : float
        Time step of the simulation in seconds.

    Returns
    -------
    P : ndarray or csc_array
        Time propagator ``exp(L*t)``.
    """

    # Report the start of the propagator construction.
    status("Constructing the time propagator...")
    time_start = time.time()

    # Evaluate the propagator from the Liouvillian matrix exponential.
    P = expm(L * t, parameters.zero_propagator)

    # Determine the density of the propagator for storage selection.
    if sp.issparse(P):
        density = P.nnz / (P.shape[0] ** 2)
    else:
        density = np.count_nonzero(P) / (P.shape[0] ** 2)
    status(f"Propagator density: {100 * density:.4f}%")

    # Convert sparse propagators to dense form if they are sufficiently full.
    if sp.issparse(P) and density > parameters.propagator_density:
        status(
            "Propagator density exceeds threshold. "
            "Converting from sparse to dense array..."
        )
        P = P.toarray()

    # Convert dense propagators to sparse form if they are sufficiently empty.
    elif not sp.issparse(P) and density < parameters.propagator_density:
        status(
            "Propagator density below threshold. "
            "Converting from dense to sparse array..."
        )
        P = sp.csc_array(P)

    # Report the completion of the propagator construction.
    status(f"Completed in {time.time() - time_start:.4f} seconds.\n")

    return P


def pulse(
    spin_system: SpinSystem,
    operator: str,
    angle: float,
) -> np.ndarray | sp.csc_array:
    """
    Construct a pulse superoperator that acts from the left on a state.

    Usage: ``pulse(spin_system, operator, angle)``.

    Parameters
    ----------
    spin_system : SpinSystem
        Spin system for which the pulse superoperator is created.
    operator : str
        Operator definition of the pulse. The supported syntax is summarised
        below:

        - Cartesian or ladder operator at specific index or for all spins::

            operator = "I(component, index)"
            operator = "I(component)"

        - Spherical tensor operator at specific index or for all spins::

            operator = "T(l, q, index)"
            operator = "T(l, q)"

        - Product operators::

            operator = "I(component1, index1) * I(component2, index2)"

        - Sum of operators::

            operator = "I(component1, index1) + I(component2, index2)"

        - Unit operators are ignored in the input. These are identical::

            operator = "E * I(component, index)"
            operator = "I(component, index)"

        Special case: An empty `operator` string is considered as unit operator.

        Whitespace will be ignored in the input.

        NOTE: Python indexing starting from 0 is used.
    angle : float
        Pulse angle in degrees.

    Returns
    -------
    P : ndarray or csc_array
        Pulse superoperator.

    Warns
    -----
    UserWarning
        Raised if the pulse is generated from a product operator, which does
        not straightforwardly correspond to a rotation operator.
    """
    # Ensure that the working basis has been constructed.
    if spin_system.basis.basis is None:
        raise ValueError("Please build the basis before constructing pulse "
                         "superoperators. Use `spin_system.basis.build()`.")

    # Report the start of the pulse-superoperator construction.
    time_start = time.time()
    status("Constructing the pulse superoperator...")

    # Warn if the requested pulse uses a product operator.
    if '*' in operator:
        warnings.warn("Using a product operator to define a pulse does not "
                      "straightforwardly correspond to a rotation operator.")

    # Construct the commutation superoperator for the pulse generator.
    op = superoperator(spin_system, operator, side="comm")

    # Convert the pulse angle from degrees to radians.
    angle = angle * np.pi / 180

    # Evaluate the pulse propagator while silencing nested status messages.
    with HidePrints():
        P = expm(-1j * angle * op, parameters.zero_pulse)

    # Report the completion of the pulse-superoperator construction.
    status(f"Completed in {time.time() - time_start:.4f} seconds.\n")

    return P


def propagator_to_rotframe(
    spin_system: SpinSystem,
    P: np.ndarray | sp.csc_array,
    t: float,
    center_frequencies: dict[str, float] | None=None,
) -> np.ndarray | sp.csc_array:
    """
    Transform a time propagator to the rotating frame defined by the
    center frequencies (Zeeman + chemical shift interactions) 
    of each spin.

    Usage: ``propagator_to_rotframe(spin_system, P, t, center_frequencies)``.

    Parameters
    ----------
    spin_system : SpinSystem
        Spin system whose time propagator is transformed.
    P : ndarray or csc_array
        Time propagator in the laboratory frame.
    t : float
        Time step of the propagator in seconds.
    center_frequencies : dict, default=None
        Dictionary of centre frequencies in ppm for each isotope.

    Returns
    -------
    P_rot : ndarray or csc_array
        Time propagator transformed into the rotating frame.
    """

    # Replace a missing centre-frequency dictionary with an empty mapping.
    if center_frequencies is None:
        center_frequencies = {}

    # Report the start of the rotating-frame transformation.
    status("Applying rotating frame transformation...")
    time_start = time.time()

    # Assemble the isotope-dependent centre-frequency array.
    center = np.zeros(spin_system.nspins)
    for spin in range(spin_system.nspins):
        isotope = spin_system.isotopes[spin]
        if isotope in center_frequencies:
            center[spin] = center_frequencies[isotope]

    # Copy the spin system and insert the rotating-frame reference shifts.
    with HidePrints():
        spin_system_copy = deepcopy(spin_system)
        spin_system_copy.chemical_shifts = center

    # Construct the Hamiltonian that defines the interaction frame.
    with HidePrints():
        H_frame = hamiltonian(spin_system_copy, ["zeeman", "chemical_shift"])

    # Evaluate the frame-transformation propagator.
    with HidePrints():
        expm_H0t = expm(1j * H_frame * t, parameters.zero_propagator)

    # Left-transform the laboratory-frame propagator into the rotating frame.
    P_rot = expm_H0t @ P

    # Report the completion of the rotating-frame transformation.
    status(f"Completed in {time.time() - time_start:.4f} seconds.\n")

    return P_rot