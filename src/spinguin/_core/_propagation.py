"""
Time-propagation utilities for Hilbert-space and Liouville-space calculations.

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


def _report_completion(
    time_start: float,
) -> None:
    """
    Print the elapsed run time of a completed propagation task.

    Usage: ``_report_completion(time_start)``.

    Parameters
    ----------
    time_start : float
        Start time returned by ``time.time()``.

    Returns
    -------
    None
        The elapsed wall-clock time is reported via the status printer.
    """

    # Report the elapsed wall-clock time of the completed operation.
    status(f"Completed in {time.time() - time_start:.4f} seconds.\n")

def propagator(
    L: np.ndarray | sp.csc_array,
    t: float
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
    status("Constructing propagator...")
    time_start = time.time()

    # Evaluate the propagator from the Liouvillian matrix exponential.
    P = expm(L * t, parameters.zero_propagator)

    # Determine the density of the propagator for storage selection.
    if sp.issparse(P):
        density = P.nnz / (P.shape[0] ** 2)
    else:
        density = np.count_nonzero(P) / (P.shape[0] ** 2)
    status(f"Propagator density: {density:.4f}")

    # Convert sparse propagators to dense form if they are sufficiently full.
    if sp.issparse(P) and density > parameters.propagator_density:
        status("Density exceeds threshold. Converting to NumPy array.")
        P = P.toarray()

    # Report the completion of the propagator construction.
    _report_completion(time_start)

    return P


def pulse(
    spin_system: SpinSystem,
    operator: str,
    angle: float
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

        NOTE: Indexing starts from 0!
    angle : float
        Pulse angle in degrees.

    Returns
    -------
    P : ndarray or csc_array
        Pulse superoperator.

    Warns
    -----
    UserWarning
        Raised if the pulse is generated from a product operator, for which the
        pulse angle is not uniquely defined.
    """

    # Ensure that the working basis has been constructed.
    if spin_system.basis.basis is None:
        raise ValueError("Please build the basis before constructing pulse "
                         "superoperators.")

    # Report the start of the pulse-superoperator construction.
    time_start = time.time()
    status("Creating a pulse superoperator...")

    # Warn if the requested pulse uses a product operator.
    if '*' in operator:
        warnings.warn("Applying a pulse using a product operator does not have "
                      "a well-defined angle.")

    # Construct the commutation superoperator for the pulse generator.
    op = superoperator(spin_system, operator, side="comm")

    # Convert the pulse angle from degrees to radians.
    angle = angle / 180 * np.pi

    # Evaluate the pulse propagator while silencing nested status messages.
    with HidePrints():
        P = expm(-1j * angle * op, parameters.zero_pulse)

    # Report the completion of the pulse-superoperator construction.
    _report_completion(time_start)

    return P


def propagator_to_rotframe(
    spin_system: SpinSystem,
    P: np.ndarray | sp.csc_array,
    t: float,
    center_frequencies: dict=None
) -> np.ndarray | sp.csc_array:
    """
    Transform a time propagator to the rotating frame.

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
        if spin_system.isotopes[spin] in center_frequencies:
            center[spin] = center_frequencies[spin_system.isotopes[spin]]

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
    _report_completion(time_start)
    
    return P_rot