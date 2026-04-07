"""
Hamiltonian-construction helpers for spin systems.

This module provides helper functions for constructing Hamiltonian
superoperators from the interactions present in a spin system.
"""

# Imports
from __future__ import annotations

import time
from typing import TYPE_CHECKING, Literal

import numpy as np
from scipy.sparse import csc_array

from spinguin._core._la import eliminate_small
from spinguin._core._parameters import parameters
from spinguin._core._status import status
from spinguin._core._superoperators import superoperator

if TYPE_CHECKING:
    from spinguin._core._spin_system import SpinSystem

# Define the default set of Hamiltonian interactions.
DEFAULT_INTERACTIONS = ("zeeman", "chemical_shift", "J_coupling")


def _empty_hamiltonian(
    spin_system: SpinSystem,
) -> np.ndarray | csc_array:
    """
    Create an empty Hamiltonian superoperator of the configured type.

    Parameters
    ----------
    spin_system : SpinSystem
        Spin system whose basis dimension determines the operator size.

    Returns
    -------
    ndarray or csc_array
        Empty Hamiltonian superoperator with complex data type.
    """

    # Obtain the Liouville-space dimension from the current basis.
    dim = spin_system.basis.dim

    # Allocate the Hamiltonian in sparse or dense format.
    if parameters.sparse_superoperator:
        return csc_array((dim, dim), dtype=complex)
    return np.zeros((dim, dim), dtype=complex)


def _validate_interactions(
    interactions: list[Literal["zeeman", "chemical_shift", "J_coupling"]],
) -> None:
    """
    Validate the list of requested Hamiltonian interactions.

    Parameters
    ----------
    interactions : list
        Requested interaction labels.

    Returns
    -------
    None
    """

    # Check that each requested interaction appears only once.
    if len(set(interactions)) != len(interactions):
        raise ValueError("Duplicate interactions were specified.")

    # Check that at least one interaction has been requested.
    if len(interactions) == 0:
        raise ValueError(
            "Cannot compute Hamiltonian, as no interactions were specified."
        )

    # Check that every requested interaction is recognised.
    for interaction in interactions:
        if interaction not in DEFAULT_INTERACTIONS:
            raise ValueError(
                f"Invalid interaction: {interaction}. "
                f"Valid interactions are: {DEFAULT_INTERACTIONS}."
            )


def _sop_H_Z_CS(
    spin_system: SpinSystem,
    side: Literal["comm", "left", "right"],
    zeeman: bool,
    cs: bool,
) -> np.ndarray | csc_array:
    """
    Construct the Zeeman and chemical-shift contributions to the Hamiltonian.

    Parameters
    ----------
    spin_system : SpinSystem
        Spin system for which to calculate the Hamiltonian.
    side : {'comm', 'left', 'right'}
        Type of superoperator:

        - 'comm' -- commutation superoperator
        - 'left' -- left superoperator
        - 'right' -- right superoperator
    zeeman : bool
        If True, include the Zeeman Hamiltonian.
    cs : bool
        If True, include the chemical-shift Hamiltonian.

    Returns
    -------
    sop_H : ndarray or csc_array
        Hamiltonian superoperator contribution from the Zeeman interaction and
        the chemical shift.
    """

    # Ensure that the magnetic field has been defined.
    if parameters.magnetic_field is None:
        raise ValueError(
            "Please set the magnetic field before constructing the Zeeman or "
            "chemical shift Hamiltonian."
        )

    # Initialise the requested Hamiltonian contribution.
    sop_H = _empty_hamiltonian(spin_system)

    # Accumulate the single-spin Zeeman and chemical-shift terms.
    for n in range(spin_system.nspins):

        # Compute the bare Larmor frequency of the current nucleus.
        omega_0 = -spin_system.gammas[n] * parameters.magnetic_field

        # Select the requested frequency contribution.
        if zeeman and cs:
            omega = omega_0 * (1 + spin_system.chemical_shifts[n] * 1e-6)
        elif zeeman:
            omega = omega_0
        elif cs:
            omega = omega_0 * spin_system.chemical_shifts[n] * 1e-6
        else:
            raise ValueError("zeeman or cs must be True.")

        # Build the z-superoperator for the current spin.
        sop_Iz = superoperator(spin_system, f"I(z, {n})", side)

        # Add the single-spin contribution to the Hamiltonian.
        sop_H += omega * sop_Iz

    return sop_H


def _sop_H_J(
    spin_system: SpinSystem,
    side: Literal["comm", "left", "right"],
) -> np.ndarray | csc_array:
    """
    Construct the scalar J-coupling contribution to the Hamiltonian.

    Parameters
    ----------
    spin_system : SpinSystem
        Spin system for which to calculate the Hamiltonian.
    side : {'comm', 'left', 'right'}
        Type of superoperator:

        - 'comm' -- commutation superoperator
        - 'left' -- left superoperator
        - 'right' -- right superoperator

    Returns
    -------
    sop_H : ndarray or csc_array
        J-coupling Hamiltonian superoperator contribution.
    """

    # Initialise the J-coupling contribution.
    sop_H = _empty_hamiltonian(spin_system)

    # Accumulate pairwise scalar-coupling terms over unique spin pairs.
    for n in range(spin_system.nspins):
        for k in range(n):

            # Obtain the coupling constant and the required operator products.
            J = spin_system.J_couplings[n][k]
            IzIz = superoperator(spin_system, f"I(z,{n})*I(z,{k})", side)
            IpIm = superoperator(spin_system, f"I(+,{n})*I(-,{k})", side)
            ImIp = superoperator(spin_system, f"I(-,{n})*I(+,{k})", side)

            # Add the isotropic scalar-coupling term for the spin pair.
            sop_H += 2 * np.pi * J * (IzIz + 1 / 2 * (IpIm + ImIp))

    return sop_H


def hamiltonian(
    spin_system: SpinSystem,
    interactions: list[Literal["zeeman", "chemical_shift", "J_coupling"]]
    = DEFAULT_INTERACTIONS,
    side: Literal["comm", "left", "right"] = "comm",
) -> np.ndarray | csc_array:
    """
    Construct the Hamiltonian superoperator for a spin system.

    Parameters
    ----------
    spin_system : SpinSystem
        Spin system for which the Hamiltonian is generated.
    interactions : list, default=["zeeman", "chemical_shift", "J_coupling"]
        Interactions to include in the Hamiltonian. The available options are:

        - 'zeeman' -- Zeeman interaction
        - 'chemical_shift' -- Isotropic chemical shift
        - 'J_coupling' -- Scalar J-coupling
    side : {'comm', 'left', 'right'}
        Type of superoperator:

        - 'comm' -- commutation superoperator (default)
        - 'left' -- left superoperator
        - 'right' -- right superoperator

    Returns
    -------
    H : ndarray or csc_array
        Hamiltonian superoperator.
    """

    # Record the start time for status reporting.
    time_start = time.time()
    status("Constructing the Hamiltonian...")

    # Ensure that the basis has been built before constructing the Hamiltonian.
    if spin_system.basis.basis is None:
        raise ValueError(
            "Please build basis before constructing the Hamiltonian."
        )

    # Validate the list of requested interactions.
    _validate_interactions(interactions)

    # Initialise the full Hamiltonian superoperator.
    sop_H = _empty_hamiltonian(spin_system)

    # Add the Zeeman and chemical-shift contributions if requested.
    zeeman = "zeeman" in interactions
    cs = "chemical_shift" in interactions
    if zeeman or cs:
        sop_H += _sop_H_Z_CS(spin_system, side, zeeman, cs)

    # Add the scalar J-coupling contribution if requested.
    if "J_coupling" in interactions:
        sop_H += _sop_H_J(spin_system, side)

    # Remove very small values to improve sparsity and numerical cleanliness.
    eliminate_small(sop_H, parameters.zero_hamiltonian)

    # Report the completion of the Hamiltonian construction.
    status(
        f"Hamiltonian constructed in {time.time() - time_start:.4f} "
        "seconds.\n"
    )

    return sop_H