"""
Hamiltonian-construction functions.

This module provides helper functions for constructing Hamiltonian
superoperators from the interactions present in a spin system.
"""

# Imports
from __future__ import annotations

import time
from typing import TYPE_CHECKING, Literal

import numpy as np
from scipy.sparse import csc_array

from spinguin._core._interactions import resonance_frequencies
from spinguin._core._la import eliminate_small
from spinguin._core._parameters import parameters
from spinguin._core._status import status
from spinguin._core._superoperators import superoperator
from spinguin._core._validation import require

if TYPE_CHECKING:
    from spinguin._core._spin_system import SpinSystem


def _empty_hamiltonian(spin_system: SpinSystem) -> np.ndarray | csc_array:
    """
    Create an empty Hamiltonian superoperator.

    Parameters
    ----------
    spin_system : SpinSystem
        Spin system whose basis dimension determines
        the size of the Hamiltonian superoperator.

    Returns
    -------
    ndarray or csc_array
        Empty Hamiltonian superoperator.
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
    """

    # Check that each requested interaction appears only once.
    if len(set(interactions)) != len(interactions):
        raise ValueError("Duplicate interactions were specified.")

    # Check that at least one interaction has been requested.
    if len(interactions) == 0:
        raise ValueError(
            "Cannot compute Hamiltonian, as no interactions were specified."
        )
    
    # Define the set of valid interactions
    valid_interactions = ("zeeman", "chemical_shift", "J_coupling")

    # Check that every requested interaction is recognised.
    for interaction in interactions:
        if interaction not in valid_interactions:
            raise ValueError(
                f"Invalid interaction: {interaction}. "
                f"Valid interactions are: {valid_interactions}."
            )


def _sop_H_Z_CS(
    spin_system: SpinSystem,
    side: Literal["comm", "left", "right"],
    zeeman: bool,
    cs: bool,
) -> np.ndarray | csc_array:
    """
    Constructs the Zeeman (bare nucleus-field interaction) and chemical-shift
    (isotropic shielding interaction) contributions to the Hamiltonian
    superoperator. 

    Parameters
    ----------
    spin_system : SpinSystem
        Spin system for which to calculate the Hamiltonian.
    side : {'comm', 'left', 'right'}
        Type of superoperator:

        - 'comm' -- commutation superoperator
        - 'left' -- left multiplication superoperator
        - 'right' -- right multiplication superoperator

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
    require(
        parameters, 
        "magnetic_field", 
        "constructing the Zeeman and/or chemical shift Hamiltonian"
    )

    # Initialise the requested Hamiltonian contribution.
    sop_H = _empty_hamiltonian(spin_system)

    # Calculate the requested frequencies
    omegas = resonance_frequencies(spin_system, zeeman, cs)

    # Accumulate the single-spin Zeeman and chemical-shift terms.
    for n in range(spin_system.nspins):

        # Build the z-superoperator for the current spin.
        sop_Iz = superoperator(spin_system, f"I(z, {n})", side)

        # Add the single-spin contribution to the Hamiltonian.
        sop_H += omegas[n] * sop_Iz

    return sop_H


def _sop_H_J(
    spin_system: SpinSystem,
    side: Literal["comm", "left", "right"],
) -> np.ndarray | csc_array:
    """
    Construct the scalar J-coupling (isotropic indirect spin-spin coupling)
    contribution to the Hamiltonian.

    Parameters
    ----------
    spin_system : SpinSystem
        Spin system for which to calculate the Hamiltonian.
    side : {'comm', 'left', 'right'}
        Type of superoperator:

        - 'comm' -- commutation superoperator
        - 'left' -- left multiplication superoperator
        - 'right' -- right multiplication superoperator

    Returns
    -------
    sop_H : ndarray or csc_array
        J-coupling Hamiltonian superoperator contribution.
    """

    # Initialise the J-coupling contribution.
    sop_H = _empty_hamiltonian(spin_system)

    # Accumulate pairwise scalar-coupling terms over unique spin pairs.
    # NOTE: It is assumed that the J-coupling matrix in spin_system.J_couplings
    # contains the coupling constants on the lower triangle.
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
    interactions: list[
        Literal["zeeman", "chemical_shift", "J_coupling"]
    ] = ["zeeman", "chemical_shift", "J_coupling"],
    side: Literal["comm", "left", "right"] = "comm",
) -> np.ndarray | csc_array:
    """
    Construct the Hamiltonian superoperator for a spin system.

    Parameters
    ----------
    spin_system : SpinSystem
        Spin system for which the Hamiltonian is constructed.
    interactions : list, default=["zeeman", "chemical_shift", "J_coupling"]
        Interactions to include in the Hamiltonian. The available options are:

        - 'zeeman' -- Zeeman interaction
        - 'chemical_shift' -- Isotropic chemical shift
        - 'J_coupling' -- Scalar J-coupling
        
    side : {'comm', 'left', 'right'}
        Type of superoperator:

        - 'comm' -- commutation superoperator (default)
        - 'left' -- left multiplication superoperator
        - 'right' -- right multiplication superoperator

    Returns
    -------
    H : ndarray or csc_array
        Hamiltonian superoperator.
    """

    # Record the start time for status reporting.
    time_start = time.time()
    status("Constructing the Hamiltonian...")

    # Ensure that the basis has been built before constructing the Hamiltonian.
    require(spin_system, "basis.basis", "constructing the Hamiltonian")

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