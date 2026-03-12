"""
This module provides functions for calculating Hamiltonian superoperators.
"""
# Referencing SpinSystem class
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from spinguin._core._spin_system import SpinSystem

# Imports
import numpy as np
import time
from typing import Literal
from scipy.sparse import csc_array
from spinguin._core._la import eliminate_small
from spinguin._core._superoperators import superoperator
from spinguin._core._parameters import parameters
from spinguin._core._status import status

def _sop_H_Z_CS(
    spin_system: SpinSystem,
    side: Literal["comm", "left", "right"],
    zeeman: bool,
    cs: bool
) -> np.ndarray | csc_array:
    """
    Computes the Hamiltonian superoperator for the Zeeman interaction and the
    chemical shift.

    Parameters
    ----------
    spin_system : SpinSystem
        Spin system for which to calculate the Zeeman and chemical shift
        Hamiltonian.
    side : {'comm', 'left', 'right'}
        Specifies the type of superoperator:
        - 'comm' -- commutation superoperator
        - 'left' -- left superoperator
        - 'right' -- right superoperator
    zeeman : bool
        If True, the Zeeman Hamiltonian is included.
    cs : bool
        If True, the chemical shift Hamiltonian is included.

    Returns
    -------
    sop_H : ndarray or csc_array
        The Hamiltonian superoperator for the Zeeman interaction and the
        chemical shift.
    """
    # Check that the magnetic field has been set
    if parameters.magnetic_field is None:
        raise ValueError(
            "Please set the magnetic field before constructing the Zeeman or "
            "chemical shift Hamiltonian."
        )

    # Initialize the Hamiltonian
    dim = spin_system.basis.dim
    if parameters.sparse_superoperator:
        sop_H = csc_array((dim, dim), dtype=complex)
    else:
        sop_H = np.zeros((dim, dim), dtype=complex)

    # Iterate over each spin
    for n in range(spin_system.nspins):

        # Obtain the bare nucleus frequency
        omega_0 = -spin_system.gammas[n] * parameters.magnetic_field

        # Obtain the requested frequency
        if zeeman and cs:
            omega = omega_0 * (1 + spin_system.chemical_shifts[n] * 1e-6)
        elif zeeman:
            omega = omega_0
        elif cs:
            omega = omega_0 * spin_system.chemical_shifts[n] * 1e-6
        else:
            raise ValueError("zeeman or cs must be True")

        # Get the z-operator for the current spin
        Iz = superoperator(spin_system, f"I(z, {n})", side)

        # Add the Hamiltonian term for the current spin
        sop_H += omega * Iz

    return sop_H

def _sop_H_J(
    spin_system: SpinSystem,
    side: Literal["comm", "left", "right"],
) -> np.ndarray | csc_array:
    """
    Computes the J-coupling term of the Hamiltonian.

    Parameters
    ----------
    spin_system : SpinSystem
        Spin system for which to calculate the J-coupling Hamiltonian.
    side : {'comm', 'left', 'right'}
        Specifies the type of superoperator:
        - 'comm' -- commutation superoperator
        - 'left' -- left superoperator
        - 'right' -- right superoperator

    Returns
    -------
    sop_H : ndarray or csc_array
        The J-coupling Hamiltonian superoperator.
    """
    # Initialise the J-coupling Hamiltonian
    dim = spin_system.basis.dim
    if parameters.sparse_superoperator:
        sop_H = csc_array((dim, dim), dtype=complex)
    else:
        sop_H = np.zeros((dim, dim), dtype=complex)

    # Loop over the spin pairs (consider only lower triangular part)
    for n in range(spin_system.nspins):
        for k in range(n):

            # Obtain the spin operators and the J-coupling
            J = spin_system.J_couplings[n][k]
            IzIz = superoperator(spin_system, f"I(z,{n})*I(z,{k})", side)
            IpIm = superoperator(spin_system, f"I(+,{n})*I(-,{k})", side)
            ImIp = superoperator(spin_system, f"I(-,{n})*I(+,{k})", side)

            # Compute the J-coupling term
            sop_H += 2 * np.pi * J * (IzIz + 1/2*(IpIm + ImIp))
            
    return sop_H

INTERACTIONTYPE = Literal["zeeman", "chemical_shift", "J_coupling"]
INTERACTIONDEFAULT = ["zeeman", "chemical_shift", "J_coupling"]
def hamiltonian(
    spin_system: SpinSystem,
    interactions: list[INTERACTIONTYPE] = INTERACTIONDEFAULT,
    side: Literal["comm", "left", "right"] = "comm"
) -> np.ndarray | csc_array:
    """
    Creates the Hamiltonian superoperator for the spin system.

    Parameters
    ----------
    spin_system : SpinSystem
        Spin system for which the Hamiltonian is going to be generated.
    interactions : list, default=["zeeman", "chemical_shift", "J_coupling"]
        Specifies which interactions are taken into account. The options are:

        - 'zeeman' -- Zeeman interaction
        - 'chemical_shift' -- Isotropic chemical shift
        - 'J_coupling' -- Scalar J-coupling

    side : {'comm', 'left', 'right'}
        The type of superoperator:
        
        - 'comm' -- commutation superoperator (default)
        - 'left' -- left superoperator
        - 'right' -- right superoperator

    Returns
    -------
    H : ndarray or csc_array
        Hamiltonian superoperator.
    """
    time_start = time.time()
    status("Constructing Hamiltonian...")
        
    # Check that the basis has been built
    if spin_system.basis.basis is None:
        raise ValueError("Please build basis before constructing " 
                         "the Hamiltonian.")

    # Check that each item in the interactions list is unique
    if not len(set(interactions)) == len(interactions):
        raise ValueError("Duplicate interactions were specified.")
    
    # Check that at least one interaction has been specified
    if len(interactions) == 0:
        raise ValueError("Cannot compute Hamiltonian, as no interactions were "
                         "specified.")
    
    # Check that each requested interaction is valid
    for interaction in interactions:
        if interaction not in INTERACTIONDEFAULT:
            raise ValueError(
                f"Invalid interaction: {interaction}. "
                f"Valid interactions are: {INTERACTIONDEFAULT}."
            )

    # Initialize the Hamiltonian
    dim = spin_system.basis.dim
    if parameters.sparse_superoperator:
        sop_H = csc_array((dim, dim), dtype=complex)
    else:
        sop_H = np.zeros((dim, dim), dtype=complex)

    # Calculate the Zeeman and chemical shift Hamiltonian
    zeeman = "zeeman" in interactions
    cs = "chemical_shift" in interactions
    if zeeman or cs:
        sop_H += _sop_H_Z_CS(spin_system, side, zeeman, cs)
        
    # Calculate the J-coupling Hamiltonian
    if "J_coupling" in interactions:
        sop_H += _sop_H_J(spin_system, side)

    # Remove small values to increase sparsity
    eliminate_small(sop_H, parameters.zero_hamiltonian)

    status(f'Completed in {time.time() - time_start:.4f} seconds.\n')

    return sop_H