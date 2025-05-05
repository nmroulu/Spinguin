"""
hamiltonian.py

This module provides functions for calculating Hamiltonian superoperators.
"""

# For referencing the SpinSystem class
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from spinguin.system.spin_system import SpinSystem

# Imports
import numpy as np
import time
from scipy.sparse import csc_array
from spinguin.utils.la import increase_sparsity
from spinguin.qm.superoperators import sop_prod
from spinguin.config import Config

def hamiltonian_zeeman(spin_system: SpinSystem, side: str = 'comm', include_shifts: bool=True) -> csc_array:
    """
    Computes the Hamiltonian superoperator for the Zeeman interaction.

    Parameters
    ----------
    spin_system : SpinSystem
        The spin system object containing information about the spins.
    side : str
        Specifies the type of superoperator:
        - 'comm' -- commutation superoperator (default)
        - 'left' -- left superoperator
        - 'right' -- right superoperator
    include_shifts : bool, default=True
        Specifies whether to include the chemical shifts in the Zeeman Hamiltonian.

    Returns
    -------
    sop_Hz : csc_array
        The Hamiltonian superoperator for the Zeeman interaction.
    """

    # Extract relevant information from the spin system
    dim = spin_system.basis.dim
    nspins = spin_system.size
    gammas = spin_system.gammas
    chemical_shifts = spin_system.chemical_shifts

    # Initialize the Hamiltonian
    sop_Hz = csc_array((dim, dim), dtype=complex)

    # Iterate over each spin in the system
    for n in range(nspins):

        # Define the operator for the Z term of the nth spin
        op_def = tuple(2 if i == n else 0 for i in range(nspins))

        # Compute the Zeeman interaction for the current spin
        if include_shifts:
            sop_Hz = sop_Hz - gammas[n] * Config.magnetic_field * (1 + chemical_shifts[n] * 1e-6) * sop_prod(spin_system, op_def, side)
        else:
            sop_Hz = sop_Hz - gammas[n] * Config.magnetic_field * sop_prod(spin_system, op_def, side)

    return sop_Hz

def hamiltonian_J_coupling(spin_system: SpinSystem, side: str = 'comm') -> csc_array:
    """
    Computes the J-coupling term of the Hamiltonian.

    Parameters
    ----------
    spin_system : SpinSystem
        The spin system object containing information about the spins.
    side : str
        Specifies the type of superoperator:
        - 'comm' -- commutation superoperator (default)
        - 'left' -- left superoperator
        - 'right' -- right superoperator

    Returns
    -------
    sop_Hj : csc_array
        The J-coupling Hamiltonian superoperator.
    """

    # Extract relevant information from the spin system
    dim = spin_system.basis.dim
    J_couplings = spin_system.J_couplings
    nspins = spin_system.size

    # Initialize the Hamiltonian
    sop_Hj = csc_array((dim, dim), dtype=complex)
    
    # Loop over all spin pairs
    for n in range(nspins):
        for k in range(nspins):

            # Process only the lower triangular part of the J-coupling matrix
            if n > k:
                
                # Define the operator for the zz-term
                op_def_00 = tuple(2 if i == n or i == k else 0 for i in range(nspins))

                # Define the operators for flip-flop terms
                op_def_p1m1 = tuple(1 if i == n else 3 if i == k else 0 for i in range(nspins))
                op_def_m1p1 = tuple(3 if i == n else 1 if i == k else 0 for i in range(nspins))

                # Compute the J-coupling term
                sop_Hj += 2 * np.pi * J_couplings[n][k] * (
                    sop_prod(spin_system, op_def_00, side) - 
                    (sop_prod(spin_system, op_def_p1m1, side) + sop_prod(spin_system, op_def_m1p1, side))
                )

    return sop_Hj

def hamiltonian(spin_system: SpinSystem, side: str = 'comm') -> csc_array:
    """
    Computes the coherent part of the Hamiltonian superoperator, including the Zeeman
    interaction and J-couplings.

    Parameters
    ----------
    spin_system : SpinSystem
        The spin system object containing information about the spins.
    side : str
        Specifies the type of superoperator:
        - 'comm' -- commutation superoperator (default)
        - 'left' -- left superoperator
        - 'right' -- right superoperator

    Returns
    -------
    sop_H : csc_array
        The coherent Hamiltonian.
    """

    time_start = time.time()
    print("Constructing Hamiltonian...")

    # Compute the Zeeman and J-coupling Hamiltonians
    sop_Hz = hamiltonian_zeeman(spin_system, side)
    sop_Hj = hamiltonian_J_coupling(spin_system, side)

    # Combine the terms
    sop_H = sop_Hz + sop_Hj

    # Remove small values to enhance sparsity
    increase_sparsity(sop_H, Config.ZERO_HAMILTONIAN)

    print(f'Hamiltonian constructed in {time.time() - time_start:.4f} seconds.') # NOTE: Perttu's edit
    print()

    return sop_H
