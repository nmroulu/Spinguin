"""
hamiltonian.py

This module provides functions for calculating the Hamiltonian superoperators.
"""

# For referencing the SpinSystem class
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from spinguin.spin_system import SpinSystem

# Imports
import numpy as np
import time
from scipy.sparse import csc_array
from spinguin import la
from spinguin.operators import sop_P

def hamiltonian_zeeman(spin_system:SpinSystem, B: float, side: str='comm') -> csc_array:
    """
    Calculates the Hamiltonian superoperator of Zeeman interaction.

    Parameters
    ----------
    spin_system : SpinSystem
    B : float
        Magnetic field in the units of T.
    side : str
        Can be one of the options:
        - 'comm' -- commutation superoperator (default)
        - 'left' -- left superoperator
        - 'right' -- right superoperator

    Returns
    -------
    sop_Hz : csc_array
        Hamiltonian superoperator of Zeeman interaction.
    """

    # Extract the necessary information from the spin system
    dim = spin_system.basis.dim
    nspins = spin_system.size
    gammas = spin_system.gammas
    chemical_shifts = spin_system.chemical_shifts

    # Initialize the Hamiltonian
    sop_Hz = csc_array((dim, dim), dtype=complex)

    # Go over each spin in the system
    for n in range(nspins):

        # Make operator definition for Z term of n:th spin
        op_def = tuple(2 if i==n else 0 for i in range(nspins))

        # Calculate the Zeeman interaction for current spin
        sop_Hz = sop_Hz - gammas[n] * B * (1 + chemical_shifts[n]*1e-6) * sop_P(spin_system, op_def, side)

    return sop_Hz

def hamiltonian_jcoupling(spin_system:SpinSystem, side: str='comm') -> csc_array:
    """
    Calculates the J-coupling term of the Hamiltonian.

    Parameters
    ----------
    spin_system : SpinSystem
    side : str
        Can be one of the options:
        - 'comm' -- commutation superoperator (default)
        - 'left' -- left superoperator
        - 'right' -- right superoperator

    Returns
    -------
    sop_Hj : csc_array
        J-coupling term of the Hamiltonian.
    """

    # Extract the necessary information from spin system
    dim = spin_system.basis.dim
    scalar_couplings = spin_system.scalar_couplings
    nspins = spin_system.size

    # Initialize the Hamiltonian
    sop_Hj = csc_array((dim, dim), dtype=complex)
    
    # Loop over spin different spin pairs
    for n in range(nspins):
        for k in range(nspins):

            # Process only bottom half of the J-coupling array
            if n > k:
                
                # Operator definition for zz-term
                op_def_00 = tuple(2 if i==n or i==k else 0 for i in range(nspins))

                # Operator definition for flip-flop terms
                op_def_p1m1 = tuple(1 if i == n else 3 if i == k else 0 for i in range(nspins))
                op_def_m1p1 = tuple(3 if i == n else 1 if i == k else 0 for i in range(nspins))

                # Calculate the J-coupling term
                sop_Hj = sop_Hj + 2*np.pi * scalar_couplings[n][k] \
                        * (sop_P(spin_system, op_def_00, side) - (sop_P(spin_system, op_def_p1m1, side) + sop_P(spin_system, op_def_m1p1, side)))

    return sop_Hj

def hamiltonian(spin_system:SpinSystem, B: float, side: str='comm', zero_value: float = 1e-12) -> csc_array:
    """
    Calculates the coherent part of the Hamiltonian that includes the Zeeman
    interaction and the J-couplings.

    Parameters
    ----------
    spin_system : SpinSystem
    B : float
        Magnetic field in the units of T.
    side : str
        Can be one of the options:
        - 'comm' -- commutation superoperator (default)
        - 'left' -- left superoperator
        - 'right' -- right superoperator
    zero_value : float
        Default: 1e-12. Values less than threshold will be set to zero after constructing
        the total Hamiltonian.

    Returns
    -------
    sop_H : csc_array
        Coherent Hamiltonian.
    """

    time_start = time.time()
    print("Starting to construct the Hamiltonian.")

    # Get the Zeeman and J-coupling Hamiltonians
    sop_Hz = hamiltonian_zeeman(spin_system, B, side)
    sop_Hj = hamiltonian_jcoupling(spin_system, side)

    # Sum together
    sop_H = sop_Hz + sop_Hj

    # Remove small values
    la.increase_sparsity(sop_H, zero_value)

    print("Hamiltonian constructed.")
    print(f"Elapsed time: {time.time() - time_start} seconds.")

    return sop_H
