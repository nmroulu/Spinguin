"""
hamiltonian.py

This module provides functions for calculating Hamiltonian superoperators.
"""

# For referencing the SpinSystem class
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from spinguin.system.spin_system import SpinSystem
    from spinguin.system.basis import Basis

# Imports
import numpy as np
import time
from numpy.typing import ArrayLike
from typing import Literal
from scipy.sparse import csc_array
from spinguin.utils.la import increase_sparsity
from spinguin.qm.superoperators import sop_prod
from spinguin.config import Config

def sop_H_Z(basis: Basis,
            gammas: ArrayLike,
            spins: ArrayLike,
            B: float,
            side: Literal["comm", "left", "right"] = "comm",
            sparse: bool=True) -> np.ndarray | csc_array:
    """
    Computes the Hamiltonian superoperator for the Zeeman interaction.

    Parameters
    ----------
    basis : Basis
        Basis set that consists of products of irreducible spherical tensors defined
        by tuples of integers.
    gammas : ArrayLike
        A 1-dimensional array containing the gyromagnetic ratios of each spin in the
        units of rad/s/T
    spins : ArrayLike
        A 1-dimensional array containing the spin quantum numbers of each spin.
    B : float
        External magnetic field in the units of T.
    side : {'comm', 'left', 'right'}
        Specifies the type of superoperator:
        - 'comm' -- commutation superoperator (default)
        - 'left' -- left superoperator
        - 'right' -- right superoperator
    sparse : bool, default=True
        Specifies whether to construct the Hamiltonian as sparse or dense array.

    Returns
    -------
    sop_Hz : ndarray or csc_array
        The Hamiltonian superoperator for the Zeeman interaction.
    """

    # Initialize the Hamiltonian
    if sparse:
        sop_Hz = csc_array((basis.dim, basis.dim), dtype=complex)
    else:
        sop_Hz = np.zeros((basis.dim, basis.dim), dtype=complex)

    # Iterate over each spin
    for n in range(basis.nspins):

        # Define the operator for the Z term of the nth spin
        op_def = tuple(2 if i == n else 0 for i in range(basis.nspins))

        # Compute the Zeeman interaction for the current spin
        sop_Hz = sop_Hz - gammas[n] * B * sop_prod(op_def, basis, spins, side, sparse)

    return sop_Hz

def sop_H_CS(basis: Basis,
             gammas: ArrayLike,
             spins: ArrayLike,
             chemical_shifts: ArrayLike,
             B: float,
             side: Literal["comm", "left", "right"] = "comm",
             sparse: bool=True) -> np.ndarray | csc_array:
    """
    Computes the Hamiltonian superoperator for the chemical shift.

    Parameters
    ----------
    basis : Basis
        Basis set that consists of products of irreducible spherical tensors defined
        by tuples of integers.
    gammas : ArrayLike
        A 1-dimensional array containing the gyromagnetic ratios of each spin in the
        units of rad/s/T
    spins : ArrayLike
        A 1-dimensional array containing the spin quantum numbers of each spin.
    chemical_shifts : ArrayLike
        A 1-dimensional array containing the chemical shifts of each spin in the units
        of ppm.
    B : float
        External magnetic field in the units of T.
    side : {'comm', 'left', 'right'}
        Specifies the type of superoperator:
        - 'comm' -- commutation superoperator (default)
        - 'left' -- left superoperator
        - 'right' -- right superoperator
    sparse : bool, default=True
        Specifies whether to construct the Hamiltonian as sparse or dense array.

    Returns
    -------
    sop_Hz : ndarray or csc_array
        The Hamiltonian superoperator for the chemical shift.
    """

    # Initialize the Hamiltonian
    if sparse:
        sop_Hcs = csc_array((basis.dim, basis.dim), dtype=complex)
    else:
        sop_Hcs = np.zeros((basis.dim, basis.dim), dtype=complex)

    # Iterate over each spin
    for n in range(basis.nspins):

        # Define the operator for the Z term of the nth spin
        op_def = tuple(2 if i == n else 0 for i in range(basis.nspins))

        # Compute the contribution from chemical shift for the current spin
        sop_Hcs = sop_Hcs - gammas[n] * B * chemical_shifts[n] * 1e-6 * sop_prod(op_def, basis, spins, side, sparse)

    return sop_Hcs

def sop_H_J(basis: Basis,
            spins: ArrayLike,
            J_couplings: ArrayLike,
            side: Literal["comm", "left", "right"] = "comm",
            sparse: bool=True) -> np.ndarray | csc_array:
    """
    Computes the J-coupling term of the Hamiltonian.

    Parameters
    ----------
    basis : Basis
        Basis set that consists of products of irreducible spherical tensors defined
        by tuples of integers.
    spins : ArrayLike
        A 1-dimensional array containing the spin quantum numbers of each spin.
    J_couplings : ArrayLike
        A 2-dimensional array containing the scalar J-couplings between each spin in
        the units of Hz. Only the bottom triangle is considered.
    side : {'comm', 'left', 'right'}
        Specifies the type of superoperator:
        - 'comm' -- commutation superoperator (default)
        - 'left' -- left superoperator
        - 'right' -- right superoperator
    sparse : bool, default=True
        Specifies whether to construct the Hamiltonian as sparse or dense array.

    Returns
    -------
    sop_Hj : ndarray or csc_array
        The J-coupling Hamiltonian superoperator.
    """

    # Initialize the Hamiltonian
    if sparse:
        sop_Hj = csc_array((basis.dim, basis.dim), dtype=complex)
    else:
        sop_Hj = np.zeros((basis.dim, basis.dim), dtype=complex)
    
    # Loop over all spin pairs
    for n in range(basis.nspins):
        for k in range(basis.nspins):

            # Process only the lower triangular part of the J-coupling matrix
            if n > k:
                
                # Define the operator for the zz-term
                op_def_00 = tuple(2 if i == n or i == k else 0 for i in range(basis.nspins))

                # Define the operators for flip-flop terms
                op_def_p1m1 = tuple(1 if i == n else 3 if i == k else 0 for i in range(basis.nspins))
                op_def_m1p1 = tuple(3 if i == n else 1 if i == k else 0 for i in range(basis.nspins))

                # Compute the J-coupling term
                sop_Hj += 2 * np.pi * J_couplings[n][k] * (
                    sop_prod(op_def_00, basis, spins, side, sparse) \
                        - sop_prod(op_def_p1m1, basis, spins, side, sparse) \
                        - sop_prod(op_def_m1p1, basis, spins, side, sparse))

    return sop_Hj

def hamiltonian(spin_system: SpinSystem,
                B: float,
                side: Literal["comm", "left", "right"] = "comm",
                sparse: bool=True) -> np.ndarray | csc_array:
    """
    Computes the coherent part of the Hamiltonian superoperator, including the Zeeman
    interaction and J-couplings.

    Parameters
    ----------
    spin_system : SpinSystem
        The spin system object containing information about the spins.
    side : {'comm', 'left', 'right'}
        Specifies the type of superoperator:
        - 'comm' -- commutation superoperator (default)
        - 'left' -- left superoperator
        - 'right' -- right superoperator

    Returns
    -------
    sop_H : ndarray or csc_array
        The coherent Hamiltonian.
    """

    time_start = time.time()
    print("Constructing Hamiltonian...")

    # Compute the Zeeman and J-coupling Hamiltonians
    sop_Hz = sop_H_Z(spin_system.basis, spin_system.gammas, spin_system.spins,
                     B, side, sparse)
    sop_Hcs = sop_H_CS(spin_system.basis, spin_system.gammas, spin_system.spins,
                       spin_system.chemical_shifts, B, side, sparse)
    sop_Hj = sop_H_J(spin_system.basis, spin_system.spins, spin_system.J_couplings,
                     side, sparse)

    # Combine the terms
    sop_H = sop_Hz + sop_Hcs + sop_Hj

    # Remove small values to enhance sparsity
    increase_sparsity(sop_H, Config.ZERO_HAMILTONIAN)

    print(f'Hamiltonian constructed in {time.time() - time_start:.4f} seconds.')
    print()

    return sop_H
