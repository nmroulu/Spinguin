"""
hamiltonian.py

This module provides functions for calculating Hamiltonian superoperators.
"""

# Imports
import numpy as np
import time
from typing import Literal
from scipy.sparse import csc_array
from spinguin.core.la import increase_sparsity
from spinguin.core.superoperators import sop_prod

def sop_H_Z(basis: np.ndarray,
            gammas: np.ndarray,
            spins: np.ndarray,
            B: float,
            side: Literal["comm", "left", "right"] = "comm",
            sparse: bool=True) -> np.ndarray | csc_array:
    """
    Computes the Hamiltonian superoperator for the Zeeman interaction.

    Parameters
    ----------
    basis : ndarray
        A 2-dimensional array containing the basis set that consists sequences
        of integers describing the Kronecker products of irreducible spherical
        tensors.
    gammas : ndarray
        A 1-dimensional array containing the gyromagnetic ratios of each spin in
        the units of rad/s/T
    spins : ndarray
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

    # Obtain the basis set dimension and number of spins
    dim = basis.shape[0]
    nspins = spins.shape[0]

    # Initialize the Hamiltonian
    if sparse:
        sop_Hz = csc_array((dim, dim), dtype=complex)
    else:
        sop_Hz = np.zeros((dim, dim), dtype=complex)

    # Iterate over each spin
    for n in range(nspins):

        # Define the operator for the Z term of the nth spin
        op_def = np.array([2 if i == n else 0 for i in range(nspins)])

        # Compute the Zeeman interaction for the current spin
        sop_Hz = sop_Hz - gammas[n] * B * sop_prod(op_def, basis, spins, side,
                                                   sparse)

    return sop_Hz

def sop_H_CS(basis: np.ndarray,
             gammas: np.ndarray,
             spins: np.ndarray,
             chemical_shifts: np.ndarray,
             B: float,
             side: Literal["comm", "left", "right"] = "comm",
             sparse: bool=True) -> np.ndarray | csc_array:
    """
    Computes the Hamiltonian superoperator for the chemical shift.

    Parameters
    ----------
    basis : ndarray
        A 2-dimensional array containing the basis set that consists sequences
        of integers describing the Kronecker products of irreducible spherical
        tensors.
    gammas : ndarray
        A 1-dimensional array containing the gyromagnetic ratios of each spin in
        the units of rad/s/T
    spins : ndarray
        A 1-dimensional array containing the spin quantum numbers of each spin.
    chemical_shifts : ndarray
        A 1-dimensional array containing the chemical shifts of each spin in the
        units of ppm.
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

    # Obtain the basis set dimension and number of spins
    dim = basis.shape[0]
    nspins = spins.shape[0]

    # Initialize the Hamiltonian
    if sparse:
        sop_Hcs = csc_array((dim, dim), dtype=complex)
    else:
        sop_Hcs = np.zeros((dim, dim), dtype=complex)

    # Iterate over each spin
    for n in range(nspins):

        # Define the operator for the Z term of the nth spin
        op_def = np.array([2 if i == n else 0 for i in range(nspins)])

        # Compute the contribution from chemical shift for the current spin
        sop_Hcs = sop_Hcs - gammas[n] * B * chemical_shifts[n] * 1e-6 * \
            sop_prod(op_def, basis, spins, side, sparse)

    return sop_Hcs

def sop_H_J(basis: np.ndarray,
            spins: np.ndarray,
            J_couplings: np.ndarray,
            side: Literal["comm", "left", "right"] = "comm",
            sparse: bool=True) -> np.ndarray | csc_array:
    """
    Computes the J-coupling term of the Hamiltonian.

    Parameters
    ----------
    basis : ndarray
        A 2-dimensional array containing the basis set that consists sequences
        of integers describing the Kronecker products of irreducible spherical
        tensors.
    spins : ndarray
        A 1-dimensional array containing the spin quantum numbers of each spin.
    J_couplings : ndarray
        A 2-dimensional array containing the scalar J-couplings between each
        spin in the units of Hz. Only the bottom triangle is considered.
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

    # Obtain the basis set dimension and number of spins
    dim = basis.shape[0]
    nspins = spins.shape[0]

    # Initialize the Hamiltonian
    if sparse:
        sop_Hj = csc_array((dim, dim), dtype=complex)
    else:
        sop_Hj = np.zeros((dim, dim), dtype=complex)
    
    # Loop over all spin pairs
    for n in range(nspins):
        for k in range(nspins):

            # Process only the lower triangular part of the J-coupling matrix
            if n > k:
                
                # Define the operator for the zz-term
                op_def_00 = np.array(
                    [2 if i == n or i == k else 0 for i in range(nspins)])

                # Define the operators for flip-flop terms
                op_def_p1m1 = np.array([
                    1 if i == n else 
                    3 if i == k else 0 for i in range(nspins)])
                op_def_m1p1 = np.array([
                    3 if i == n else 
                    1 if i == k else 0 for i in range(nspins)])

                # Compute the J-coupling term
                sop_Hj += 2 * np.pi * J_couplings[n][k] * (
                    sop_prod(op_def_00, basis, spins, side, sparse) \
                        - sop_prod(op_def_p1m1, basis, spins, side, sparse) \
                        - sop_prod(op_def_m1p1, basis, spins, side, sparse))

    return sop_Hj

def sop_H_coherent(basis: np.ndarray,
                   gammas: np.ndarray,
                   spins: np.ndarray,
                   chemical_shifts: np.ndarray,
                   J_couplings: np.ndarray,
                   B: float,
                   side: Literal["comm", "left", "right"] = "comm",
                   sparse: bool=True,
                   zero_value: float=1e-12) -> np.ndarray | csc_array:
    """
    Computes the coherent part of the Hamiltonian superoperator, including the
    Zeeman interaction and J-couplings.

    Parameters
    ----------
    basis : ndarray
        A 2-dimensional array containing the basis set that consists sequences
        of integers describing the Kronecker products of irreducible spherical
        tensors.
    gammas : ndarray
        A 1-dimensional array containing the gyromagnetic ratios of each spin in
        the units of rad/s/T.
    spins : ndarray
        A 1-dimensional array containing the spin quantum numbers of each spin.
    chemical_shifts : ndarray
        A 1-dimensional array containing the chemical shifts of each spin in the
        units of ppm.
    B : float
        External magnetic field in the units of T.
    side : {'comm', 'left', 'right'}
        Specifies the type of superoperator:
        - 'comm' -- commutation superoperator (default)
        - 'left' -- left superoperator
        - 'right' -- right superoperator
    sparse : bool, default=True
        Specifies whether to construct the Hamiltonian as sparse or dense array.
    zero_value : float, default=1e-12
        Smaller values than this threshold are made equal to zero after
        calculating the Hamiltonian. When using sparse arrays, larger values
        decrease the memory requirement at the cost of accuracy.

    Returns
    -------
    sop_H : ndarray or csc_array
        The coherent Hamiltonian.
    """

    time_start = time.time()
    print("Constructing Hamiltonian...")

    # Compute the Zeeman and J-coupling Hamiltonians
    sop_Hz = sop_H_Z(basis, gammas, spins, B, side, sparse)
    sop_Hcs = sop_H_CS(basis, gammas, spins, chemical_shifts, B, side, sparse)
    sop_Hj = sop_H_J(basis, spins, J_couplings, side, sparse)

    # Combine the terms
    sop_H = sop_Hz + sop_Hcs + sop_Hj

    # Remove small values to enhance sparsity
    increase_sparsity(sop_H, zero_value)

    print(f'Hamiltonian constructed in {time.time() - time_start:.4f} seconds.')
    print()

    return sop_H