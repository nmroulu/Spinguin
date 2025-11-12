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
from spinguin._core._superoperators import sop_prod
from spinguin._core._parameters import parameters

def sop_H_Z(
    basis: np.ndarray,
    gammas: np.ndarray,
    spins: np.ndarray,
    B: float,
    side: Literal["comm", "left", "right"] = "comm"
) -> np.ndarray | csc_array:
    """
    Computes the Hamiltonian superoperator for the Zeeman interaction.

    Parameters
    ----------
    basis : ndarray
        A 2-dimensional array containing the basis set that contains sequences
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

    Returns
    -------
    sop_Hz : ndarray or csc_array
        The Hamiltonian superoperator for the Zeeman interaction.
    """

    # Obtain the basis set dimension and number of spins
    dim = basis.shape[0]
    nspins = spins.shape[0]

    # Initialize the Hamiltonian
    if parameters.sparse_superoperator:
        sop_Hz = csc_array((dim, dim), dtype=complex)
    else:
        sop_Hz = np.zeros((dim, dim), dtype=complex)

    # Iterate over each spin
    for n in range(nspins):

        # Define the operator for the Z term of the nth spin
        op_def = np.array([2 if i == n else 0 for i in range(nspins)])

        # Compute the Zeeman interaction for the current spin
        sop_Hz = sop_Hz - gammas[n] * B * sop_prod(op_def, basis, spins, side)

    return sop_Hz

def sop_H_CS(
    basis: np.ndarray,
    gammas: np.ndarray,
    spins: np.ndarray,
    chemical_shifts: np.ndarray,
    B: float,
    side: Literal["comm", "left", "right"] = "comm",
) -> np.ndarray | csc_array:
    """
    Computes the Hamiltonian superoperator for the chemical shift.

    Parameters
    ----------
    basis : ndarray
        A 2-dimensional array containing the basis set that contains sequences
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

    Returns
    -------
    sop_Hz : ndarray or csc_array
        The Hamiltonian superoperator for the chemical shift.
    """

    # Obtain the basis set dimension and number of spins
    dim = basis.shape[0]
    nspins = spins.shape[0]

    # Initialize the Hamiltonian
    if parameters.sparse_superoperator:
        sop_Hcs = csc_array((dim, dim), dtype=complex)
    else:
        sop_Hcs = np.zeros((dim, dim), dtype=complex)

    # Iterate over each spin
    for n in range(nspins):

        # Define the operator for the Z term of the nth spin
        op_def = np.array([2 if i == n else 0 for i in range(nspins)])

        # Compute the contribution from chemical shift for the current spin
        sop_Hcs = sop_Hcs - gammas[n] * B * chemical_shifts[n] * 1e-6 * \
            sop_prod(op_def, basis, spins, side)

    return sop_Hcs

def sop_H_J(
    basis: np.ndarray,
    spins: np.ndarray,
    J_couplings: np.ndarray,
    side: Literal["comm", "left", "right"] = "comm",
) -> np.ndarray | csc_array:
    """
    Computes the J-coupling term of the Hamiltonian.

    Parameters
    ----------
    basis : ndarray
        A 2-dimensional array containing the basis set that contains sequences
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

    Returns
    -------
    sop_Hj : ndarray or csc_array
        The J-coupling Hamiltonian superoperator.
    """

    # Obtain the basis set dimension and number of spins
    dim = basis.shape[0]
    nspins = spins.shape[0]

    # Initialize the Hamiltonian
    if parameters.sparse_superoperator:
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
                    sop_prod(op_def_00, basis, spins, side) \
                        - sop_prod(op_def_p1m1, basis, spins, side) \
                        - sop_prod(op_def_m1p1, basis, spins, side))

    return sop_Hj

INTERACTIONTYPE = Literal["zeeman", "chemical_shift", "J_coupling"]
INTERACTIONDEFAULT = ["zeeman", "chemical_shift", "J_coupling"]
def sop_H(
    basis: np.ndarray,
    spins: np.ndarray,
    gammas: np.ndarray = None,
    B: float = None,
    chemical_shifts: np.ndarray = None,
    J_couplings: np.ndarray = None,
    interactions: list[INTERACTIONTYPE] = INTERACTIONDEFAULT,
    side: Literal["comm", "left", "right"] = "comm",
    zero_value: float=1e-12
) -> np.ndarray | csc_array:
    """
    Computes the coherent part of the Hamiltonian superoperator, including the
    Zeeman interaction, isotropic chemical shift, and J-couplings.

    Parameters
    ----------
    basis : ndarray
        A 2-dimensional array containing the basis set that contains sequences
        of integers describing the Kronecker products of irreducible spherical
        tensors.
    spins : ndarray
        A 1-dimensional array containing the spin quantum numbers of each spin.
    gammas : ndarray
        A 1-dimensional array containing the gyromagnetic ratios of each spin in
        the units of rad/s/T.
    B : float
        External magnetic field in the units of T.
    chemical_shifts : ndarray
        A 1-dimensional array containing the chemical shifts of each spin in the
        units of ppm.
    J_couplings : ndarray
        A 2-dimensional array containing the scalar J-couplings between each
        spin in the units of Hz. Only the bottom triangle is considered.
    interactions : list, default=["zeeman", "chemical_shift", "J_coupling"]
        Specifies which interactions are taken into account. The options are:
        - 'zeeman' -- Zeeman interaction
        - 'chemical_shift' -- Isotropic chemical shift
        - 'J_coupling' -- Scalar J-coupling
    side : {'comm', 'left', 'right'}
        Specifies the type of superoperator:
        - 'comm' -- commutation superoperator (default)
        - 'left' -- left superoperator
        - 'right' -- right superoperator
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

    # Check that each item in the interactions list is unique
    if not len(set(interactions)) == len(interactions):
        raise ValueError("Cannot compute Hamiltonian, as duplicate "
                         "interactions were specified.")
    
    # Check that at least one interaction has been specified
    if len(interactions) == 0:
        raise ValueError("Cannot compute Hamiltonian, as no interactions were "
                         "specified.")

    # Obtain the basis set dimension
    dim = basis.shape[0]

    # Initialize the Hamiltonian
    if parameters.sparse_superoperator:
        sop_H = csc_array((dim, dim), dtype=complex)
    else:
        sop_H = np.zeros((dim, dim), dtype=complex)

    # Compute the Zeeman and J-coupling Hamiltonians
    for interaction in interactions:
        if interaction == "zeeman":
            sop_H += sop_H_Z(basis, gammas, spins, B, side)
        elif interaction == "chemical_shift":
            sop_H += sop_H_CS(basis, gammas, spins, chemical_shifts, B, side)
        elif interaction == "J_coupling":
            sop_H += sop_H_J(basis, spins, J_couplings, side)
        else:
            raise ValueError(f"Unsupported interaction type: {interaction}. "
                             f"The possible options are: {INTERACTIONDEFAULT}.")

    # Remove small values to enhance sparsity
    eliminate_small(sop_H, zero_value)

    print(f'Hamiltonian constructed in {time.time() - time_start:.4f} seconds.')
    print()

    return sop_H

def hamiltonian(
    spin_system: SpinSystem,
    interactions: list[INTERACTIONTYPE] = INTERACTIONDEFAULT,
    side: Literal["comm", "left", "right"] = "comm"
) -> np.ndarray | csc_array:
    """
    Creates the requested Hamiltonian superoperator for the spin system.

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
        
    # Check that the required attributes have been set
    if spin_system.basis.basis is None:
        raise ValueError("Please build basis before constructing " 
                         "the Hamiltonian.")
    if "zeeman" in interactions:
        if parameters.magnetic_field is None:
            raise ValueError("Please set the magnetic field before "
                             "constructing the Zeeman Hamiltonian.")
    if "chemical_shift" in interactions:
        if parameters.magnetic_field is None:
            raise ValueError("Please set the magnetic field before "
                             "constructing the chemical shift Hamiltonian.")
        
    H = sop_H(
        basis = spin_system.basis.basis,
        spins = spin_system.spins,
        gammas = spin_system.gammas,
        B = parameters.magnetic_field,
        chemical_shifts = spin_system.chemical_shifts,
        J_couplings = spin_system.J_couplings,
        interactions = interactions,
        side = side,
        zero_value = parameters.zero_hamiltonian
    )

    return H