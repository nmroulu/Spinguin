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
import scipy.sparse as sp
import time
from typing import Literal
from spinguin._core._config import config
from spinguin._core._la import eliminate_small
from spinguin._core._parameters import parameters
from spinguin._core._superoperators import superoperator_from_op_def

def _hamiltonian_Z(
    spin_system: SpinSystem,
    side: Literal["comm", "left", "right"] = "comm"
) -> np.ndarray | sp.csc_array:
    """
    Computes the Hamiltonian superoperator for the Zeeman interaction.

    Parameters
    ----------
    spin_system : SpinSystem
        Spin system for which the Zeeman Hamiltonian is calculated.
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
    dim = spin_system.basis.dim
    nspins = spin_system.nspins

    # Initialize the Hamiltonian
    if config.sparse_superoperator:
        sop_Hz = sp.csc_array((dim, dim), dtype=complex)
    else:
        sop_Hz = np.zeros((dim, dim), dtype=complex)

    # Iterate over each spin
    for n in range(nspins):

        # Define the operator for the Z term of the nth spin
        op_def = np.array([2 if i == n else 0 for i in range(nspins)])

        # Compute the Zeeman interaction for the current spin
        sop_Hz = sop_Hz - spin_system.gammas[n] * \
                          parameters.magnetic_field * \
                          superoperator_from_op_def(spin_system, op_def, side)

    return sop_Hz

def _hamiltonian_CS(
    spin_system: SpinSystem,
    side: Literal["comm", "left", "right"] = "comm"
) -> np.ndarray | sp.csc_array:
    """
    Computes the Hamiltonian superoperator for the chemical shift.

    Parameters
    ----------
    spin_system : SpinSystem
        Spin system for which the chemical shift Hamiltonian is calculated.
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
    dim = spin_system.basis.dim
    nspins = spin_system.nspins

    # Initialize the Hamiltonian
    if config.sparse_superoperator:
        sop_Hcs = sp.csc_array((dim, dim), dtype=complex)
    else:
        sop_Hcs = np.zeros((dim, dim), dtype=complex)

    # Iterate over each spin
    for n in range(nspins):

        # Define the operator for the Z term of the nth spin
        op_def = np.array([2 if i == n else 0 for i in range(nspins)])

        # Compute the contribution from chemical shift for the current spin
        sop_Hcs = sop_Hcs - spin_system.gammas[n] * \
                            parameters.magnetic_field * \
                            spin_system.chemical_shifts[n] * 1e-6 * \
                            superoperator_from_op_def(spin_system, op_def, side)

    return sop_Hcs

def _hamiltonian_J(
    spin_system: SpinSystem,
    side: Literal["comm", "left", "right"] = "comm",
) -> np.ndarray | sp.csc_array:
    """
    Computes the J-coupling term of the Hamiltonian.

    Parameters
    ----------
    spin_system : SpinSystem
        Spin system whose J-coupling Hamiltonian is going to be calculated.
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
    dim = spin_system.basis.dim
    nspins = spin_system.nspins

    # Initialize the Hamiltonian
    if config.sparse_superoperator:
        sop_Hj = sp.csc_array((dim, dim), dtype=complex)
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
                sop_Hj += 2 * np.pi * spin_system.J_couplings[n][k] * (
                    superoperator_from_op_def(spin_system, op_def_00, side) -\
                    superoperator_from_op_def(spin_system, op_def_p1m1, side) -\
                    superoperator_from_op_def(spin_system, op_def_m1p1, side)
                )

    return sop_Hj

_INTERACTIONTYPE = Literal["zeeman", "chemical_shift", "J_coupling"]
_INTERACTIONDEFAULT = ["zeeman", "chemical_shift", "J_coupling"]
def hamiltonian(
    spin_system: SpinSystem,
    interactions: list[_INTERACTIONTYPE] = _INTERACTIONDEFAULT,
    side: Literal["comm", "left", "right"] = "comm"
) -> np.ndarray | sp.csc_array:
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
    time_start = time.time()
    print("Constructing Hamiltonian...")
        
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
        
    # Check that each item in the interactions list is unique
    if not len(set(interactions)) == len(interactions):
        raise ValueError("Cannot compute Hamiltonian, as duplicate "
                         "interactions were specified.")
    
    # Check that at least one interaction has been specified
    if len(interactions) == 0:
        raise ValueError("Cannot compute Hamiltonian, as no interactions were "
                         "specified.")

    # Obtain the basis set dimension
    dim = spin_system.basis.dim

    # Initialize the Hamiltonian
    if config.sparse_superoperator:
        H = sp.csc_array((dim, dim), dtype=complex)
    else:
        H = np.zeros((dim, dim), dtype=complex)

    # Compute the Zeeman and J-coupling Hamiltonians
    for interaction in interactions:
        if interaction == "zeeman":
            H += _hamiltonian_Z(spin_system, side)
        elif interaction == "chemical_shift":
            H += _hamiltonian_CS(spin_system, side)
        elif interaction == "J_coupling":
            H += _hamiltonian_J(spin_system, side)
        else:
            raise ValueError(
                f"Unsupported interaction type: {interaction}. "
                f"The possible options are: {_INTERACTIONDEFAULT}."
            )

    # Remove small values to enhance sparsity
    eliminate_small(H, config.zero_hamiltonian)

    print(f'Hamiltonian constructed in {time.time() - time_start:.4f} seconds.')
    print()

    return H