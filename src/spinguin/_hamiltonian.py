"""
This module provides functions for calculating Hamiltonian superoperators.
"""
# Type checking
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import numpy as np
    import scipy.sparse as sp
    from spinguin._core._spin_system import SpinSystem

# Imports
from typing import Literal
from spinguin._config import config
from spinguin._parameters import parameters
from spinguin._core.hamiltonian import sop_H as _sop_H

INTERACTIONTYPE = Literal["zeeman", "chemical_shift", "J_coupling"]
INTERACTIONDEFAULT = ["zeeman", "chemical_shift", "J_coupling"]
def hamiltonian(
        spin_system: SpinSystem,
        interactions: list[INTERACTIONTYPE] = INTERACTIONDEFAULT,
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
        
    H = _sop_H(
        basis = spin_system.basis.basis,
        spins = spin_system.spins,
        gammas = spin_system.gammas,
        B = parameters.magnetic_field,
        chemical_shifts = spin_system.chemical_shifts,
        J_couplings = spin_system.J_couplings,
        interactions = interactions,
        side = side,
        sparse = config.sparse_hamiltonian,
        zero_value = config.zero_hamiltonian
    )

    return H