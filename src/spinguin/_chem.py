"""
This module contains functions responsible for chemical kinetics.
"""

# Type checking
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import numpy as np
    import scipy.sparse as sp
    from spinguin._spin_system import SpinSystem

from spinguin._core.chem import (
    dissociate as _dissociate,
    associate as _associate,
    permute_spins as _permute_spins
)

def dissociate(spin_system_A: SpinSystem,
               spin_system_B: SpinSystem,
               spin_system_C: SpinSystem,
               rho_C: np.ndarray | sp.csc_array,
               spin_map_A: list | tuple | np.ndarray,
               spin_map_B: list | tuple | np.ndarray
               ) -> tuple[np.ndarray | sp.csc_array, np.ndarray | sp.csc_array]:
    """
    Dissociates the density vector of composite system C into density vectors of
    two subsystems A and B in a chemical reaction C -> A + B.

    Example. Spin system C has five spins, which are indexed as (0, 1, 2, 3, 4).
    We want to dissociate this into two subsystems A and B. Spins 0 and 2 should
    go to subsystem A and the rest to subsystem B. In this case, we define the
    following spin maps::

        spin_map_A = np.array([0, 2])
        spin_map_B = np.array([1, 3, 4])

    Parameters
    ----------
    spin_system_A : SpinSystem
        Spin system A.
    spin_system_B : SpinSystem
        Spin system B.
    spin_system_C : SpinSystem
        Spin system C.
    rho_C : ndarray or csc_array
        State vector of the composite spin system C.
    spin_map_A : list or tuple or ndarray
        Indices of spin system A within spin system C.
    spin_map_B : list or tuple or ndarray
        Indices of spin system B within spin system C.

    Returns
    -------
    rho_A : ndarray or csc_array
        State vector of spin system A.
    rho_B : ndarray or csc_array
        State vector of spin system B.
    """
    # Perform the dissociation
    rho_A, rho_B = _dissociate(
        basis_A = spin_system_A.basis.basis,
        basis_B = spin_system_B.basis.basis,
        basis_C = spin_system_C.basis.basis,
        spins_A = spin_system_A.spins,
        spins_B = spin_system_B.spins,
        rho_C = rho_C,
        spin_map_A = spin_map_A,
        spin_map_B = spin_map_B
    )

    return rho_A, rho_B

def associate(spin_system_A: SpinSystem,
              spin_system_B: SpinSystem,
              spin_system_C: SpinSystem,
              rho_A: np.ndarray | sp.csc_array,
              rho_B: np.ndarray | sp.csc_array,
              spin_map_A: list | tuple | np.ndarray,
              spin_map_B: list | tuple | np.ndarray
) -> np.ndarray | sp.csc_array:
    """
    Combines two state vectors when spin systems associate in a chemical
    reaction A + B -> C.

    Example. We have spin system A that has two spins and spin system B that has
    three spins. These systems associate to form a composite spin system C that
    has five spins that are indexed (0, 1, 2, 3, 4). Of these, spins (0, 2) are
    from subsystem A and (1, 3, 4) from subsystem B. We have to choose how the
    spin systems A and B will be indexed in spin system C by defining the spin
    maps as follows::

        spin_map_A = np.ndarray([0, 2])
        spin_map_B = np.ndarray([1, 3, 4])

    Parameters
    ----------
    spin_system_A : SpinSystem
        Spin system A.
    spin_system_B : SpinSystem
        Spin system B.
    spin_system_C : SpinSystem
        Spin system C.
    rho_A : ndarray or csc_array
        State vector of spin system A.
    rho_B : ndarray or csc_array
        State vector of spin system B.
    spin_map_A : list or tuple or ndarray
        Indices of spin system A within spin system C.
    spin_map_B : list or tuple or ndarray
        Indices of spin system B within spin system C.

    Returns
    -------
    rho_C : ndarray or csc_array
        State vector of the composite spin system C.
    """
    # Perform the association
    rho_C = _associate(
        basis_A = spin_system_A.basis.basis,
        basis_B = spin_system_B.basis.basis,
        basis_C = spin_system_C.basis.basis,
        rho_A = rho_A,
        rho_B = rho_B,
        spin_map_A = spin_map_A,
        spin_map_B = spin_map_B
    )

    return rho_C

def permute_spins(spin_system: SpinSystem,
                  rho: np.ndarray | sp.csc_array,
                  spin_map: list | tuple | np.ndarray
) -> np.ndarray | sp.csc_array:
    """
    Permutes the state vector of a spin system to correspond to a reordering
    of the spins in the system. 

    Example. Our spin system has three spins, which are indexed (0, 1, 2). We
    want to perform the following permulation:

    - 0 --> 2 (Spin 0 goes to position 2)
    - 1 --> 0 (Spin 1 goes to position 0)
    - 2 --> 1 (Spin 2 goes to position 1)

    In this case, we want to assign the following map::

        spin_map = np.array([2, 0, 1])

    Parameters
    ----------
    spin_system : SpinSystem
        The spin system whose density vector is going to be permuted.
    rho : ndarray or csc_array
        State vector of the spin system.
    spin_map : list or tuple or ndarray
        Indices of the spins in the spin system after permutation.

    Returns
    -------
    rho : ndarray or csc_array
        Permuted state vector of the spin system.
    """
    # Perform the permutation
    rho = _permute_spins(
        basis = spin_system.basis.basis,
        rho = rho,
        spin_map = spin_map
    )

    return rho