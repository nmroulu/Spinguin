"""
This module contains functions that build a basis set.
"""
# Type checking
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Iterator

# Imports
import numpy as np
from itertools import combinations, product

def build_basis(spins: np.ndarray, max_spin_order: int):
    """
    Constructs a Liouville-space basis set, where the basis is spanned by all
    possible Kronecker products of irreducible spherical tensor operators, up
    to the defined maximum spin order.

    The Kronecker products themselves are not calculated. Instead, the operators
    are expressed as sequences of integers, where each integer represents a
    spherical tensor operator of rank `l` and projection `q` using the following
    relation: `N = l^2 + l - q`. The indexing scheme has been adapted from:

    Hogben, H. J., Hore, P. J., & Kuprov, I. (2010):
    https://doi.org/10.1063/1.3398146

    Parameters
    ----------
    spins : ndarray
        A one-dimensional array that specifies the spin quantum numbers of the
        spin system.
    max_spin_order : int
        Defines the maximum spin entanglement that is considered in the basis
        set.

    Returns
    -------
    basis : ndarray
        Two-dimensional array with dimensions (N, M), where N is the number of
        states in the basis and M is the number of spins in the system. The
        basis set is constructed from Kronecker products of irreducible
        spherical tensor operators, which are indexed using integers starting
        from 0 with increasing rank `l` and decreasing projection `q`:

        - 0 --> T(0, 0)
        - 1 --> T(1, 1)
        - 2 --> T(1, 0)
        - 3 --> T(1, -1) and so on...
    """

    # Find the number of spins in the system
    nspins = spins.shape[0]

    # Catch out-of-range maximum spin orders
    if max_spin_order < 1:
        raise ValueError("'max_spin_order' must be at least 1.")
    if max_spin_order > nspins:
        raise ValueError("'max_spin_order' must not be larger than number of"
                         "spins in the system.")

    # Get all possible subsystems of the specified maximum spin order
    indices = [i for i in range(nspins)]
    subsystems = combinations(indices, max_spin_order)

    # Create an empty dictionary for the basis set
    basis = {}

    # Iterate through all subsystems
    state_index = 0
    for subsystem in subsystems:

        # Get the basis for the subsystem
        sub_basis = _build_subsystem_basis(spins, subsystem)

        # Iterate through the states in the subsystem basis
        for state in sub_basis:

            # Add state to the basis set if not already added
            if state not in basis:
                basis[state] = state_index
                state_index += 1

    # Convert dictionary to NumPy array
    basis = np.array(list(basis.keys()))

    # Sort the basis (index of the first spin changes the slowest)
    sorted_indices = np.lexsort(
        tuple(basis[:, i] for i in reversed(range(basis.shape[1]))))
    basis = basis[sorted_indices]
    
    return basis

def _build_subsystem_basis(spins: np.ndarray, subsystem: tuple) -> Iterator:
    """
    Generates the basis set for a given subsystem.

    Parameters
    ----------
    spins : ndarray
        A one-dimensional array that specifies the spin quantum numbers of the
        spin system.
    subsystem : tuple
        Indices of the spins involved in the subsystem.

    Returns
    -------
    basis : Iterator
        An iterator over the basis set for the given subsystem, represented as
        tuples.

        For example, identity operator and z-operator for the 3rd spin:
        `[(0, 0, 0), (0, 0, 2), ...]`
    """

    # Extract the necessary information from the spin system
    nspins = spins.shape[0]
    mults = (2*spins + 1).astype(int)

    # Define all possible spin operators for each spin
    operators = []

    # Loop through every spin in the full system
    for spin in range(nspins):

        # Add spin if it exists in the subsystem
        if spin in subsystem:

            # Add all possible states of the given spin
            operators.append(list(range(mults[spin] ** 2)))

        # Add identity state if not
        else:
            operators.append([0])

    # Get all possible product operator states in the subsystem
    basis = product(*operators)

    return basis