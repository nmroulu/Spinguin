"""
Chemical-exchange helpers for Liouville-space simulations.

This module provides helper functions for simple chemical exchange operations
in Liouville space, including association, dissociation, and spin-order
permutation. These helpers also support non-linear chemical-kinetic
workflows.
"""

# Imports
from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np
import scipy.sparse as sp

from spinguin._core._la import arraylike_to_array
from spinguin._core._parameters import parameters
from spinguin._core._states import empty_state

if TYPE_CHECKING:
    from spinguin._core._spin_system import SpinSystem


###############################################################################
# Helper functions
###############################################################################    

def _validate_bipartite_spin_maps(
    spin_system_A: SpinSystem,
    spin_system_B: SpinSystem,
    spin_system_C: SpinSystem,
    spin_map_A: list[int] | tuple[int, ...] | np.ndarray,
    spin_map_B: list[int] | tuple[int, ...] | np.ndarray,
) -> None:
    """
    Validate spin maps used for association and dissociation
    (error handling).

    Parameters
    ----------
    spin_system_A : SpinSystem
        First subsystem.
    spin_system_B : SpinSystem
        Second subsystem.
    spin_system_C : SpinSystem
        Composite system.
    spin_map_A : list or tuple or ndarray
        Indices of subsystem A within the composite system.
    spin_map_B : list or tuple or ndarray
        Indices of subsystem B within the composite system.
    """

    # Check that subsystem A has the expected number of mapped spins.
    if len(spin_map_A) != spin_system_A.nspins:
        raise ValueError(
            "length of spin_map_A does not match the number of spins "
            "in spin_system_A"
        )

    # Check that subsystem B has the expected number of mapped spins.
    if len(spin_map_B) != spin_system_B.nspins:
        raise ValueError(
            "length of spin_map_B does not match the number of spins "
            "in spin_system_B"
        )

    # Check that both maps form a complete non-overlapping partition of C.
    spin_map_C = set(spin_map_A).union(set(spin_map_B))
    if spin_map_C != set(range(spin_system_C.nspins)):
        raise ValueError(
            "spin maps contain incorrect indices or overlapping indices"
        )


def _validate_permutation_spin_map(
    spin_system: SpinSystem,
    spin_map: list[int] | tuple[int, ...] | np.ndarray,
) -> None:
    """
    Validate a spin map used for spin permutation (error handling).

    Parameters
    ----------
    spin_system : SpinSystem
        Spin system to be permuted.
    spin_map : list or tuple or ndarray
        Permuted spin indices.
    """

    # Check that the map is a valid permutation of all spin indices.
    if set(spin_map) != set(range(spin_system.nspins)):
        raise ValueError(
            "length of spin_map does not match the number of spins in "
            "the system or spin_map contains incorrect or overlapping indices"
        )


###############################################################################
# "Actual" functions
###############################################################################

@lru_cache(maxsize=16)
def _dissociate_index_map(
    basis_A_bytes: bytes,
    basis_B_bytes: bytes,
    basis_C_bytes: bytes,
    spin_map_A_bytes: bytes,
    spin_map_B_bytes: bytes,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate index maps from a composite basis to two subsystem bases.

    This helper is used internally by `dissociate()`.

    Example: Basis set C contains five spins, which are indexed as
    (0, 1, 2, 3, 4). We want to dissociate this into two subsystems A and B.
    Spins 0 and 2 should go to subsystem A and the rest to subsystem B. In this
    case, we define the following spin maps::

        spin_map_A = np.array([0, 2])
        spin_map_B = np.array([1, 3, 4])

    Parameters
    ----------
    basis_A_bytes : bytes
        Basis set for subsystem A converted to bytes.
    basis_B_bytes : bytes
        Basis set for subsystem B converted to bytes.
    basis_C_bytes : bytes
        Basis set for the composite system C converted to bytes.
    spin_map_A_bytes : bytes
        Indices of spin system A within spin system C converted to bytes.
    spin_map_B_bytes : bytes
        Indices of spin system B within spin system C converted to bytes.

    Returns
    -------
    index_map_A : ndarray
        Indices of states in A that also appear in C.
    index_map_CA : ndarray
        Corresponding indices of the matching elements in C. The array length is
        equal to `index_map_A`.
    index_map_B : ndarray
        Indices of states in B that also appear in C.
    index_map_CB : ndarray
        Corresponding indices of the matching elements in C. The array length is
        equal to `index_map_B`.
    """

    # Reconstruct the spin maps and basis arrays from cached byte strings.
    spin_map_A = np.frombuffer(spin_map_A_bytes, dtype=int)
    spin_map_B = np.frombuffer(spin_map_B_bytes, dtype=int)
    nspins_A = spin_map_A.shape[0]
    nspins_B = spin_map_B.shape[0]
    nspins_C = nspins_A + nspins_B
    basis_A = np.frombuffer(basis_A_bytes, dtype=int).reshape(-1, nspins_A)
    basis_B = np.frombuffer(basis_B_bytes, dtype=int).reshape(-1, nspins_B)
    basis_C = np.frombuffer(basis_C_bytes, dtype=int).reshape(-1, nspins_C)

    # Initialise the lists that store the subsystem-to-composite mappings.
    index_map_A = []
    index_map_CA = []
    index_map_B = []
    index_map_CB = []

    # Build a lookup table for basis states of the composite system.
    basis_C_lookup = {tuple(row): idx for idx, row in enumerate(basis_C)}

    # Map basis states of subsystem A into the composite basis.
    for idx_A, state in enumerate(basis_A):

        # Initialise the composite operator definition for the current state.
        op_def_C = np.zeros(nspins_C, dtype=int)

        # Place the subsystem operators into their composite-system positions.
        for op, idx in zip(state, spin_map_A):
            op_def_C[idx] = op

        # Convert the state definition to a hashable tuple for lookup.
        op_def_C = tuple(op_def_C)

        # Store the mapping when the state exists in the composite basis.
        if op_def_C in basis_C_lookup:
            idx_C = basis_C_lookup[op_def_C]
            index_map_CA.append(idx_C)
            index_map_A.append(idx_A)

    # Map basis states of subsystem B into the composite basis.
    for idx_B, state in enumerate(basis_B):

        # Initialise the composite operator definition for the current state.
        op_def_C = np.zeros(nspins_C, dtype=int)

        # Place the subsystem operators into their composite-system positions.
        for op, idx in zip(state, spin_map_B):
            op_def_C[idx] = op

        # Convert the state definition to a hashable tuple for lookup.
        op_def_C = tuple(op_def_C)

        # Store the mapping when the state exists in the composite basis.
        if op_def_C in basis_C_lookup:
            idx_C = basis_C_lookup[op_def_C]
            index_map_CB.append(idx_C)
            index_map_B.append(idx_B)

    # Convert the accumulated mappings to NumPy arrays.
    index_map_A = np.array(index_map_A)
    index_map_CA = np.array(index_map_CA)
    index_map_B = np.array(index_map_B)
    index_map_CB = np.array(index_map_CB)

    return index_map_A, index_map_CA, index_map_B, index_map_CB


@lru_cache(maxsize=16)
def _associate_index_map(
    basis_A_bytes: bytes,
    basis_B_bytes: bytes,
    basis_C_bytes: bytes,
    spin_map_A_bytes: bytes,
    spin_map_B_bytes: bytes,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate index maps from two subsystem bases to a composite basis.

    This helper is used internally by `associate()`.

    Example: We have spin system A that has two spins and spin system B that has
    three spins. These systems associate to form a composite spin system C that
    has five spins that are indexed (0, 1, 2, 3, 4). Of these, spins (0, 2) are
    from subsystem A and (1, 3, 4) from subsystem B. 
    In this case, the spin maps are defined as follows::

        spin_map_A = np.ndarray([0, 2])
        spin_map_B = np.ndarray([1, 3, 4])

    Parameters
    ----------
    basis_A_bytes : bytes
        Basis set for the subsystem A converted to bytes.
    basis_B_bytes : bytes
        Basis set for the subsystem B converted to bytes.
    basis_C_bytes : bytes
        Basis set for the composite system C converted to bytes.
    spin_map_A_bytes : bytes
        Indices of spin system A within spin system C converted to bytes.
    spin_map_B_bytes : bytes
        Indices of spin system B within spin system C converted to bytes.

    Returns
    -------
    index_map_A : ndarray
        Mapping of indices for spin system A.
    index_map_B : ndarray
        Mapping of indices for spin system B.
    index_map_C : ndarray
        Mapping of indices for spin system C.
    """

    # Reconstruct the spin maps and basis arrays from cached byte strings.
    spin_map_A = np.frombuffer(spin_map_A_bytes, dtype=int)
    spin_map_B = np.frombuffer(spin_map_B_bytes, dtype=int)
    nspins_A = spin_map_A.shape[0]
    nspins_B = spin_map_B.shape[0]
    nspins_C = nspins_A + nspins_B
    basis_A = np.frombuffer(basis_A_bytes, dtype=int).reshape(-1, nspins_A)
    basis_B = np.frombuffer(basis_B_bytes, dtype=int).reshape(-1, nspins_B)
    basis_C = np.frombuffer(basis_C_bytes, dtype=int).reshape(-1, nspins_C)

    # Initialise the lists that store the subsystem-to-composite mappings.
    index_map_A = []
    index_map_B = []
    index_map_C = []

    # Build lookup tables for the subsystem basis states.
    basis_A_lookup = {tuple(row): idx for idx, row in enumerate(basis_A)}
    basis_B_lookup = {tuple(row): idx for idx, row in enumerate(basis_B)}

    # Find composite states that can be assembled from both subsystems.
    for idx_C, state in enumerate(basis_C):

        # Extract the subsystem states that correspond to the current state.
        state_A = tuple(state[i] for i in spin_map_A)
        state_B = tuple(state[i] for i in spin_map_B)

        # Store the mapping when both subsystem states exist.
        if (state_A in basis_A_lookup) and (state_B in basis_B_lookup):
            index_map_A.append(basis_A_lookup[state_A])
            index_map_B.append(basis_B_lookup[state_B])
            index_map_C.append(idx_C)

    # Convert the accumulated mappings to NumPy arrays.
    index_map_A = np.array(index_map_A)
    index_map_B = np.array(index_map_B)
    index_map_C = np.array(index_map_C)

    return index_map_A, index_map_B, index_map_C


@lru_cache(maxsize=16)
def _permutation_matrix(
    basis_bytes: bytes,
    spin_map_bytes: bytes,
) -> sp.csc_array:
    """
    Build a cached permutation matrix for a specific basis and spin map.
    (See `permutation_matrix()` for the public interface.)

    Parameters
    ----------
    basis_bytes : bytes
        Basis array converted to bytes.
    spin_map_bytes : bytes
        Permutation map converted to bytes.

    Returns
    -------
    csc_array
        Sparse permutation matrix in CSC format.
    """

    # Reconstruct the spin map and basis array from cached byte strings.
    spin_map = np.frombuffer(spin_map_bytes, dtype=int)
    nspins = spin_map.shape[0]
    basis = np.frombuffer(basis_bytes, dtype=int).reshape(-1, nspins)

    # Obtain the dimension of the basis.
    dim = basis.shape[0]

    # Allocate the array that stores row indices after permutation.
    indices = np.empty(dim, dtype=int)

    # Build a lookup table for basis states in the original ordering.
    basis_lookup = {tuple(row): idx for idx, row in enumerate(basis)}

    # Find the destination index for every permuted basis state.
    for idx, state in enumerate(basis):

        # Permute the current state according to the supplied spin map.
        state_permuted = tuple(state[i] for i in spin_map)

        # Look up the permuted state in the original basis ordering.
        idx_permuted = basis_lookup[state_permuted]

        # Store the row index selected by the permutation.
        indices[idx] = idx_permuted

    # Start from the identity matrix in a row-addressable sparse format.
    perm = sp.eye_array(dim, dtype=int, format='lil')

    # Reorder the rows according to the permutation indices.
    perm = perm[indices]

    # Convert the matrix to CSC format for later multiplication.
    perm = perm.tocsc()

    return perm

def permutation_matrix(
    spin_system: SpinSystem,
    spin_map: list[int] | tuple[int, ...] | np.ndarray,
) -> sp.csc_array:
    """
    Create a permutation matrix that reorders spins in a spin system.

    Example: The spin system has three spins, which are indexed (0, 1, 2). 
    We want to perform the following permutation:

    - 0 --> 2 (Spin 0 goes to position 2)
    - 1 --> 0 (Spin 1 goes to position 0)
    - 2 --> 1 (Spin 2 goes to position 1)

    In this case, the spin map is defined as follows::

        spin_map = np.array([2, 0, 1])

    The permutation can be applied by::

        rho_permuted = perm @ rho

    Parameters
    ----------
    spin_system : SpinSystem
        The spin system for which the permutation matrix is going to be created.
    spin_map : list or tuple or ndarray
        Indices of the spins in the spin system after permutation.

    Returns
    -------
    perm : csc_array
        The permutation matrix.
    """

    # Convert the supplied spin map to a NumPy array.
    spin_map = arraylike_to_array(spin_map)

    # Convert the basis and the spin map to cached byte representations.
    basis_bytes = spin_system.basis.basis.tobytes()
    spin_map_bytes = spin_map.tobytes()

    # Return a copy so that callers cannot modify the cached matrix.
    perm = _permutation_matrix(basis_bytes, spin_map_bytes).copy()

    return perm

def dissociate(
    spin_system_A: SpinSystem,
    spin_system_B: SpinSystem,
    spin_system_C: SpinSystem,
    rho_C: np.ndarray | sp.csc_array,
    spin_map_A: list[int] | tuple[int, ...] | np.ndarray,
    spin_map_B: list[int] | tuple[int, ...] | np.ndarray,
) -> tuple[np.ndarray | sp.csc_array, np.ndarray | sp.csc_array]:
    """
    Dissociate a composite density vector into two subsystem vectors.

    This function models the chemical step or reaction `C -> A + B`.

    Example: Spin system C has five spins, which are indexed as (0, 1, 2, 3, 4).
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

    # Validate that the subsystem maps form a complete partition of C.
    _validate_bipartite_spin_maps(
        spin_system_A,
        spin_system_B,
        spin_system_C,
        spin_map_A,
        spin_map_B,
    )

    # Convert the supplied spin maps to NumPy arrays.
    spin_map_A = arraylike_to_array(spin_map_A)
    spin_map_B = arraylike_to_array(spin_map_B)

    # Convert basis sets and spin maps to bytes for hashing
    basis_A_bytes = spin_system_A.basis.basis.tobytes()
    basis_B_bytes = spin_system_B.basis.basis.tobytes()
    basis_C_bytes = spin_system_C.basis.basis.tobytes()
    spin_map_A_bytes = spin_map_A.tobytes()
    spin_map_B_bytes = spin_map_B.tobytes()

    # Obtain the cached index mappings for the dissociation step.
    idx_A, idx_CA, idx_B, idx_CB = _dissociate_index_map(
        basis_A_bytes,
        basis_B_bytes,
        basis_C_bytes,
        spin_map_A_bytes,
        spin_map_B_bytes,
    )

    # Allocate empty state vectors for the two product systems.
    rho_A = empty_state(spin_system_A)
    rho_B = empty_state(spin_system_B)

    # Transfer the matching composite populations into the subsystems.
    rho_A[idx_A, [0]] = rho_C[idx_CA, [0]]
    rho_B[idx_B, [0]] = rho_C[idx_CB, [0]]

    # Normalise the subsystem vectors to preserve the unit operator scale.
    rho_A = rho_A / (rho_A[0, 0] * np.sqrt(np.prod(spin_system_A.mults)))
    rho_B = rho_B / (rho_B[0, 0] * np.sqrt(np.prod(spin_system_B.mults)))

    # Convert to csc_array if using sparse.
    if parameters.sparse_state:
        rho_A = rho_A.tocsc()
        rho_B = rho_B.tocsc()

    return rho_A, rho_B

def associate(
    spin_system_A: SpinSystem,
    spin_system_B: SpinSystem,
    spin_system_C: SpinSystem,
    rho_A: np.ndarray | sp.csc_array,
    rho_B: np.ndarray | sp.csc_array,
    spin_map_A: list[int] | tuple[int, ...] | np.ndarray,
    spin_map_B: list[int] | tuple[int, ...] | np.ndarray,
) -> np.ndarray | sp.csc_array:
    """
    Combine two subsystem density vectors into a composite vector.

    This function models the chemical step or reaction `A + B -> C`.

    Example: We have spin system A that has two spins and spin system B that has
    three spins. These systems associate to form a composite spin system C that
    has five spins that are indexed (0, 1, 2, 3, 4). Of these, spins (0, 2) are
    from subsystem A and (1, 3, 4) from subsystem B. 
    In this case, the spin maps are defined as follows::

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

    # Validate that the subsystem maps form a complete partition of C.
    _validate_bipartite_spin_maps(
        spin_system_A,
        spin_system_B,
        spin_system_C,
        spin_map_A,
        spin_map_B,
    )

    # Convert the supplied spin maps to NumPy arrays.
    spin_map_A = arraylike_to_array(spin_map_A)
    spin_map_B = arraylike_to_array(spin_map_B)

    # Convert the arrays to bytes for hashing
    basis_A_bytes = spin_system_A.basis.basis.tobytes()
    basis_B_bytes = spin_system_B.basis.basis.tobytes()
    basis_C_bytes = spin_system_C.basis.basis.tobytes()
    spin_map_A_bytes = spin_map_A.tobytes()
    spin_map_B_bytes = spin_map_B.tobytes()

    # Obtain the cached index mappings for the association step.
    idx_A, idx_B, idx_C = _associate_index_map(
        basis_A_bytes,
        basis_B_bytes,
        basis_C_bytes,
        spin_map_A_bytes,
        spin_map_B_bytes,
    )

    # Allocate an empty state vector for the composite system.
    rho_C = empty_state(spin_system_C)

    # Combine the subsystem amplitudes in the composite basis.
    rho_C[idx_C, [0]] = rho_A[idx_A, [0]] * rho_B[idx_B, [0]]

    # Convert to csc_array if using sparse.
    if parameters.sparse_state:
        rho_C = rho_C.tocsc()

    return rho_C

def permute_spins(
    spin_system: SpinSystem,
    rho: np.ndarray | sp.csc_array,
    spin_map: list[int] | tuple[int, ...] | np.ndarray,
) -> np.ndarray | sp.csc_array:
    """
    Permute a state vector to match a reordering of spins.

    Example: Our spin system has three spins, which are indexed (0, 1, 2). We
    want to perform the following permutation:

    - 0 --> 2 (Spin 0 goes to position 2)
    - 1 --> 0 (Spin 1 goes to position 0)
    - 2 --> 1 (Spin 2 goes to position 1)

    In this case, the spin map is defined as follows::

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

    # Validate that the permutation map contains each spin exactly once.
    _validate_permutation_spin_map(spin_system, spin_map)

    # Build the permutation matrix for the requested spin ordering.
    perm = permutation_matrix(spin_system, spin_map)

    # Apply the permutation matrix to the state vector.
    rho = perm @ rho

    # Convert to sparse if the result is dense and sparsity is desired.
    if parameters.sparse_state and not sp.issparse(rho):
        rho = sp.csc_array(rho)
    if not parameters.sparse_state and sp.issparse(rho):
        rho = rho.toarray()

    return rho


def clear_cache_associate_index_map() -> None:
    """
    Clear the cache used by `_associate_index_map()`.
    """

    # Reset the cached association index maps.
    _associate_index_map.cache_clear()


def clear_cache_dissociate_index_map() -> None:
    """
    Clear the cache used by `_dissociate_index_map()`.
    """

    # Reset the cached dissociation index maps.
    _dissociate_index_map.cache_clear()


def clear_cache_permutation_matrix() -> None:
    """
    Clear the cache used by `_permutation_matrix()`.
    """

    # Reset the cached permutation matrices.
    _permutation_matrix.cache_clear()