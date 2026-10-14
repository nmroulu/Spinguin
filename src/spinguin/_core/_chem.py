"""
Chemical-exchange helpers for Liouville-space simulations.

This module provides helper functions for simple chemical exchange operations
in Liouville space, including association, dissociation, and spin-order
permutation. These helpers also support non-linear chemical-kinetic
workflows. In addition, the module provides linear (Bloch-McConnell level)
exchange between sites with identical bases at chemical equilibrium, where the
exchange enters the Liouvillian of the ensemble directly.
"""

# Imports
from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np
import scipy.sparse as sp
from scipy.linalg import block_diag

from spinguin._core._la import arraylike_to_array
from spinguin._core._parameters import parameters
from spinguin._core._states import empty_state
from spinguin._core._validation import require

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
    
def _validate_elementary_reaction_spin_map(
    reactant_systems : list[SpinSystem],
    product_systems : list[SpinSystem],
    spin_map : dict[tuple[int, int], tuple[int, int]]
) -> None:
    """
    Validate a spin map used for an elementary reaction.

    Parameters
    ----------
    reactant_systems : list of SpinSystem
        List of spin systems for the reactants.
    product_systems : list of SpinSystem
        List of spin systems for the products.
    spin_map : dict of tuple of int to tuple of int
        Dictionary that maps the nuclear spins in the reactants to those in the
        products.    
    """
    # Define the exact sets of spins that must exist
    expected_reactants = set(
        (r, rs)
        for r, ss in enumerate(reactant_systems)
        for rs in range(ss.nspins)
    )
    expected_products = set(
        (p, ps)
        for p, ss in enumerate(product_systems)
        for ps in range(ss.nspins)
    )

    # Extract the user-provided mappings
    mapped_reactants = set(spin_map.keys())
    mapped_products_list = list(spin_map.values())
    mapped_products_set = set(mapped_products_list)

    # Check that the number of spins in the product and in the reactant match
    if len(expected_reactants) != len(expected_products):
        raise ValueError(
            "The number of reactant and product spins do not match"
        )

    # Check that every reactant spin is mapped
    if mapped_reactants != expected_reactants:
        raise ValueError("Not every reactant spin is mapped.")

    # Check that every product spin receives a map
    if mapped_products_set != expected_products:
        raise ValueError("Not every product spin is mapped.")


def _validate_exchange_sites(
    spin_systems: list[SpinSystem],
) -> None:
    """
    Validate that the spin systems of chemically exchanging sites have
    identical bases (error handling).

    Parameters
    ----------
    spin_systems : list of SpinSystem
        Spin systems of the exchanging sites.
    """

    # Ensure that the basis of each site has been built.
    for spin_system in spin_systems:
        require(spin_system, "basis.basis", "setting up chemical exchange")

    # Check that all sites contain the same isotopes in the same order.
    isotopes = spin_systems[0].isotopes
    for spin_system in spin_systems[1:]:
        if not np.array_equal(spin_system.isotopes, isotopes):
            raise ValueError(
                "all exchanging sites must contain the same isotopes in the "
                "same order"
            )

    # Check that all sites share the same basis set.
    basis = spin_systems[0].basis.basis
    for spin_system in spin_systems[1:]:
        if not np.array_equal(spin_system.basis.basis, basis):
            raise ValueError("all exchanging sites must have identical bases")


def _validate_exchange_matrix(
    K: np.ndarray,
    nsites: int,
) -> None:
    """
    Validate an exchange matrix of first-order rate constants
    (error handling).

    Parameters
    ----------
    K : ndarray
        Exchange matrix of shape (nsites, nsites).
    nsites : int
        Number of exchanging sites.
    """

    # Check that the exchange matrix has one row and column per site.
    if K.shape != (nsites, nsites):
        raise ValueError(
            "shape of the exchange matrix does not match the number of sites"
        )

    # Check that the off-diagonal rate constants are non-negative.
    off_diagonal = K - np.diag(np.diag(K))
    if np.any(off_diagonal < 0):
        raise ValueError(
            "off-diagonal elements of the exchange matrix must be "
            "non-negative"
        )

    # Check that the exchange conserves the total population.
    tolerance = 1e-10 * max(1, np.abs(K).max())
    if not np.allclose(K.sum(axis=0), 0, atol=tolerance):
        raise ValueError("columns of the exchange matrix must sum to zero")


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

@lru_cache(maxsize=16)
def _elementary_reaction_index_map(
    reactant_bases_bytes: tuple[tuple[bytes, int], ...],
    product_bases_bytes: tuple[tuple[bytes, int], ...],
    spin_map_tuple: tuple[tuple[int, int, int, int], ...]
) -> list[tuple[list[int], list[int]]]:
    """
    Generate index maps for an elementary reaction.

    Parameters
    ----------
    reactant_bases_bytes : tuple of tuple of bytes and int
        Reactant basis sets in bytes and the corresponding number of spins.
    product_bases_bytes : tuple of tuple of bytes and int
        Product basis sets in bytes and the corresponding number of spins.
    spin_map_tuple : tuple of tuple of int
        Spin map flattened into a format ((r, rs, p, ps), ...), where
        r and p are reactant and product system indices, and
        rs and ps are the corresponding spin indices.
    """
    # Convert the reactant and product bases from bytes to NumPy arrays.
    reactant_bases = [
        np.frombuffer(basis_bytes, dtype=int).reshape(-1, nspins)
        for basis_bytes, nspins in reactant_bases_bytes
    ]
    product_bases = [
        np.frombuffer(basis_bytes, dtype=int).reshape(-1, nspins)
        for basis_bytes, nspins in product_bases_bytes
    ]

    # Create lookups for the reactant bases to enable fast index retrieval
    lookups = [
        {tuple(row): idx for idx, row in enumerate(basis)}
        for basis in reactant_bases
    ]

    # Re-construct the spin map
    spin_map = {
        (r, p): []
        for r in range(len(reactant_bases_bytes))
        for p in range(len(product_bases_bytes))
    }
    for r, rs, p, ps in spin_map_tuple:
        spin_map[(r, p)].append((rs, ps))

    # Generate index maps from reactant basis sets to the product basis sets
    idx_maps = []
    for p, product_basis in enumerate(product_bases):
        idx_map_p = []
        idx_map_rp = []

        # For every product basis state, search for matching reactant states
        for idx_p, op_def_p in enumerate(product_basis):
            idx_rp = []
            for r, reactant_basis in enumerate(reactant_bases):
                
                # Build the corresponding reactant basis state 
                op_def_r = np.zeros(reactant_basis.shape[1], dtype=int)
                for rs, ps in spin_map[(r, p)]:
                    op_def_r[rs] = op_def_p[ps]
                op_def_r = tuple(op_def_r)

                # Store the index if the state exists
                if op_def_r in lookups[r]:
                    idx_rp.append(lookups[r][op_def_r])
                
                # Otherwise skip the current product basis state completely
                else:
                    break

            # Store the mapping if all reactant states were found
            else:
                idx_map_p.append(idx_p)
                idx_map_rp.append(idx_rp)

        # Transpose the reactant index map
        idx_map_rp = [list(row) for row in zip(*idx_map_rp)]
        
        # Store the mapping for the current product system
        idx_maps.append((idx_map_p, idx_map_rp))

    return idx_maps


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


def elementary_reaction(
    reactant_systems: SpinSystem | list[SpinSystem],
    product_systems: SpinSystem | list[SpinSystem],
    reactant_rhos: np.ndarray | sp.csc_array | list[np.ndarray | sp.csc_array],
    spin_map: dict[tuple[int, int], tuple[int, int]]
) -> np.ndarray | sp.csc_array | tuple[np.ndarray | sp.csc_array, ...]:
    """
    Perform an elementary chemical reaction that transforms reactant states into
    product states. If the reaction has stoichiometric coefficients greater than
    one, specify the same reactant or product system multiple times.

    Parameters
    ----------
    reactant_systems : SpinSystem or list of SpinSystem
        List of spin systems for the reactants.
    product_systems : SpinSystem or list of SpinSystem
        List of spin systems for the products.
    reactant_rhos : ndarray or csc_array or list of ndarray or csc_array
        List of state vectors for the reactants. The order should match that of
        `reactant_systems`.
    spin_map : dict of tuple of int to tuple of int
        Dictionary that maps the nuclear spins in the reactants to those in the
        products. The keys are tuples specifying the reactant index and the
        reactant spin index, and the values are identical tuples for the
        products. Example::

        spin_map = {
            (0, 0): (0, 0),  # reactant 0, spin 0 -> product 0, spin 0
            (0, 1): (1, 0),  # reactant 0, spin 1 -> product 1, spin 0
            (1, 0): (0, 1),  # reactant 1, spin 0 -> product 0, spin 1
            (1, 1): (1, 1),  # reactant 1, spin 1 -> product 1, spin 1
        }

    Returns
    -------
    rhos : ndarray or csc_array or tuple of ndarray or csc_array
        State vectors for the products. The order matches that of
        `product_systems`. A single state vector is returned if there is only
        one product system, and a tuple of state vectors is returned if there
        are multiple product systems.
    """
    # Convert single systems and states to lists for uniform processing
    if not isinstance(reactant_systems, (list, tuple)):
        reactant_systems = [reactant_systems]
    if not isinstance(product_systems, (list, tuple)):
        product_systems = [product_systems]
    if not isinstance(reactant_rhos, (list, tuple)):
        reactant_rhos = [reactant_rhos]

    # Validate that the spin map is consistent
    _validate_elementary_reaction_spin_map(
        reactant_systems,
        product_systems,
        spin_map
    )

    # Convert the reactant and product bases to bytes for hashing
    reactant_bases_bytes = tuple(
        (ss.basis.basis.tobytes(), ss.nspins)
        for ss in reactant_systems
    )
    product_bases_bytes = tuple(
        (ss.basis.basis.tobytes(), ss.nspins)
        for ss in product_systems
    )

    # Convert the dictionary to a flattened tuple for hashing
    spin_map_tuple = tuple(
        (r, rs, p, ps)
        for (r, rs), (p, ps) in spin_map.items()
    )

    # Generate the index maps
    idx_maps = _elementary_reaction_index_map(
        reactant_bases_bytes,
        product_bases_bytes,
        spin_map_tuple
    )

    # Calculate all product states
    product_rhos = []
    for p, product_system in enumerate(product_systems):
        idx_map_p, idx_map_rp = idx_maps[p]

        # Allocate an empty state vector for the product system
        rho_p = empty_state(product_system)

        # Populate the product state
        rho_p[idx_map_p, [0]] = reactant_rhos[0][idx_map_rp[0], [0]]
        for r in range(1, len(reactant_systems)):
            rho_p[idx_map_p, [0]] *= reactant_rhos[r][idx_map_rp[r], [0]]

        # Normalise the product state to preserve the unit operator scale
        rho_p /= (rho_p[0, 0] * np.sqrt(np.prod(product_system.mults)))

        # Convert to csc_array if using sparse
        if parameters.sparse_state:
            rho_p = rho_p.tocsc()
        
        # Store the product state
        product_rhos.append(rho_p)

    # A single state is returned if there is only one product system
    if len(product_rhos) == 1:
        return product_rhos[0]

    return tuple(product_rhos)


def equilibrium_concentrations(
    K_eq: float, 
    c_A0: float, 
    c_B0: float
) -> tuple[float, float, float]:
    """
    Calculate the equilibrium concentrations of species in a 
    chemical exchange system of the form A + B <-> AB.

    Solves the quadratic equation
        K_eq = [AB]_eq / ([A]_eq * [B]_eq)
    subject to the mass balance constraints
        [A]_eq + [AB]_eq = c_A0
        [B]_eq + [AB]_eq = c_B0,
    which gives the quadratic equation for [AB]_eq
        K_eq*[AB]_eq^2 - (K_eq*c_A0 + K_eq*c_B0 + 1)*[AB]_eq + K_eq*c_A0*c_B0 = 0.

    Parameters
    ----------
    K_eq : float
        Equilibrium constant for the reaction A + B <-> AB.
    c_A0 : float
        Initial concentration of species A.
    c_B0 : float
        Initial concentration of species B.

    Returns
    -------
    c_A_eq : float
        Equilibrium concentration of species A.
    c_B_eq : float
        Equilibrium concentration of species B.
    c_AB_eq : float
        Equilibrium concentration of species AB.
    """
    # Coefficients of the quadratic equation for [AB]_eq.
    a = K_eq
    b = -(K_eq*c_A0 + K_eq*c_B0 + 1)
    c = K_eq*c_A0*c_B0

    # Calculate the discriminant and check for non-negativity.
    discriminant = b**2 - 4*a*c
    if discriminant < 0:
        raise ValueError("No real solution for equilibrium concentrations.")

    # Calculate the two roots of the quadratic equation for [AB]_eq and
    # select the physically meaningful one that lies within the mass balance limits.
    root1 = (-b + np.sqrt(discriminant)) / (2*a)
    root2 = (-b - np.sqrt(discriminant)) / (2*a)
    c_AB_eq = None
    for root in (root1, root2):
        if 0 <= root <= min(c_A0, c_B0):
            c_AB_eq = root
            break

    if c_AB_eq is None:
        raise ValueError("No physically meaningful solution for equilibrium concentrations.")

    # Calculate the equilibrium concentrations of species A and B using mass balance.
    c_A_eq = c_A0 - c_AB_eq
    c_B_eq = c_B0 - c_AB_eq

    return c_A_eq, c_B_eq, c_AB_eq


def equilibrium_two_site_populations(
    K_eq: float, 
    c_A0: float,
    c_B0: float
) -> tuple[float, float]:
    """
    Calculate the equilibrium populations of sites A and AB in a chemically exchanging
    system of the form A + B <-> AB. Uses the `equilibrium_concentrations()` function 
    to compute the equilibrium concentrations and then converts these to populations
    by normalising with respect to the total concentration of A + AB.

    Parameters
    ----------
    K_eq : float
        Equilibrium constant for the reaction A + B <-> AB.
    c_A0 : float
        Initial concentration of species A.
    c_B0 : float
        Initial concentration of species B.

    Returns
    -------
    p_A_eq : float
        Equilibrium population of site A.
    p_AB_eq : float
        Equilibrium population of site AB.
    """
    # Calculate the equilibrium concentrations of species A, B, and AB.
    c_A_eq, _, c_AB_eq = equilibrium_concentrations(K_eq, c_A0, c_B0)

    # Calculate the total concentration of sites A and AB.
    c_total = c_A_eq + c_AB_eq

    # Calculate the equilibrium populations of sites A and AB by normalising
    # with respect to the total concentration.
    p_A_eq = c_A_eq / c_total
    p_AB_eq = c_AB_eq / c_total

    return p_A_eq, p_AB_eq


def two_site_exchange_matrix(
    k_ex: float,
    populations: list[float] | tuple[float, float] | np.ndarray,
) -> np.ndarray:
    """
    Construct the exchange matrix of two-site chemical exchange A <-> B at
    chemical equilibrium.

    The populations evolve as dP/dt = K P, where P = (p_A, p_B). The
    exchange matrix is

        K = k_ex * [[-p_B,  p_A],
                    [ p_B, -p_A]],

    so that the rate constant of A -> B is k_ex * p_B, the rate constant of
    B -> A is k_ex * p_A, and their sum is the total exchange rate k_ex. For
    the binding equilibrium A + B <-> AB with populations from
    `equilibrium_two_site_populations()`, k_ex * p_AB is the pseudo-first-order
    association rate constant k_on * [B] and k_ex * p_A is the dissociation
    rate constant k_off.

    Parameters
    ----------
    k_ex : float
        Total exchange rate in s^-1.
    populations : list or tuple or ndarray
        Equilibrium populations (p_A, p_B) of the two sites. These must sum to
        one.

    Returns
    -------
    K : ndarray
        Exchange matrix of shape (2, 2) in s^-1.
    """

    # Extract the equilibrium populations of the two sites.
    p_A, p_B = populations

    # Construct the population-conserving exchange matrix.
    K = k_ex * np.array([
        [-p_B, p_A],
        [p_B, -p_A],
    ])

    return K


def exchange_liouvillian(
    spin_systems: list[SpinSystem],
    liouvillians: list[np.ndarray | sp.csc_array],
    K: np.ndarray,
) -> np.ndarray | sp.csc_array:
    """
    Construct the Liouvillian of linear chemical exchange between sites that
    have identical spin systems and bases (Bloch-McConnell level).

    The state of the exchanging ensemble is the concatenation of the
    population-weighted state vectors of the sites (see `exchange_state()`).
    Its Liouvillian is

        L = diag(L_1, L_2, ..., L_S) + kron(K, 1),

    where L_s is the Liouvillian of site s, K is the exchange matrix and 1 is
    the unit matrix of the dimension of the basis. The chemical kinetics are
    assumed to be at equilibrium, so that the exchange is linear in the state.

    NOTE:
    A rotating-frame transformation with `rotating_frame()` and irradiation
    with `rotating_frame_irradiation_hamiltonian()` must be applied to the
    Liouvillian of each site before calling this function. The ``K`` argument
    of `liouvillian()` is not used for exchange between sites.

    Parameters
    ----------
    spin_systems : list of SpinSystem
        Spin systems of the exchanging sites. All sites must contain the same
        isotopes and have identical bases.
    liouvillians : list of ndarray or csc_array
        Liouvillians of the sites, in the same order as ``spin_systems``.
    K : ndarray
        Exchange matrix in s^-1, of shape (S, S) for S sites. The element
        K[i, j] is the first-order rate constant of the exchange from site j to
        site i, and each column sums to zero, so that the populations evolve
        as dP/dt = K P. For two sites, see `two_site_exchange_matrix()`.

    Returns
    -------
    L : ndarray or csc_array
        Liouvillian of the exchanging ensemble.

    Raises
    ------
    ValueError
        Raised if the sites do not have identical bases, if the number or the
        shapes of the Liouvillians do not match the sites, or if the exchange
        matrix is invalid.
    """

    # Validate the sites and the exchange matrix.
    _validate_exchange_sites(spin_systems)
    K = np.asarray(K, dtype=float)
    _validate_exchange_matrix(K, len(spin_systems))

    # Check that there is one Liouvillian of matching dimension per site.
    dim = spin_systems[0].basis.dim
    if len(liouvillians) != len(spin_systems):
        raise ValueError(
            "number of Liouvillians does not match the number of sites"
        )
    for L_site in liouvillians:
        if L_site.shape != (dim, dim):
            raise ValueError(
                "shape of a Liouvillian does not match the basis dimension"
            )

    # Combine the site Liouvillians and the exchange superoperator.
    if parameters.sparse_superoperator:
        L_sites = sp.block_diag(
            [sp.csc_array(L_site) for L_site in liouvillians],
            format="csc",
        )
        L_exchange = sp.kron(
            sp.csc_array(K),
            sp.eye_array(dim, format="csc"),
            format="csc",
        )
    else:
        L_sites = block_diag(*[
            L_site.toarray() if sp.issparse(L_site) else L_site
            for L_site in liouvillians
        ])
        L_exchange = np.kron(K, np.eye(dim))
    L = L_sites + L_exchange

    return L


def exchange_state(
    spin_systems: list[SpinSystem],
    rhos: list[np.ndarray | sp.csc_array],
    populations: list[float] | tuple[float, ...] | np.ndarray,
) -> np.ndarray | sp.csc_array:
    """
    Construct the state vector of an ensemble of chemically exchanging sites
    by concatenating the population-weighted state vectors of the sites.

    For the equilibrium state of the ensemble to be stationary under
    `exchange_liouvillian()`, the populations must satisfy K P = 0. The
    populations may also be given as concentrations, because only their
    ratios enter the dynamics. The expectation value of an operator in the
    ensemble is the sum of the expectation values in the weighted site states
    returned by `split_exchange_state()`.

    Parameters
    ----------
    spin_systems : list of SpinSystem
        Spin systems of the exchanging sites. All sites must contain the same
        isotopes and have identical bases.
    rhos : list of ndarray or csc_array
        Trace-normalised state vectors of the sites, in the same order as
        ``spin_systems``.
    populations : list or tuple or ndarray
        Populations of the sites, in the same order as ``spin_systems``.

    Returns
    -------
    rho : ndarray or csc_array
        Population-weighted state vector of the exchanging ensemble.

    Raises
    ------
    ValueError
        Raised if the sites do not have identical bases, or if the number of
        state vectors or populations does not match the number of sites.
    """

    # Validate the sites and the numbers of states and populations.
    _validate_exchange_sites(spin_systems)
    if len(rhos) != len(spin_systems) or len(populations) != len(spin_systems):
        raise ValueError(
            "numbers of state vectors and populations must match the number "
            "of sites"
        )

    # Weight the state vector of each site by its population.
    weighted_rhos = [p * rho_site for rho_site, p in zip(rhos, populations)]

    # Concatenate the weighted state vectors of the sites.
    if parameters.sparse_state:
        rho = sp.vstack(
            [sp.csc_array(rho_site) for rho_site in weighted_rhos],
            format="csc",
        )
    else:
        rho = np.concatenate([
            rho_site.toarray() if sp.issparse(rho_site) else rho_site
            for rho_site in weighted_rhos
        ])

    return rho


def split_exchange_state(
    spin_systems: list[SpinSystem],
    rho: np.ndarray | sp.csc_array,
) -> list[np.ndarray | sp.csc_array]:
    """
    Split the state vector of an ensemble of chemically exchanging sites into
    the population-weighted state vectors of the sites.

    The expectation value of an operator in a weighted site state, obtained
    with `measure()`, is the population-weighted contribution of that site to
    the expectation value of the ensemble.

    Parameters
    ----------
    spin_systems : list of SpinSystem
        Spin systems of the exchanging sites. All sites must contain the same
        isotopes and have identical bases.
    rho : ndarray or csc_array
        Population-weighted state vector of the exchanging ensemble.

    Returns
    -------
    rhos : list of ndarray or csc_array
        Population-weighted state vectors of the sites, in the same order as
        ``spin_systems``.

    Raises
    ------
    ValueError
        Raised if the sites do not have identical bases, or if the length of
        the state vector does not match the sites.
    """

    # Validate the sites and the length of the state vector.
    _validate_exchange_sites(spin_systems)
    dim = spin_systems[0].basis.dim
    if rho.shape[0] != len(spin_systems) * dim:
        raise ValueError(
            "length of the state vector does not match the number of sites "
            "and the basis dimension"
        )

    # Extract the block of each site from the concatenated state vector.
    rhos = [rho[s*dim:(s+1)*dim] for s in range(len(spin_systems))]

    return rhos


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