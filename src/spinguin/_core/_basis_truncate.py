"""
This module contains functions for truncating the basis set.
"""
# Type checking
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import scipy.sparse as sp

# Imports
import numpy as np
import time
import math
from typing import Literal
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree
from spinguin._core._basis_indexing import coherence_order
from spinguin._core._hide_prints import HidePrints
from spinguin._core._la import isvector, eliminate_small, expm_vec

def truncate_basis_by_coherence(
    basis: np.ndarray,
    coherence_orders: list
) -> tuple[np.ndarray, list]:
    """
    Truncates the basis set by retaining only the product operators that
    correspond to coherence orders specified in the `coherence_orders` list.

    The function generates an index map from the original basis to the truncated
    basis.
    This map can be used to transform superoperators or state vectors to the new
    basis.

    Parameters
    ----------
    basis : ndarray
        A two-dimensional array where each row contains integers that represent
        a Kronecker product of single-spin irreducible spherical tensors.
    coherence_orders : list
        List of coherence orders to be retained in the basis.

    Returns
    -------
    truncated_basis : ndarray
        A two-dimensional array containing the basis set with only the specified
        coherence orders retained.
    index_map : list
        List that contains an index map from the original basis to the truncated
        basis.
    """

    print("Truncating the basis set. The following coherence orders are "
          f"retained: {coherence_orders}")
    time_start = time.time()

    # Create an empty list for the new basis
    truncated_basis = []

    # Create an empty list for the mapping from old to new basis
    index_map = []

    # Iterate over the basis
    for idx, state in enumerate(basis):

        # Check if coherence order is in the list
        if coherence_order(state) in coherence_orders:

            # Assign state to the truncated basis and increment index
            truncated_basis.append(state)

            # Assign index to the index map
            index_map.append(idx)

    # Convert basis to NumPy array
    truncated_basis = np.array(truncated_basis)

    print("Truncated basis created.")
    print(f"Original dimension: {len(basis)}")
    print(f"Truncated dimension: {len(truncated_basis)}")
    print(f"Elapsed time: {time.time() - time_start:.4f} seconds.")
    print()

    return truncated_basis, index_map

def _truncate_basis_by_coupling_weakest_link(
    basis: np.ndarray,
    J_couplings: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, list]:
    """
    Removes basis states based on the scalar J-couplings. Whenever there exists
    a coupling network between the spins that constitute the product state, in
    which the couplings surpass the given threshold, the basis state is kept.
    Otherwise, the basis state is dropped.

    Parameters
    ----------
    basis : ndarray
        A two-dimensional array where each row contains integers that represent
        a Kronecker product of single-spin irreducible spherical tensors.
    J_couplings : ndarray
        A two-dimensional array that contains the scalar J-couplings between
        the spins in Hz.
    threshold : float
        J-coupling between two spins must be above this value in order for the
        algorithm to consider them connected.

    Returns
    -------
    truncated_basis : ndarray
        A two-dimensional array containing the truncated basis set.
    index_map : list
        List that contains an index map from the original basis to the truncated
        basis.
    """
    print("Truncating the basis set based on J-couplings.")
    time_start = time.time()

    # Create an empty list for the new basis
    truncated_basis = []

    # Create an empty list for the mapping from old to new basis
    index_map = []

    # Create the connectivity matrix from the J-couplings
    J_connectivity = J_couplings.copy()
    eliminate_small(J_connectivity, zero_value=threshold)
    J_connectivity[J_connectivity!=0] = 1

    # Cache the connectivity of spins
    connectivity = dict()

    # Iterate over the basis
    for idx, state in enumerate(basis):

        # Obtain the indices of the participating spins
        idx_spins = np.nonzero(state)[0]

        # Special case always include the unit state:
        if len(idx_spins) == 0:
            truncated_basis.append(state)
            index_map.append(idx)
            continue

        # Analyse the connectivity if not already
        if not tuple(idx_spins) in connectivity:

            # Obtain the current connectivity graph
            J_connectivity_curr = J_connectivity[np.ix_(idx_spins, idx_spins)]

            # Calculate the number of connected components
            n_components = connected_components(
                csgraph = J_connectivity_curr,
                directed = False,
                return_labels = False
            )

            connectivity[tuple(idx_spins)] = n_components

        # If the state is connected, keep it
        if connectivity[tuple(idx_spins)] == 1:
            truncated_basis.append(state)
            index_map.append(idx)

    # Convert basis to NumPy array
    truncated_basis = np.array(truncated_basis)

    print("Truncated basis created.")
    print(f"Original dimension: {len(basis)}")
    print(f"Truncated dimension: {len(truncated_basis)}")
    print(f"Elapsed time: {time.time() - time_start:.4f} seconds.")
    print()

    return truncated_basis, index_map

def _truncate_basis_by_coupling_network_strength(
    basis: np.ndarray,
    J_couplings: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, list]:
    """
    Removes basis states based on the scalar J-couplings. The coupling network
    within a product state is evaluated based on the maximum overall coupling
    strength defined as the geometric mean of the J-couplings divided by the
    factorial of the number of the couplings.
    
    TODO: More rigorous way to estimate the network strength?

    Parameters
    ----------
    basis : ndarray
        A two-dimensional array where each row contains integers that represent
        a Kronecker product of single-spin irreducible spherical tensors.
    J_couplings : ndarray
        A two-dimensional array that contains the scalar J-couplings between
        the spins in Hz.
    threshold : float
        Calculated effective J-coupling network strength must be above this
        value in order for the algorithm to consider them connected.

    Returns
    -------
    truncated_basis : ndarray
        A two-dimensional array containing the truncated basis set.
    index_map : list
        List that contains an index map from the original basis to the truncated
        basis.
    """
    print("Truncating the basis set based on J-couplings.")
    time_start = time.time()

    # Create an empty list for the new basis
    truncated_basis = []

    # Create an empty list for the mapping from old to new basis
    index_map = []

    # Prepare the coupling matrix for the Kruskal's algorithm
    J_couplings = -np.abs(J_couplings)

    # Iterate over the basis
    for idx, state in enumerate(basis):

        # Obtain the indices of the participating spins
        idx_spins = np.nonzero(state)[0]

        # Special case: always include the unit state and one-spin states:
        if len(idx_spins) in {0, 1}:
            truncated_basis.append(state)
            index_map.append(idx)
            continue

        # Obtain the current J-coupling network
        J_couplings_curr = J_couplings[np.ix_(idx_spins, idx_spins)]

        # Obtain the maximum spanning tree
        maxtree = abs(minimum_spanning_tree(J_couplings_curr))

        # Continue only if the state is connected in the first place
        connections = len(idx_spins) - 1
        if maxtree.nnz == connections:

            # Calculate the coupling strength
            geomean = np.prod(maxtree.data) ** (1/connections)
            coupling_strength = geomean / math.factorial(connections)

            # Include the state if the coupling strength is above threshold
            if coupling_strength >= threshold:
                truncated_basis.append(state)
                index_map.append(idx)

    # Convert basis to NumPy array
    truncated_basis = np.array(truncated_basis)

    print("Truncated basis created.")
    print(f"Original dimension: {len(basis)}")
    print(f"Truncated dimension: {len(truncated_basis)}")
    print(f"Elapsed time: {time.time() - time_start:.4f} seconds.")
    print()

    return truncated_basis, index_map

def truncate_basis_by_coupling(
    basis: np.ndarray,
    J_couplings: np.ndarray,
    threshold: float,
    method: Literal["weakest_link", "network_strength"] = "weakest_link"
) -> tuple[np.ndarray, list]:
    """
    Removes basis states based on the scalar J-couplings. Whenever there exists
    a J-coupling network of sufficient strength between spins that constitute a
    product state, the particular state is kept in the basis. Otherwise, the
    state is removed. The coupling strength is evaluated either by the weakest
    link or by the overall network strength.

    Parameters
    ----------
    basis : ndarray
        A two-dimensional array where each row contains integers that represent
        a Kronecker product of single-spin irreducible spherical tensors.
    J_couplings : ndarray
        A two-dimensional array that contains the scalar J-couplings between
        the spins in Hz.
    threshold : float
        Coupling strength must be above this value in order for the product
        state to be considered in the basis set.
    method : {"weakest_link", "network_strength"}
        Decides how the importance of a product state is evaluated. Weakest link
        method considers a J-coupling network invalid based on the smallest J-
        coupling within that network. Network strength method calculates the
        effective coupling as a geometric mean scaled by the factorial of the
        number of couplings within the network.

    Returns
    -------
    truncated_basis : ndarray
        A two-dimensional array containing the truncated basis set.
    index_map : list
        List that contains an index map from the original basis to the truncated
        basis.
    """
    if method == "weakest_link":
        return _truncate_basis_by_coupling_weakest_link(
            basis = basis,
            J_couplings = J_couplings,
            threshold = threshold
        )
    elif method == "network_strength":
        return _truncate_basis_by_coupling_network_strength(
            basis = basis,
            J_couplings = J_couplings,
            threshold = threshold
        )
    else:
        raise ValueError(f"Invalid truncation method {method}.")
    
def truncate_basis_by_zte(
    basis: np.ndarray,
    L: np.ndarray | sp.csc_array,
    rho: np.ndarray | sp.csc_array,
    time_step: float,
    nsteps: int,
    zero_zte: float,
    zero_expm_vec: float
) -> tuple[np.ndarray, list]:
    """
    Removes basis states using the Zero-Track Elimination (ZTE) described in:

    Kuprov, I. (2008):
    https://doi.org/10.1016/j.jmr.2008.08.008

    Parameters
    ----------
    basis : ndarray
        A two-dimensional array where each row contains integers that represent
        a Kronecker product of single-spin irreducible spherical tensors.
    L : ndarray or csc_array
        Liouvillian superoperator, L = -iH - R + K.
    rho : ndarray or csc_array
        Initial spin density vector.
    time_step : float
        Time step of the propagation within the ZTE.
    nsteps : int
        Number of steps to take in the ZTE.
    zero_zte : float
        If state population is below this value, it is dropped from the basis.
    zero_expm_vec: float
        Convergence criterion to be used when calculating the action of matrix
        exponential of the Liouvillian to the state vector.

    Returns
    -------
    truncated_basis : ndarray
        A two-dimensional array containing the truncated basis set.
    index_map : list
        List that contains an index map from the original basis to the truncated
        basis.
    """
    print("Truncating the basis set using zero-track elimination.")
    time_start = time.time()

    # Create empty vector for the maximum values of rho
    rho_max = abs(np.array(rho))

    # Scale the zero value of the ZTE to take into account different norms
    scaling_zv = abs(rho).max()
    zero_zte = zero_zte / scaling_zv

    # Propagate for few steps
    for i in range(nsteps):
        print(f"ZTE step {i+1} of {nsteps}...")
        with HidePrints():
            rho = expm_vec(L*time_step, rho, zero_expm_vec)
            rho_max = np.maximum(rho_max, abs(rho))

    # Obtain indices of states that should remain
    index_map = list(np.where(rho_max > zero_zte)[0])

    # Obtain the truncated basis
    truncated_basis = basis[index_map]

    print("Truncated basis created.")
    print(f"Original dimension: {len(basis)}")
    print(f"Truncated dimension: {len(truncated_basis)}")
    print(f"Elapsed time: {time.time() - time_start:.4f} seconds.")
    print()

    return truncated_basis, index_map

def truncate_basis_by_indices(
    basis: np.ndarray,
    indices: list | np.ndarray
) -> np.ndarray:
    """
    Truncate the basis set to include only the basis states specified by the
    `indices` supplied by the user.

    Parameters
    ----------
    basis : ndarray
        A two-dimensional array where each row contains integers that represent
        a Kronecker product of single-spin irreducible spherical tensors.
    indices : list or ndarray
        List of indices that specify which basis states to retain.

    Returns
    -------
    truncated_basis : ndarray
        A two-dimensional array containing the truncated basis set.
    """
    print("Truncating the basis set based on supplied indices.")
    time_start = time.time()

    # Sort the indices
    indices = np.sort(indices)

    # Obtain the truncated basis
    truncated_basis = basis[indices]

    print("Truncated basis created.")
    print(f"Original dimension: {len(basis)}")
    print(f"Truncated dimension: {len(truncated_basis)}")
    print(f"Elapsed time: {time.time() - time_start:.4f} seconds.")
    print()

    return truncated_basis

def _state_to_truncated_basis(
    index_map: list,
    rho: np.ndarray | sp.csc_array
) -> np.ndarray | sp.csc_array:
    """
    Transforms a state vector to a truncated basis using the `index_map`,
    which contains indices that determine the elements that are retained
    after the transformation.

    Parameters
    ----------
    index_map : list
        Index mapping from the original basis to the truncated basis.
    rho : ndarray or csc_array
        State vector to be transformed.

    Returns
    -------
    rho_transformed : ndarray or csc_array
        State vector transformed into the truncated basis.
    """

    print("Transforming the state vector into the truncated basis.")
    time_start = time.time()

    # Perform the transformation to truncated basis
    rho_transformed = rho[index_map]

    print("Transformation completed.")
    print(f"Elapsed time: {time.time() - time_start:.4f} seconds.")
    print()

    return rho_transformed

def _superoperator_to_truncated_basis(
    index_map: list,
    sop: np.ndarray | sp.csc_array
) -> np.ndarray | sp.csc_array:
    """
    Transforms a superoperator to a truncated basis using the `index_map`,
    which contains indices that determine the elements that are retained
    after the transformation.

    Parameters
    ----------
    index_map : list
        Index mapping from the original basis to the truncated basis.
    sop : ndarray or csc_array
        Superoperators to be transformed.

    Returns
    -------
    sop_transformed : ndarray or csc_array
        Superoperator transformed into the truncated basis.
    """

    print("Transforming the superoperator into the truncated basis.")
    time_start = time.time()

    # Perform the transformation to truncated basis
    sop_transformed = sop[np.ix_(index_map, index_map)]

    print("Transformation completed.")
    print(f"Elapsed time: {time.time() - time_start:.4f} seconds.")
    print()

    return sop_transformed

def truncate_objects(objs, index_map):
    objs_transformed = []
    for obj in objs:

        # Consider state vectors
        if isvector(obj):
            objs_transformed.append(_state_to_truncated_basis(
                index_map=index_map,
                rho=obj
            ))
            
        # Consider superoperators
        else:
            objs_transformed.append(_superoperator_to_truncated_basis(
                index_map=index_map,
                sop=obj
            ))

    # Convert to tuple or just single value
    if len(objs_transformed) == 1:
        objs_transformed = objs_transformed[0]
    else:
        objs_transformed = tuple(objs_transformed)

    return objs_transformed