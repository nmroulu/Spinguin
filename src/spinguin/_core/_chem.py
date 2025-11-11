"""
This module contains functions responsible for chemical kinetics.
"""
# Referencing SpinSystem class
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from spinguin._core._spin_system import SpinSystem

# Imports
import numpy as np
import scipy.sparse as sp
from functools import lru_cache
from spinguin._core._la import arraylike_to_array
from spinguin._core._parameters import parameters

@lru_cache(maxsize=16)
def _dissociate_index_map(
    basis_A_bytes: bytes,
    basis_B_bytes: bytes,
    basis_C_bytes: bytes,
    spin_map_A_bytes: bytes,
    spin_map_B_bytes: bytes
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    
    # Convert bytes back to arrays
    spin_map_A = np.frombuffer(spin_map_A_bytes, dtype=int)
    spin_map_B = np.frombuffer(spin_map_B_bytes, dtype=int)
    nspins_A = spin_map_A.shape[0]
    nspins_B = spin_map_B.shape[0]
    nspins_C = nspins_A + nspins_B
    basis_A = np.frombuffer(basis_A_bytes, dtype=int).reshape(-1, nspins_A)
    basis_B = np.frombuffer(basis_B_bytes, dtype=int).reshape(-1, nspins_B)
    basis_C = np.frombuffer(basis_C_bytes, dtype=int).reshape(-1, nspins_C)

    # Create empty lists for the index maps
    index_map_A = []
    index_map_CA = []
    index_map_B = []
    index_map_CB = []

    # Make a dictionary of the C basis for fast lookup
    basis_C_lookup = {tuple(row): idx for idx, row in enumerate(basis_C)}

    # Loop over the basis set of spin system A
    for idx_A, state in enumerate(basis_A):

        # Initialize the state definition for spin system C
        op_def_C = np.zeros(nspins_C, dtype=int)

        # Map the states
        for op, idx in zip(state, spin_map_A):
            op_def_C[idx] = op

        # Convert to tuple for efficient searching
        op_def_C = tuple(op_def_C)

        # Find the index of this state in spin system C
        if op_def_C in basis_C_lookup:
            idx_C = basis_C_lookup[op_def_C]

            # Add the indices to the index maps
            index_map_CA.append(idx_C)
            index_map_A.append(idx_A)

    # Loop over the basis set of spin system B
    for idx_B, state in enumerate(basis_B):

        # Initialize the state definition for spin system C
        op_def_C = np.zeros(nspins_C, dtype=int)

        # Map the states
        for op, idx in zip(state, spin_map_B):
            op_def_C[idx] = op

        # Convert to tuple for efficient searching
        op_def_C = tuple(op_def_C)

        # Find the index of this state in spin system C
        if op_def_C in basis_C_lookup:
            idx_C = basis_C_lookup[op_def_C]

            # Add the indices to the index maps
            index_map_CB.append(idx_C)
            index_map_B.append(idx_B)

    # Convert the lists to NumPy arrays
    index_map_A = np.array(index_map_A)
    index_map_CA = np.array(index_map_CA)
    index_map_B = np.array(index_map_B)
    index_map_CB = np.array(index_map_CB)

    return index_map_A, index_map_CA, index_map_B, index_map_CB

def dissociate_index_map(
    basis_A: np.ndarray,
    basis_B: np.ndarray,
    basis_C: np.ndarray,
    spin_map_A: np.ndarray,
    spin_map_B: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates arrays that map the state indices from the basis set C to basis
    sets A and B. This function is used in `dissociate()`.

    Example. Basis set C contains five spins, which are indexed as
    (0, 1, 2, 3, 4). We want to dissociate this into two subsystems A and B.
    Spins 0 and 2 should go to subsystem A and the rest to subsystem B. In this
    case, we define the following spin maps::

        spin_map_A = np.array([0, 2])
        spin_map_B = np.array([1, 3, 4])

    Parameters
    ----------
    basis_A : ndarray
        Basis set for the subsystem A. It is a 2-dimensional array containing
        sequences of integers describing the Kronecker products of irreducible
        spherical tensors.
    basis_B : ndarray
        Basis set for the subsystem B. It is a 2-dimensional array containing
        sequences of integers describing the Kronecker products of irreducible
        spherical tensors.
    basis_C : ndarray
        Basis set for the composite system C. It is a 2-dimensional array
        containing sequences of integers describing the Kronecker products of
        irreducible spherical tensors.
    spin_map_A : ndarray
        Indices of spin system A within spin system C.
    spin_map_B : ndarray
        Indices of spin system B within spin system C.

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

    # Convert the arrays to bytes for hashing
    basis_A_bytes = basis_A.tobytes()
    basis_B_bytes = basis_B.tobytes()
    basis_C_bytes = basis_C.tobytes()
    spin_map_A_bytes = spin_map_A.tobytes()
    spin_map_B_bytes = spin_map_B.tobytes()

    # Acquire the index maps using the cached function
    index_map_A, index_map_CA, index_map_B, index_map_CB = \
        _dissociate_index_map(basis_A_bytes, basis_B_bytes, basis_C_bytes,
                              spin_map_A_bytes, spin_map_B_bytes)

    # Ensure that a different instance is returned
    index_map_A = index_map_A.copy()
    index_map_CA = index_map_CA.copy()
    index_map_B = index_map_B.copy()
    index_map_CB = index_map_CB.copy()

    return index_map_A, index_map_CA, index_map_B, index_map_CB

def _dissociate(
    basis_A: np.ndarray,
    basis_B: np.ndarray,
    basis_C: np.ndarray,
    spins_A : np.ndarray,
    spins_B : np.ndarray,
    rho_C: np.ndarray | sp.csc_array,
    spin_map_A: list | tuple | np.ndarray,
    spin_map_B: list | tuple | np.ndarray,
    sparse: bool
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
    basis_A : ndarray
        Basis set for the subsystem A. It is a 2-dimensional array containing
        sequences of integers describing the Kronecker products of irreducible
        spherical tensors.
    basis_B : ndarray
        Basis set for the subsystem B. It is a 2-dimensional array containing
        sequences of integers describing the Kronecker products of irreducible
        spherical tensors.
    basis_C : ndarray
        Basis set for the composite system C. It is a 2-dimensional array
        containing sequences of integers describing the Kronecker products of
        irreducible spherical tensors.
    spins_A : ndarray
        Spin quantum numbers for each spin in system A.
    spins_B : ndarray
        Spin quantum numbers for each spin in system B.
    rho_C : ndarray or csc_array
        State vector of the composite spin system C.
    spin_map_A : list or tuple or ndarray
        Indices of spin system A within spin system C.
    spin_map_B : list or tuple or ndarray
        Indices of spin system B within spin system C.
    sparse : bool
        Decides whether to return dense or sparse arrays.

    Returns
    -------
    rho_A : ndarray or csc_array
        State vector of spin system A.
    rho_B : ndarray or csc_array
        State vector of spin system B.
    """
    
    # Convert the spin maps to NumPy arrays
    spin_map_A = arraylike_to_array(spin_map_A)
    spin_map_B = arraylike_to_array(spin_map_B)

    # Obtain the basis set dimensions
    dim_A = basis_A.shape[0]
    dim_B = basis_B.shape[0]

    # Get spin multiplicities for normalization
    mults_A = (2*spins_A + 1).astype(int)
    mults_B = (2*spins_B + 1).astype(int)

    # Get index mappings
    idx_A, idx_CA, idx_B, idx_CB = \
        dissociate_index_map(basis_A, basis_B, basis_C, spin_map_A, spin_map_B)

    # Initialize empty state vectors
    if sparse:
        rho_A = sp.lil_array((dim_A, 1), dtype=complex)
        rho_B = sp.lil_array((dim_B, 1), dtype=complex)
    else:
        rho_A = np.zeros((dim_A, 1), dtype=complex)
        rho_B = np.zeros((dim_B, 1), dtype=complex)

    # Populate the state vectors
    rho_A[idx_A, [0]] = rho_C[idx_CA, [0]]
    rho_B[idx_B, [0]] = rho_C[idx_CB, [0]]

    # Normalize the state vectors
    rho_A = rho_A / (rho_A[0, 0] * np.sqrt(np.prod(mults_A)))
    rho_B = rho_B / (rho_B[0, 0] * np.sqrt(np.prod(mults_B)))

    # Convert to csc_array if using sparse
    if sparse:
        rho_A = rho_A.tocsc()
        rho_B = rho_B.tocsc()

    return rho_A, rho_B

@lru_cache(maxsize=16)
def _associate_index_map(
    basis_A_bytes: bytes,
    basis_B_bytes: bytes,
    basis_C_bytes: bytes,
    spin_map_A_bytes: bytes,
    spin_map_B_bytes: bytes
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    
    # Convert bytes back to arrays
    spin_map_A = np.frombuffer(spin_map_A_bytes, dtype=int)
    spin_map_B = np.frombuffer(spin_map_B_bytes, dtype=int)
    nspins_A = spin_map_A.shape[0]
    nspins_B = spin_map_B.shape[0]
    nspins_C = nspins_A + nspins_B
    basis_A = np.frombuffer(basis_A_bytes, dtype=int).reshape(-1, nspins_A)
    basis_B = np.frombuffer(basis_B_bytes, dtype=int).reshape(-1, nspins_B)
    basis_C = np.frombuffer(basis_C_bytes, dtype=int).reshape(-1, nspins_C)

    # Create empty lists for the index mappings
    index_map_A = []
    index_map_B = []
    index_map_C = []

    # Make a dictionary of the A and B basis sets for fast lookup
    basis_A_lookup = {tuple(row): idx for idx, row in enumerate(basis_A)}
    basis_B_lookup = {tuple(row): idx for idx, row in enumerate(basis_B)}

    # Loop over the basis states of spin system C
    for idx_C, state in enumerate(basis_C):

        # Extract the corresponding states of A and B
        state_A = tuple(state[i] for i in spin_map_A)
        state_B = tuple(state[i] for i in spin_map_B)

        # Only include states that exist in both A and B
        if (state_A in basis_A_lookup) and (state_B in basis_B_lookup):
            index_map_A.append(basis_A_lookup[state_A])
            index_map_B.append(basis_B_lookup[state_B])
            index_map_C.append(idx_C)

    # Convert to NumPy arrays
    index_map_A = np.array(index_map_A)
    index_map_B = np.array(index_map_B)
    index_map_C = np.array(index_map_C)

    return index_map_A, index_map_B, index_map_C

def associate_index_map(
    basis_A: np.ndarray,
    basis_B: np.ndarray,
    basis_C: np.ndarray,
    spin_map_A: np.ndarray,
    spin_map_B: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates arrays that map the state indices from spin systems A and B to the
    composite spin system C. This function is used in `associate()`.

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
    basis_A : ndarray
        Basis set for the subsystem A. It is a 2-dimensional array containing
        sequences of integers describing the Kronecker products of irreducible
        spherical tensors.
    basis_B : ndarray
        Basis set for the subsystem B. It is a 2-dimensional array containing
        sequences of integers describing the Kronecker products of irreducible
        spherical tensors.
    basis_C : ndarray
        Basis set for the composite system C. It is a 2-dimensional array
        containing sequences of integers describing the Kronecker products of
        irreducible spherical tensors.
    spin_map_A : ndarray
        Indices of spin system A within spin system C.
    spin_map_B : ndarray
        Indices of spin system B within spin system C.

    Returns
    -------
    index_map_A : ndarray
        Mapping of indices for spin system A.
    index_map_B : ndarray
        Mapping of indices for spin system B.
    index_map_C : ndarray
        Mapping of indices for spin system C.
    """

    # Convert the arrays to bytes for hashing
    basis_A_bytes = basis_A.tobytes()
    basis_B_bytes = basis_B.tobytes()
    basis_C_bytes = basis_C.tobytes()
    spin_map_A_bytes = spin_map_A.tobytes()
    spin_map_B_bytes = spin_map_B.tobytes()

    # Calculate the index maps using cached function
    index_map_A, index_map_B, index_map_C = \
        _associate_index_map(basis_A_bytes, basis_B_bytes, basis_C_bytes,
                             spin_map_A_bytes, spin_map_B_bytes)

    # Ensure that a different instance is returned
    index_map_A = index_map_A.copy()
    index_map_B = index_map_B.copy()
    index_map_C = index_map_C.copy()

    return index_map_A, index_map_B, index_map_C

def _associate(
    basis_A: np.ndarray,
    basis_B: np.ndarray,
    basis_C: np.ndarray,
    rho_A: np.ndarray | sp.csc_array,
    rho_B: np.ndarray | sp.csc_array,
    spin_map_A: list | tuple | np.ndarray,
    spin_map_B: list | tuple | np.ndarray,
    sparse: bool
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
    basis_A : ndarray
        Basis set for the subsystem A. It is a 2-dimensional array containing
        sequences of integers describing the Kronecker products of irreducible
        spherical tensors.
    basis_B : ndarray
        Basis set for the subsystem B. It is a 2-dimensional array containing
        sequences of integers describing the Kronecker products of irreducible
        spherical tensors.
    basis_C : ndarray
        Basis set for the composite system C. It is a 2-dimensional array
        containing sequences of integers describing the Kronecker products of
        irreducible spherical tensors.
    rho_A : ndarray or csc_array
        State vector of spin system A.
    rho_B : ndarray or csc_array
        State vector of spin system B.
    spin_map_A : list or tuple or ndarray
        Indices of spin system A within spin system C.
    spin_map_B : list or tuple or ndarray
        Indices of spin system B within spin system C.
    sparse : bool
        Decides whether to return a dense or a sparse array.

    Returns
    -------
    rho_C : ndarray or csc_array
        State vector of the composite spin system C.
    """

    # Convert the spin maps into NumPy
    spin_map_A = arraylike_to_array(spin_map_A)
    spin_map_B = arraylike_to_array(spin_map_B)

    # Acquire the C basis dimension
    dim_C = basis_C.shape[0]

    # Get the index mappings
    idx_A, idx_B, idx_C = \
        associate_index_map(basis_A, basis_B, basis_C, spin_map_A, spin_map_B)

    # Initialize an empty state vector for the composite system
    if sparse:
        rho_C = sp.lil_array((dim_C, 1), dtype=complex)
    else:
        rho_C = np.zeros((dim_C, 1), dtype=complex)

    # Combine the state vectors
    rho_C[idx_C, [0]] = rho_A[idx_A, [0]] * rho_B[idx_B, [0]]

    # Convert to csc_array if using sparse arrays
    if sparse:
        rho_C = rho_C.tocsc()

    return rho_C

@lru_cache(maxsize=16)
def _permutation_matrix(
    basis_bytes: bytes,
    spin_map_bytes: bytes
) -> sp.csc_array:
    
    # Convert bytes back to arrays
    spin_map = np.frombuffer(spin_map_bytes, dtype=int)
    nspins = spin_map.shape[0]
    basis = np.frombuffer(basis_bytes, dtype=int).reshape(-1, nspins)

    # Obtain the basis dimension
    dim = basis.shape[0]

    # Create an empty array for the permuted indices
    indices = np.empty(dim, dtype=int)

    # Make a dictionary of the basis set for fast lookup
    basis_lookup = {tuple(row): idx for idx, row in enumerate(basis)}

    # Loop through the basis set
    for idx, state in enumerate(basis):

        # Find the permuted state
        state_permuted = tuple(state[i] for i in spin_map)

        # Find the index of the permuted state
        idx_permuted = basis_lookup[state_permuted]

        # Add to the array of indices
        indices[idx] = idx_permuted

    # Initialize the permutation matrix
    perm = sp.eye_array(dim, dtype=int, format='lil')

    # Re-order the rows
    perm = perm[indices]

    # Convert to CSC
    perm = perm.tocsc()

    return perm

def permutation_matrix(
    basis: np.ndarray,
    spin_map: list | tuple | np.ndarray
) -> sp.csc_array:
    """
    Creates a permutation matrix to reorder the spins in the system.

    Example. Our spin system has three spins, which are indexed (0, 1, 2). We
    want to perform the following permulation:

    - 0 --> 2 (Spin 0 goes to position 2)
    - 1 --> 0 (Spin 1 goes to position 0)
    - 2 --> 1 (Spin 2 goes to position 1)

    In this case, we want to assign the following map::

        spin_map = np.array([2, 0, 1])

    The permutation can be applied by::

        rho_permuted = perm @ rho

    Parameters
    ----------
    basis : ndarray
        A 2-dimensional array containing the basis set that contains sequences
        of integers describing the Kronecker products of irreducible spherical
        tensors.
    spin_map : list or tuple or ndarray
        Indices of the spins in the spin system after permutation.

    Returns
    -------
    perm : csc_array
        The permutation matrix.
    """
    # Convert the spin map into NumPy
    spin_map = arraylike_to_array(spin_map)

    # Convert the arrays to bytes for hashing
    basis_bytes = basis.tobytes()
    spin_map_bytes = spin_map.tobytes()

    # Ensure that a separate copy is returned
    perm = _permutation_matrix(basis_bytes, spin_map_bytes).copy()

    return perm

def _permute_spins(
    basis: np.ndarray,
    rho: np.ndarray | sp.csc_array,
    spin_map: list | tuple | np.ndarray,
    sparse: bool
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
    basis : ndarray
        A 2-dimensional array containing the basis set that contains sequences
        of integers describing the Kronecker products of irreducible spherical
        tensors.
    rho : ndarray or csc_array
        State vector of the spin system.
    spin_map : list or tuple or ndarray
        Indices of the spins in the spin system after permutation.
    sparse : bool
        Specifies whether to return a dense or sparse array.

    Returns
    -------
    rho : ndarray or csc_array
        Permuted state vector of the spin system.
    """
    # Convert the spin map into NumPy
    spin_map = arraylike_to_array(spin_map)

    # Get the permutation matrix
    perm = permutation_matrix(basis, spin_map)

    # Apply the permutation to the density vector
    rho = perm @ rho

    # Ensure the correct return type
    if sparse and not sp.issparse(rho):
        rho = sp.csc_array(rho)
    if not sparse and sp.issparse(rho):
        rho = rho.toarray()

    return rho

def dissociate(
    spin_system_A: SpinSystem,
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
        spin_map_B = spin_map_B,
        sparse = parameters.sparse_state
    )

    return rho_A, rho_B

def associate(
    spin_system_A: SpinSystem,
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
        spin_map_B = spin_map_B,
        sparse = parameters.sparse_state
    )

    return rho_C

def permute_spins(
    spin_system: SpinSystem,
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
        spin_map = spin_map,
        sparse = parameters.sparse_state
    )

    return rho

def clear_cache_associate_index_map():
    """
    This function clears the cache from the `_associate_index_map()` function.
    """
    # Clear the cache
    _associate_index_map.cache_clear()

def clear_cache_dissociate_index_map():
    """
    This function clears the cache from the `_dissociate_index_map()` function.
    """
    # Clear the cache
    _dissociate_index_map.cache_clear()

def clear_cache_permutation_matrix():
    """
    This function clears the cache from the `_permutation_matrix()` function.
    """
    # Clear the cache
    _permutation_matrix.cache_clear()