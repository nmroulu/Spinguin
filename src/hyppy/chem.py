"""
chem.py

This module contains functions that are responsible for the chemical kinetics.
"""
# For referencing the SpinSystem class
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hyppy.spin_system import SpinSystem

# Imports
import numpy as np
from scipy.sparse import lil_array, eye_array, csc_array
from functools import lru_cache
from typing import Tuple, Union
from hyppy.basis import state_idx

@lru_cache(maxsize=16)
def dissociate_index_map(spin_system_A:SpinSystem,
                         spin_system_B:SpinSystem,
                         spin_system_C:SpinSystem,
                         spin_map_A:tuple,
                         spin_map_B:tuple) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates arrays that map the state indices from spin system C to
    the spin systems A and B. This function is called from dissociate().

    Parameters
    ----------
    spin_system_A : SpinSystem
    spin_system_B : SpinSystem
    spin_system_C : SpinSystem
    spin_map_A : tuple
        Indices of spin system A in spin system C.
    spin_map_B : tuple
        Indices of spin system B in spin system C.

    Returns
    -------
    index_map_A : numpy.ndarray
    index_map_CA : numpy.ndarray
    index_map_B : numpy.ndarray
    index_map_CB : numpy.ndarray 
    """

    # Make empty arrays for the index maps
    index_map_A = np.zeros(spin_system_C.basis.dim, dtype=int)
    index_map_CA = np.zeros(spin_system_C.basis.dim, dtype=int)
    index_map_B = np.zeros(spin_system_C.basis.dim, dtype=int)
    index_map_CB = np.zeros(spin_system_C.basis.dim, dtype=int)

    # Loop over the basis set of spin system A
    for i, (state, idx_A) in enumerate(spin_system_A.basis.dict.items()):

        # Initialize the state definition
        op_def_C = np.zeros(spin_system_C.size, dtype=int)

        # Set the states
        for op, idx in zip(state, spin_map_A):
            op_def_C[idx] = op

        # Convert to tuple
        op_def_C = tuple(op_def_C)

        # Find the index of this state in spin system C
        index_map_CA[i] = state_idx(spin_system_C, op_def_C)
        index_map_A[i] = idx_A

    # Loop over the basis set of spin system B
    for i, (state, idx_B) in enumerate(spin_system_B.basis.dict.items()):

        # Initialize the state definition
        op_def_C = np.zeros(spin_system_C.size, dtype=int)

        # Set the states
        for op, idx in zip(state, spin_map_B):
            op_def_C[idx] = op

        # Convert to tuple
        op_def_C = tuple(op_def_C)

        # Find the index of this state in spin system C
        index_map_CB[i] = state_idx(spin_system_C, op_def_C)
        index_map_B[i] = idx_B

    return index_map_A, index_map_CA, index_map_B, index_map_CB

def dissociate(spin_system_A: SpinSystem,
               spin_system_B: SpinSystem,
               spin_system_C: SpinSystem,
               rho_C: Union[np.ndarray, csc_array],
               spin_map_A: tuple,
               spin_map_B: tuple) -> Tuple[Union[np.ndarray, csc_array], Union[np.ndarray, csc_array]]:
    """
    This function is used in chemical reactions to dissociate spins C -> A + B.
    Spin system C is thought to be the composite system of A and B.

    Parameters
    ----------
    spin_system_A : SpinSystem
    spin_system_B : SpinSystem
    spin_system_C : SpinSystem
    rho_C : numpy.ndarray or csc_array
        Density vector of spin system C.
    spin_map_A : tuple
        Indices of spin system A in spin system C.
    spin_map_B : tuple
        Indices of spin system B in spin system C.

    Returns
    -------
    rho_A : numpy.ndarray or csc_array
        Density vector of spin system A.
    rho_B : numpy.ndarray or csc_array
        Density vector of spin system B.
    """

    # Get spin multiplicities for normalization
    mults_A = spin_system_A.mults
    mults_B = spin_system_B.mults

    # Get index maps
    idx_A, idx_CA, idx_B, idx_CB = dissociate_index_map(spin_system_A, spin_system_B, spin_system_C, spin_map_A, spin_map_B)

    # Initialize empty state vectors
    rho_A = lil_array((spin_system_A.basis.dim, 1), dtype=complex)
    rho_B = lil_array((spin_system_B.basis.dim, 1), dtype=complex)

    # Convert to NumPy if needed
    if isinstance(rho_C, np.ndarray):
        rho_A = rho_A.toarray()
        rho_B = rho_B.toarray()

    # Set the value
    rho_A[idx_A, [0]] = rho_C[idx_CA, [0]]
    rho_B[idx_B, [0]] = rho_C[idx_CB, [0]]

    # Normalize
    rho_A = rho_A / (rho_A[0,0] * np.sqrt(np.prod(mults_A)))
    rho_B = rho_B / (rho_B[0,0] * np.sqrt(np.prod(mults_B)))

    # Convert to csc_array if needed
    if not isinstance(rho_C, np.ndarray):
        rho_A = rho_A.tocsc()
        rho_B = rho_B.tocsc()

    return rho_A, rho_B

@lru_cache(maxsize=16)
def associate_index_map(spin_system_A: SpinSystem,
                        spin_system_B: SpinSystem,
                        spin_system_C: SpinSystem,
                        spin_map_A: tuple,
                        spin_map_B: tuple) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates arrays that map the state indices from spin systems A and B to
    the spin system C. This function is called from associate().

    Parameters
    ----------
    spin_system_A : SpinSystem
    spin_system_B : SpinSystem
    spin_system_C : SpinSystem
    spin_map_A : tuple
        Indices of spin system A in spin system C.
    spin_map_B : tuple
        Indices of spin system B in spin system C.

    Returns
    -------
    index_map_A : numpy.ndarray
    index_map_B : numpy.ndarray
    index_map_C : numpy.ndarray 
    """

    # Make empty lists for the index maps
    index_map_A = []
    index_map_B = []
    index_map_C = []

    # Loop over the destination states
    for state, idx_C in spin_system_C.basis.dict.items():

        # Get the corresponding states of A and B
        state_A = tuple(state[i] for i in spin_map_A)
        state_B = tuple(state[i] for i in spin_map_B)

        # Continue only if states A and B are found in basis A and B
        if (state_A in spin_system_A.basis.dict) and (state_B in spin_system_B.basis.dict):

            # Append the index maps
            index_map_A.append(spin_system_A.basis.dict[state_A])
            index_map_B.append(spin_system_B.basis.dict[state_B])
            index_map_C.append(idx_C)

    return index_map_A, index_map_B, index_map_C

def associate(spin_system_A: SpinSystem,
              spin_system_B: SpinSystem,
              spin_system_C: SpinSystem,
              rho_A: Union[np.ndarray, csc_array],
              rho_B: Union[np.ndarray, csc_array],
              spin_map_A: tuple,
              spin_map_B: tuple) -> Union[np.ndarray, csc_array]:
    """
    This function is used to merge two state vectors together when the two
    spin systems associate in a chemical reaction A + B -> C.

    Parameters
    ----------
    spin_system_A : SpinSystem
    spin_system_B : SpinSystem
    spin_system_C : SpinSystem
    rho_A : numpy.ndarray or csc_array
        State vector of spin system A.
    rho_B : numpy.ndarray or csc_array
        State vector of spin system B.
    spin_map_A : tuple
        Indices where the spins in spin system A go to in spin system C.
    spin_map_B : tuple
        Indices where the spins in spin system B go to in spin system C.

    Returns
    -------
    rho_C : numpy.ndarray or csc_array
        State vector of a composite system A + B.
    """

    # Get the index map
    idx_A, idx_B, idx_C = associate_index_map(spin_system_A, spin_system_B, spin_system_C, spin_map_A, spin_map_B)

    # Initialize an empty spin density vector
    if isinstance(rho_A, np.ndarray):
        rho_C = np.zeros((spin_system_C.basis.dim, 1), dtype=complex)
    else:
        rho_C = lil_array((spin_system_C.basis.dim, 1), dtype=complex)

    # Associate the two spin density vectors
    rho_C[idx_C, [0]] = rho_A[idx_A, [0]] * rho_B[idx_B, [0]]

    # Convert to csc_array if needed
    if not isinstance(rho_A, np.ndarray):
        rho_C = rho_C.tocsc()

    return rho_C

@lru_cache(maxsize=16)
def rotation_matrix(spin_system: SpinSystem, spin_map: tuple) -> csc_array:
    """
    Creates a rotation matrix that is used to rotate a density vector from
    a basis set to another.

    Parameters
    ----------
    spin_system : SpinSystem
    spin_map : tuple
        Indices of the spins in the spin system after rotation.

    Returns
    -------
    rot : csc_array
        Rotation matrix.
    """

    # Create an empty array for the rotated indices
    indices = np.empty(spin_system.basis.dim, dtype=int)

    # Go through the basis set
    for state, idx in spin_system.basis.dict.items():

        # Find out the rotated state
        state_rot = tuple(state[i] for i in spin_map)

        # Find the index of the rotated state
        idx_rot = state_idx(spin_system, state_rot)

        # Add to the list of indices
        indices[idx] = idx_rot

    # Initialize the rotation matrix
    rot = eye_array(spin_system.basis.dim, dtype=int, format='lil')

    # Re-order the rows and convert to CSC
    rot = rot[indices].tocsc()

    return rot

def rotate_molecule(spin_system:SpinSystem, rho:Union[np.ndarray, csc_array], spin_map:tuple) -> Union[np.ndarray, csc_array]:
    """
    Rotates the density vector of a molecule to correspond to basis oriented
    differently. Useful for making different conformers.

    Parameters
    ----------
    spin_system : SpinSystem
    rho : numpy.ndarray or csc_array
        Density vector of the spin system.
    spin_map : tuple
        Indices of the spins in the spin system after rotation.

    Returns
    -------
    rho : numpy.ndarray or csc_array
        Rotated density vector of the spin system.
    """

    # Get the rotation matrix
    rot = rotation_matrix(spin_system, spin_map)

    # Rotate the density vector
    rho = rot @ rho

    return rho