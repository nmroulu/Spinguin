"""
This module provides functions for calculating Liouville-space superoperators
either in full or truncated basis set.
"""
# Type checking
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from spinguin._core._spin_system import SpinSystem

# Imports
import numpy as np
import scipy.sparse as sp
from functools import lru_cache
from itertools import product
from typing import Literal
from spinguin._core._basis_indexing import idx_to_lq, parse_operator_string
from spinguin._core._config import config
from spinguin._core._la import CG_coeff, find_common_rows
from spinguin._core._operators import op_T

@lru_cache(maxsize=16)
def _structure_coefficients(spin: float,
                            side: Literal["left", "right"]) -> np.ndarray:
    """
    Computes the (normalized) structure coefficients of the operator algebra
    for a single spin. These coefficients are used in constructing product
    superoperators.

    Logic explained in the following paper (Eq. 24, calculate f_ijk):
    (The paper does not include the normalization)
    https://doi.org/10.1063/1.3398146

    This function is called frequently and is cached for high performance.

    Parameters
    ----------
    spin : float
        Spin quantum number.
    side : {'left', 'right'}
        Specifies the side of the multiplication.
    
    Returns
    -------
    c : ndarray
        A 3-dimensional array containing all the structure coefficients.
    """

    # Get the spin multiplicity
    mult = int(2 * spin + 1)

    # Initialize the structure coefficient array
    c = np.zeros((mult**2, mult**2, mult**2), dtype=complex)

    # Iterate over the index j
    for j in range(mult**2):

        # Get the spherical tensor for j
        l_j, q_j = idx_to_lq(j)
        T_j = op_T(spin, l_j, q_j)
    
        # Iterate over the index k
        for k in range(mult**2):

            # Get the spherical tensor for k
            l_k, q_k = idx_to_lq(k)
            T_k = op_T(spin, l_k, q_k)

            # Apply normalization
            norm = np.sqrt(
                (T_j.conj().T @ T_j).trace() * (T_k.conj().T @ T_k).trace())

            # Iterate over the index i
            for i in range(mult**2):

                # Get the spherical tensor for i
                l_i, q_i = idx_to_lq(i)
                T_i = op_T(spin, l_i, q_i)

                # Compute the structure coefficient
                if side == 'left':
                    c[i, j, k] = (T_j.conj().T @ T_i @ T_k).trace() / norm
                elif side == 'right':
                    c[i, j, k] = (T_j.conj().T @ T_k @ T_i).trace() / norm
                else:
                    raise ValueError("The 'side' parameter must be either "
                                     "'left' or 'right'.")

    return c

def unit_superoperator(dim: int) -> np.ndarray | sp.csc_array:
    """
    Returns the unit superoperator.

    Parameters
    ----------
    dim : int
        Dimension of the basis set.

    Returns
    -------
    unit : ndarray or csc_array
        A sparse array corresponding to the unit operator.
    """

    # Create the unit operator
    if config.sparse_superoperator:
        unit = sp.eye_array(dim, format='csc')
    else:
        unit = np.eye(dim)

    return unit

@lru_cache(maxsize=4096)
def _superoperator_from_op_def(
    op_def_bytes: bytes,
    basis_bytes: bytes,
    spins_bytes: bytes,
    side: Literal["comm", "left", "right"],
    sparse: bool=True
) -> np.ndarray | sp.csc_array:

    # If commutation superoperator, calculate left and right superoperators and
    # return their difference
    if side == 'comm':
        left = _superoperator_from_op_def(op_def_bytes, basis_bytes,
                                          spins_bytes, 'left', sparse)
        right = _superoperator_from_op_def(op_def_bytes, basis_bytes,
                                          spins_bytes, 'right', sparse)
        sop = left - right
        return sop
    
    # Obtain the hashed elements
    op_def = np.frombuffer(op_def_bytes, dtype=int)
    basis = np.frombuffer(basis_bytes, dtype=int).reshape(-1, op_def.shape[0])
    spins = np.frombuffer(spins_bytes, dtype=float)

    # Obtain the basis dimension
    dim = basis.shape[0]

    # Find indices of the spins participating in the operator
    idx_spins = np.nonzero(op_def)[0]

    # Obtain the basis with only participating spins
    sub_basis = basis[:, idx_spins]

    # Return the unit operator if no spins participate in the operator
    if len(idx_spins) == 0:
        sop = unit_superoperator(dim)
        return sop

    # Initialize lists for storing non-zero structure coefficients and their
    # indices
    c_jk = []
    j = []
    k = []

    # Loop over the relevant spins
    for n in idx_spins:

        # Get the structure coefficients for the current spin
        c_jk_n = _structure_coefficients(spins[n], side)[op_def[n], :, :]

        # Obtain the indices of the non-zero values
        nonzero_jk = np.nonzero(c_jk_n)

        # Append to the arrays
        c_jk.append(c_jk_n[nonzero_jk])
        j.append(nonzero_jk[0])
        k.append(nonzero_jk[1])

    # Calculate the products of structure coefficients and their corresponding
    # operator definitions
    prod_c_jk = np.array([np.prod(c_jk_n) for c_jk_n in product(*c_jk)])
    op_defs_j = np.array(list(product(*j)))
    op_defs_k = np.array(list(product(*k)))

    # Initialize lists for the superoperator values and indices
    sop_vals = []
    sop_j = []
    sop_k = []

    # Iterate through each combination
    for m in range(prod_c_jk.shape[0]):

        # Get the indices of the basis set operator definitions that contain the
        # current operator definitions
        j_op = np.where(np.all(sub_basis == op_defs_j[m], axis=1))[0]
        k_op = np.where(np.all(sub_basis == op_defs_k[m], axis=1))[0]

        # Continue only if the basis contains such operator definitions
        if j_op.shape[0] != 0 and k_op.shape[0] != 0:

            # Obtain the full operator definitions from the basis
            op_def_j = basis[j_op, :]
            op_def_k = basis[k_op, :]

            # Leave only the operator definition for non-participating spins
            op_def_j = np.delete(op_def_j, idx_spins, axis=1)
            op_def_k = np.delete(op_def_k, idx_spins, axis=1)
            
            # Operator definitions must match for the product of structure
            # coefficients to be nonzero
            ind_j, ind_k = find_common_rows(op_def_j, op_def_k)

            # Append the products of structure coefficients and the indices to
            # the lists
            sop_vals.append(prod_c_jk[m] * np.ones(len(ind_j)))
            sop_j.append(j_op[ind_j])
            sop_k.append(k_op[ind_k])

    # Concatenate the arrays (sufficient to use 32-bit integers)
    sop_j = np.concatenate(sop_j, dtype=np.int32)
    sop_k = np.concatenate(sop_k, dtype=np.int32)
    sop_vals = np.concatenate(sop_vals, dtype=complex)

    # Construct the superoperator
    sop = sp.csc_array((sop_vals, (sop_j, sop_k)), shape=(dim, dim))
    if not sparse:
        sop = sop.toarray()

    return sop

def superoperator_from_op_def(
    spin_system: SpinSystem,
    op_def: np.ndarray,
    side: Literal["comm", "left", "right"],
) -> np.ndarray | sp.csc_array:
    """
    Generates a product superoperator corresponding to the product operator
    defined by `op_def`.
    
    This function is called frequently and is cached for high performance.

    Parameters
    ----------
    spin_system : SpinSystem
        Spin system for which the superoperator is going to be created.
    op_def : ndarray
        Specifies the product operator to be generated. For example,
        input `np.array([0, 2, 0, 1])` will generate `E*T_10*E*T_11`. The
        indices are given by `N = l^2 + l - q`, where `l` is the rank and `q` is
        the projection.
    side : {'comm', 'left', 'right'}
        Specifies the type of superoperator:
        - 'comm' -- commutation superoperator
        - 'left' -- left superoperator
        - 'right' -- right superoperator

    Returns
    -------
    sop : ndarray or csc_array
        Superoperator defined by `op_def`.
    """
    # Convert types suitable for hashing
    op_def_bytes = op_def.tobytes()
    basis_bytes = spin_system.basis.basis.tobytes()
    spins_bytes = spin_system.spins.tobytes()
    
    # Create the superoperator and ensure that a different instance is returned
    sop = _superoperator_from_op_def(op_def_bytes,
                                     basis_bytes,
                                     spins_bytes,
                                     side,
                                     config.sparse_superoperator).copy()

    return sop

def sop_prod_ref(op_def: np.ndarray,
                 basis: np.ndarray,
                 spins: np.ndarray,
                 side: Literal["comm", "left", "right"]) -> np.ndarray:
    """
    A reference method for calculating the superoperator.
    
    NOTE:
    This implementation is very slow and should be used for testing purposes
    only.

    Parameters
    ----------
    op_def : ndarray
        Specifies the product operator to be generated. For example,
        input `(0, 2, 0, 1)` will generate `E*T_10*E*T_11`. The indices are
        given by `N = l^2 + l - q`, where `l` is the rank and `q` is the
        projection.
    basis : ndarray
        A two-dimensional array where each row contains integers that represent
        a Kronecker product of single-spin irreducible spherical tensors.
    spins : ndarray
        A sequence of floats describing the spin quantum numbers of the spin
        system.
    side : {'comm', 'left', 'right'}
        Specifies the type of superoperator:
        - 'comm' -- commutation superoperator
        - 'left' -- left superoperator
        - 'right' -- right superoperator

    Returns
    -------
    sop : ndarray
        Superoperator defined by `op_def`.
    """

    # If commutation superoperator, calculate left and right superoperators and
    # return their difference
    if side == 'comm':
        sop = sop_prod_ref(op_def, basis, spins, 'left') \
            - sop_prod_ref(op_def, basis, spins, 'right')
        return sop
    
    # Obtain the basis dimension and number of spins
    dim = basis.shape[0]
    nspins = spins.shape[0]
    
    # Initialize the superoperator
    sop = np.zeros((dim, dim), dtype=complex)

    # Loop over each matrix row j
    for j in range(dim):

        # Loop over each matrix column k
        for k in range(dim):

            # Initialize the matrix element
            sop_jk = 1

            # Loop over the spins
            for n in range(nspins):

                # Get the single-spin operator indices
                i_ind = op_def[n]
                j_ind = basis[j, n]
                k_ind = basis[k, n]

                # Get the structure coefficients for the current spin
                c = _structure_coefficients(spins[n], side)

                # Add to the product
                sop_jk = sop_jk * c[i_ind, j_ind, k_ind]

            # Add to the superoperator
            sop[j, k] = sop_jk

    return sop

def superoperator(
    spin_system: SpinSystem,
    operator: str,
    side: Literal["comm", "left", "right"] = "comm"
) -> np.ndarray | sp.csc_array:
    """
    Generates a Liouville-space superoperator for the `spin_system` from the
    user-specified `operator` string.

    Parameters
    ----------
    spin_system : SpinSystem
        Spin system for which the superoperator is going to be generated.
    operator : str
        Defines the superoperator to be generated. The operator string must
        follow the rules below:

        - Cartesian and ladder operators: `I(component,index)` or
          `I(component)`. Examples:

            - `I(x,4)` --> Creates x-operator for spin at index 4.
            - `I(x)`--> Creates x-operator for all spins.

        - Spherical tensor operators: `T(l,q,index)` or `T(l,q)`. Examples:

            - `T(1,-1,3)` --> \
              Creates operator with `l=1`, `q=-1` for spin at index 3.
            - `T(1, -1)` --> \
              Creates operator with `l=1`, `q=-1` for all spins.
            
        - Product operators have `*` in between the single-spin operators:
          `I(z,0) * I(z,1)`
        - Sums of operators have `+` in between the operators:
          `I(x,0) + I(x,1)`
        - Unit operators are ignored in the input. Interpretation of these
          two is identical: `E * I(z,1)`, `I(z,1)`
        
        Special case: An empty `operator` string is considered as unit
        operator.

        Whitespace will be ignored in the input.

        NOTE: Indexing starts from 0!
    side : {'comm', 'left', 'right'}
        The type of superoperator:
        - 'comm' -- commutation superoperator (default)
        - 'left' -- left superoperator
        - 'right' -- right superoperator

    Returns
    -------
    sop : ndarray or csc_array
        An array representing the requested superoperator.
    """
    # Check that the basis has been built
    if spin_system.basis.basis is None:
        raise ValueError("Please build the basis before constructing "
                         "superoperators.")

    # Obtain basis dimension and number of spins
    dim = spin_system.basis.dim
    nspins = spin_system.nspins

    # Initialize the superoperator
    if config.sparse_superoperator:
        sop = sp.csc_array((dim, dim), dtype=complex)
    else:
        sop = np.zeros((dim, dim), dtype=complex)

    # Get the operator definitions and coefficients
    op_defs, coeffs = parse_operator_string(operator, nspins)

    # Build the operator
    for op_def, coeff in zip(op_defs, coeffs):
        sop = sop + coeff * superoperator_from_op_def(spin_system, op_def, side)

    return sop

def superoperator_T_coupled(
    spin_system: SpinSystem,
    l: int,
    q: int,
    spin_1: int,
    spin_2: int=None
) -> np.ndarray | sp.csc_array:
    """
    Computes the product superoperator corresponding to the coupled spherical
    tensor operator of rank `l` and projection `q`, derived from two spherical
    tensor operators of rank 1.

    Parameters
    ----------
    spin_system : SpinSystem
        Spin system for which the coupled superoperator is built.
    l : int
        Rank of the coupled operator.
    q : int
        Projection of the coupled operator.
    spin_1 : int
        Index of the first spin.
    spin_2 : int, default=None
        Index of the second spin. Leave empty for linear single-spin
        interactions (e.g., shielding).

    Returns
    -------
    sop : ndarray or csc_array
        Coupled spherical tensor superoperator of rank `l` and projection `q`.
    """
    # Obtain the basis dimension and number of spins
    dim = spin_system.basis.dim
    nspins = spin_system.nspins
    
    # Initialize the superoperator
    if config.sparse_superoperator:
        sop = sp.csc_array((dim, dim), dtype=complex)
    else:
        sop = np.zeros((dim, dim), dtype=complex)

    # Handle two-spin bilinear interactions
    if isinstance(spin_2, int):

        # Loop over the projections of the rank-1 spherical tensors
        for q1 in range(-1, 2):
            for q2 in range(-1, 2):

                # Get the product operator definition corresponding to the
                # coupled operator
                op_def = np.zeros(nspins, dtype=int)
                op_def[spin_1] = 2 - q1
                op_def[spin_2] = 2 - q2

                # Use the coupling of angular momenta equation
                sop += CG_coeff(1, q1, 1, q2, l, q) * \
                       superoperator_from_op_def(spin_system, op_def, "comm")

    # Handle linear single-spin interactions
    else:

        # Only non-zero component for the second spherical tensor is (1, 0) = 1
        for q1 in range(-1, 2):

            # Get the product operator definition corresponding to the coupled
            # operator
            op_def = np.zeros(nspins, dtype=int)
            op_def[spin_1] = 2 - q1

            # Use the coupling of angular momenta equation
            sop += CG_coeff(1, q1, 1, 1, l, q) * \
                   superoperator_from_op_def(spin_system, op_def, "comm")

    return sop