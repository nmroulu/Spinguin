"""
superoperators.py

This module provides functions for calculating Liouville-space superoperators
either in full or truncated basis set.
"""

# For referencing the SpinSystem/Basis classes
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from spinguin.system.spin_system import SpinSystem
    from spinguin.system.basis import Basis

# Imports
import numpy as np
import scipy.sparse as sp
from functools import lru_cache
from itertools import product
from spinguin.utils import la
from spinguin.system.basis import idx_to_lq, parse_operator_string
from spinguin.qm.operators import op_T

@lru_cache(maxsize=16)
def structure_coefficients(spin: float, side: str) -> np.ndarray:
    """
    Computes the (normalized) structure coefficients of the operator algebra
    for a single spin. These coefficients are used in constructing product superoperators.

    Logic explained in the following paper (Eq. 24, calculate f_ijk):
    (The paper does not include the normalization)
    https://doi.org/10.1063/1.3398146

    This function is called frequently and is cached for high performance.

    Parameters
    ----------
    spin : float
        Spin quantum number.
    side : str
        Specifies the side of the multiplication. Can be either 'left' or 'right'.
    
    Returns
    -------
    c : numpy.ndarray
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
            norm = np.sqrt((T_j.conj().T @ T_j).trace() * (T_k.conj().T @ T_k).trace())

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
                    raise ValueError("The 'side' parameter must be either 'left' or 'right'.")

    return c

def sop_E(dim: int, sparse: bool=True) -> np.ndarray | sp.csc_array:
    """
    Returns the unit superoperator.

    Parameters
    ----------
    dim : int
        Dimension of the basis set.
    sparse: bool, default=True
        Specifies whether to return the operator as sparse or dense array.

    Returns
    -------
    unit : ndarray or csc_array
        A sparse array corresponding to the unit operator.
    """

    # Create the unit operator
    unit = sp.eye_array(dim, format='csc')
    if not sparse:
        unit = unit.toarray()

    return unit

def sop_prod(op_def: np.ndarray, basis: Basis, spins: np.ndarray, side: str, sparse: bool=True) -> np.ndarray | sp.csc_array:
    """
    Generates a product superoperator corresponding to the product operator
    defined by `op_def`.
    
    This function is called frequently and is cached for high performance.

    Parameters
    ----------
    op_def : ndarray
        Specifies the product operator to be generated. For example,
        input `(0, 2, 0, 1)` will generate `E*T_10*E*T_11`. The indices are
        given by `N = l^2 + l - q`, where `l` is the rank and `q` is the projection.
    basis : Basis
        Basis set consisting of Kronecker products of single-spin irreducible
        spherical tensors, described by tuples of integers.
    spins : ndarray
        A sequence of floats describing the spin quantum numbers of the spin system.
    side : str
        Specifies the type of superoperator:
        - 'comm' -- commutation superoperator
        - 'left' -- left superoperator
        - 'right' -- right superoperator
    sparse: bool, default=True
        Specifies whether to return the operator as sparse or dense array.

    Returns
    -------
    sop : ndarray or csc_array
        Superoperator defined by `op_def`.
    """

    @lru_cache(maxsize=4096)
    def _sop_prod(op_def: tuple, basis: Basis, spins: tuple, side: str, sparse: bool=True) -> np.ndarray | sp.csc_array:

        # If commutation superoperator, calculate left and right superoperators and return their difference
        if side == 'comm':
            sop = _sop_prod(op_def, basis, spins, 'left', sparse) \
                - _sop_prod(op_def, basis, spins, 'right', sparse)
            return sop

        # Find indices of the spins participating in the operator
        idx_spins = np.nonzero(op_def)[0]

        # Return the unit operator if no spins participate in the operator
        if len(idx_spins) == 0:
            sop = sop_E(basis.dim, sparse)
            return sop

        # Initialize lists for storing non-zero structure coefficients and their indices
        c_jk = []
        j = []
        k = []

        # Loop over the relevant spins
        for n in idx_spins:

            # Get the structure coefficients for the current spin
            c_jk_n = structure_coefficients(spins[n], side)[op_def[n], :, :]

            # Obtain the indices of the non-zero values
            nonzero_jk = np.nonzero(c_jk_n)

            # Append to the arrays
            c_jk.append(c_jk_n[nonzero_jk])
            j.append(nonzero_jk[0])
            k.append(nonzero_jk[1])

        # Calculate the products of structure coefficients and their corresponding operator definitions
        prod_c_jk = np.array([np.prod(c_jk_n) for c_jk_n in product(*c_jk)])
        op_defs_j = np.array(list(product(*j)))
        op_defs_k = np.array(list(product(*k)))

        # Initialize lists for the superoperator values and indices
        sop_vals = []
        sop_j = []
        sop_k = []

        # Iterate through each combination
        for m in range(prod_c_jk.shape[0]):

            # Get the indices of the basis set operator definitions that contain the current operator definitions
            j_op = np.where(np.all(basis[:, idx_spins] == op_defs_j[m], axis=1))[0]
            k_op = np.where(np.all(basis[:, idx_spins] == op_defs_k[m], axis=1))[0]

            # Continue only if the basis contains such operator definitions
            if j_op.shape[0] != 0 or k_op.shape[0] != 0:

                # Obtain the full operator definitions from the basis
                op_def_j = basis[j_op, :]
                op_def_k = basis[k_op, :]

                # Leave only the operator definition for non-participating spins
                op_def_j = np.delete(op_def_j, idx_spins, axis=1)
                op_def_k = np.delete(op_def_k, idx_spins, axis=1)
                
                # Operator definitions must match for the product of structure coefficients to be nonzero
                ind_j, ind_k = la.find_common_rows(op_def_j, op_def_k)

                # Append the products of structure coefficients and the indices to the lists
                sop_vals.append(prod_c_jk[m] * np.ones(len(ind_j)))
                sop_j.append(j_op[ind_j])
                sop_k.append(k_op[ind_k])

        # Concatenate the arrays
        sop_j = np.concatenate(sop_j, dtype=int)
        sop_k = np.concatenate(sop_k, dtype=int)
        sop_vals = np.concatenate(sop_vals, dtype=complex)

        # Construct the superoperator
        sop = sp.csc_array((sop_vals, (sop_j, sop_k)), shape=(basis.dim, basis.dim))
        if not sparse:
            sop = sop.toarray()

        return sop
    
    # Ensure that input type is tuple for hashing
    op_def = tuple(op_def)
    spins = tuple(spins)
    
    # Ensure a different instance is returned
    sop = _sop_prod(op_def, basis, spins, side, sparse).copy()

    return sop

def sop_prod_ref(op_def: np.ndarray, basis: Basis, spins: np.ndarray, side: str) -> np.ndarray:
    """
    A reference method for calculating the superoperator.
    
    NOTE:
    This implementation is very slow and should be used for testing purposes only.

    Parameters
    ----------
    op_def : ndarray
        Specifies the product operator to be generated. For example,
        input `(0, 2, 0, 1)` will generate `E*T_10*E*T_11`. The indices are
        given by `N = l^2 + l - q`, where `l` is the rank and `q` is the projection.
    basis : Basis
        Basis set consisting of Kronecker products of single-spin irreducible
        spherical tensors, described by tuples of integers.
    spins : ndarray
        A sequence of floats describing the spin quantum numbers of the spin system.
    side : str
        Specifies the type of superoperator:
        - 'comm' -- commutation superoperator
        - 'left' -- left superoperator
        - 'right' -- right superoperator
    """

    # Convert input to NumPy
    op_def = np.asarray(op_def)
    spins = np.asarray(spins)

    # If commutation superoperator, calculate left and right superoperators and return their difference
    if side == 'comm':
        sop = sop_prod_ref(op_def, basis, spins, 'left') \
            - sop_prod_ref(op_def, basis, spins, 'right')
        return sop
    
    # Initialize the superoperator
    sop = np.zeros((basis.dim, basis.dim), dtype=complex)

    # Loop over each matrix row j
    for j in range(basis.dim):

        # Loop over each matrix column k
        for k in range(basis.dim):

            # Initialize the matrix element
            sop_jk = 1

            # Loop over the spins
            for n in range(basis.nspins):

                # Get the single-spin operator indices
                i_ind = op_def[n]
                j_ind = basis[j, n]
                k_ind = basis[k, n]

                # Get the structure coefficients for the current spin
                c = structure_coefficients(spins[n], side)

                # Add to the product
                sop_jk = sop_jk * c[i_ind, j_ind, k_ind]

            # Add to the superoperator
            sop[j, k] = sop_jk

    return sop

def superoperator(spin_system: SpinSystem, operator: str, side: str='comm', sparse: bool=True) -> np.ndarray | sp.csc_array:
    """
    Generates a superoperator from the user-specified `operators` string.

    Parameters
    ----------
    spin_system : SpinSystem
        The spin system for which the superoperator is generated.
    operator : str
        Defines the operator to be generated. The operator string must follow the rules below:

        - Cartesian and ladder operators: I(component,index). Example: I(x,4) --> Creates x-operator for spin at index 4.
        - Spherical tensor operators: T(l,q,index). Example: T(1,-1,3) --> Creates operator with l=1, q=-1 for spin at index 3.
        - Product operators have `*` in between the single-spin operators: I(z,0) * I(z,1)
        - Sums of operators have `+` in between the operators: I(x,0) + I(x,1)
        - The unit operator is not typed. Example: I(z,2) will generate E*I_z in case of a two-spin system. 
        - Whitespace will be ignored in the input.
    side : str
        Specifies the type of superoperator:
        - 'comm' -- commutation superoperator (default)
        - 'left' -- left superoperator
        - 'right' -- right superoperator
    sparse: bool, default=True
        Specifies whether to return the operator as sparse or dense array.

    Returns
    -------
    sop : ndarray or csc_array
        The requested superoperator.
    """

    # Extract necessary information from the spin system
    dim = spin_system.basis.dim

    # Initialize the superoperator
    sop = sp.csc_array((dim, dim), dtype=complex)
    if not sparse:
        sop = sop.toarray()

    # Get the operator definitions and coefficients
    op_defs, coeffs = parse_operator_string(operator, spin_system.nspins)

    # Add to the operator
    for op_def, coeff in zip(op_defs, coeffs):
        sop = sop + coeff * sop_prod(op_def, spin_system.basis, spin_system.spins, side, sparse)

    return sop

def sop_T_coupled(basis: Basis, spins: np.ndarray, l: int, q: int, spin_1: int, spin_2: int=None, sparse: bool=True) -> np.ndarray | sp.csc_array:
    """
    Computes the product superoperator corresponding to the coupled spherical tensor
    operator of rank `l` and projection `q`, derived from two spherical tensor operators of rank 1.

    This function is frequently called and is cached for high performance.

    Parameters
    ----------
    basis : Basis
        Basis set consisting of Kronecker products of single-spin irreducible
        spherical tensors, described by tuples of integers.
    spins : ndarray
        A sequence of floats describing the spin quantum numbers of the spin system.
    l : int
        Rank of the coupled operator.
    q : int
        Projection of the coupled operator.
    spin_1 : int
        Index of the first spin.
    spin_2 : int, optional
        Index of the second spin. Leave empty for linear single-spin interactions (e.g., shielding).
    sparse: bool, default=True
        Specifies whether to return the operator as sparse or dense array.

    Returns
    -------
    sop : ndarray or csc_array
        Coupled spherical tensor superoperator of rank `l` and projection `q`.
    """

    @lru_cache(maxsize=4096)
    def _sop_T_coupled(basis: Basis, spins: tuple, l: int, q: int, spin_1: int, spin_2: int=None, sparse: bool=True) -> np.ndarray | sp.csc_array:
        
        # Initialize the operator
        sop = sp.csc_array((basis.dim, basis.dim), dtype=complex)
        if not sparse:
            sop = sop.toarray()

        # Handle two-spin bilinear interactions
        if isinstance(spin_2, int):

            # Loop over the projections of the rank-1 spherical tensors
            for q1 in range(-1, 2):
                for q2 in range(-1, 2):

                    # Get the product operator definition corresponding to the coupled operator
                    op_def = np.zeros(basis.nspins, dtype=int)
                    op_def[spin_1] = 2 - q1
                    op_def[spin_2] = 2 - q2
                    op_def = tuple(op_def)

                    # Use the coupling of angular momenta equation
                    sop += la.CG_coeff(1, q1, 1, q2, l, q) * sop_prod(op_def, basis, spins, 'comm', sparse)

        # Handle linear single-spin interactions
        else:

            # The only non-zero component for the second spherical tensor is (1, 0) = 1
            for q1 in range(-1, 2):

                # Get the product operator definition corresponding to the coupled operator
                op_def = np.zeros(nspins, dtype=int)
                op_def[spin_1] = 2 - q1
                op_def = tuple(op_def)

                # Use the coupling of angular momenta equation
                sop += la.CG_coeff(1, q1, 1, 1, l, q) * sop_prod(op_def, basis, spins, 'comm', sparse)

        return sop
    
    # Convert to tuple to make hashing possible
    spins = tuple(spins)
    
    # Ensure a different instance is returned
    sop = _sop_T_coupled(basis, spins, l, q, spin_1, spin_2, sparse).copy()

    return sop