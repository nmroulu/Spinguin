"""
operators.py

This module provides functions for calculating quantum mechanical spin operators.
It includes functions for single-spin operators as well as many-spin superoperators.
"""

# For referencing the SpinSystem class
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from spinguin.system.spin_system import SpinSystem

# Imports
import numpy as np
from functools import lru_cache
from itertools import product
from spinguin.utils import la
from spinguin.system.basis import idx_to_lq, parse_operator_string
from scipy.sparse import csc_array, eye_array

def op_E(S: float) -> np.ndarray:
    """
    Generates the unit operator for a given spin quantum number `S`.

    Parameters
    ----------
    S : float
        Spin quantum number.

    Returns
    -------
    E : numpy.ndarray
        NumPy array representing the unit operator.
    """
    # Generate a unit operator of the correct dimension
    dim = int(2 * S + 1)
    E = np.eye(dim)

    return E

def op_Sx(S: float) -> np.ndarray:
    """
    Generates the spin operator Sx for a given spin quantum number `S`.

    Parameters
    ----------
    S : float
        Spin quantum number.

    Returns
    -------
    Sx : numpy.ndarray
        NumPy array representing the x-component spin operator.
    """
    # Calculate Sx using the raising and lowering operators
    Sx = 1 / 2 * (op_Sp(S) + op_Sm(S))

    return Sx

def op_Sy(S: float) -> np.ndarray:
    """
    Generates the spin operator Sy for a given spin quantum number `S`.

    Parameters
    ----------
    S : float
        Spin quantum number.

    Returns
    -------
    Sy : numpy.ndarray
        NumPy array representing the y-component spin operator.
    """
    # Calculate Sy using the raising and lowering operators
    Sy = 1 / (2j) * (op_Sp(S) - op_Sm(S))

    return Sy

def op_Sz(S: float) -> np.ndarray:
    """
    Generates the spin operator Sz for a given spin quantum number `S`.

    Parameters
    ----------
    S : float
        Spin quantum number.

    Returns
    -------
    Sz : numpy.ndarray
        NumPy array representing the z-component spin operator.
    """
    # Get the possible spin magnetic quantum numbers (from largest to smallest)
    m = -np.arange(-S, S + 1)

    # Initialize the operator
    Sz = np.zeros((len(m), len(m)), dtype=complex)

    # Populate the diagonal elements
    for i in range(len(m)):
        Sz[i, i] = m[i]  

    return Sz

def op_Sp(S: float) -> np.ndarray:
    """
    Generates the spin raising operator for a given spin quantum number `S`.

    Parameters
    ----------
    S : float
        Spin quantum number.

    Returns
    -------
    Sp : numpy.ndarray
        NumPy array representing the raising operator.
    """
    # Get the possible spin magnetic quantum numbers
    m = np.arange(-S, S + 1)

    # Initialize the operator
    Sp = np.zeros((len(m), len(m)), dtype=complex)

    # Populate the off-diagonal elements
    for i in range(len(m) - 1):
        Sp[i, i + 1] = np.sqrt(S * (S + 1) - m[i] * (m[i] + 1))

    return Sp

def op_Sm(S: float) -> np.ndarray:
    """
    Generates the spin lowering operator for a given spin quantum number `S`.

    Parameters
    ----------
    S : float
        Spin quantum number.

    Returns
    -------
    Sm : numpy.ndarray
        NumPy array representing the lowering operator.
    """
    # Get the possible spin magnetic quantum numbers
    m = np.arange(-S, S + 1)

    # Initialize the operator
    Sm = np.zeros((len(m), len(m)), dtype=complex)

    # Populate the off-diagonal elements
    for i in range(1, len(m)):
        Sm[i, i - 1] = np.sqrt(S * (S + 1) - m[i] * (m[i] - 1))

    return Sm

@lru_cache(maxsize=1024)
def op_T(S: float, l: int, q: int) -> np.ndarray:
    """
    Generates the numerical spherical tensor operator for a given spin quantum number `S`, 
    rank `l`, and projection `q`. The operator is obtained by sequential lowering of the 
    maximum projection operator.

    Source: Kuprov (2023) - Spin: From Basic Symmetries to Quantum Optimal Control, page 94.

    This function is called frequently and is cached for high performance.

    Parameters
    ----------
    S : float
        Spin quantum number.
    l : int
        Operator rank.
    q : int
        Operator projection.

    Returns
    -------
    T : numpy.ndarray
        NumPy array representing the spherical tensor operator.
    """
    # Calculate the operator with maximum projection q = l
    T = (-1)**l * 2**(-l / 2) * np.linalg.matrix_power(op_Sp(S), l)

    # Perform the necessary number of lowerings
    for i in range(l - q):
        # Get the current q
        q = l - i

        # Perform the lowering
        T = la.comm(op_Sm(S), T) / np.sqrt(l * (l + 1) - q * (q - 1))

    return T

def op_T_coupled(l: int, q: int, l1: int, s1: float, l2: int, s2: float) -> np.ndarray:
    """
    Computes the coupled irreducible spherical tensor of rank `l` and projection `q`
    from two irreducible spherical tensors of ranks `l1` and `l2`.

    Parameters
    ----------
    l : int
        Rank of the coupled operator.
    q : int
        Projection of the coupled operator.
    l1 : int
        Rank of the first operator to be coupled.
    s1 : float
        Spin quantum number of the first spin.
    l2 : int
        Rank of the second operator to be coupled.
    s2 : float
        Spin quantum number of the second spin.
    
    Returns
    -------
    T : numpy.ndarray
        Coupled spherical tensor operator of rank `l` and projection `q`.
    """

    # Initialize the operator
    dim = int((2 * s1 + 1) * (2 * s2 + 1))
    T = np.zeros((dim, dim), dtype=complex)

    # Iterate over the projections
    for q1 in range(-l1, l1 + 1):
        for q2 in range(-l2, l2 + 1):

            # Analogously to the coupling of angular momenta
            T += la.CG_coeff(l1, q1, l2, q2, l, q) * np.kron(op_T(s1, l1, q1), op_T(s2, l2, q2))

    return T

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

def sop_E(spin_system: SpinSystem) -> csc_array:
    """
    Returns the unit superoperator.

    Parameters
    ----------
    spin_system : SpinSystem
        The spin system for which the unit superoperator is generated.

    Returns
    -------
    unit : csc_array
        A sparse array corresponding to the unit operator.
    """

    # Create the unit operator
    unit = eye_array(spin_system.basis.dim, format='csc')

    return unit

def op_prod(op_def: tuple, spins: tuple, include_unit: bool=True) -> np.ndarray:
    """
    Generates a product operator defined by `op_def` in the Zeeman eigenbasis.

    Parameters
    ----------
    op_def : tuple
        Specifies the product operator to be generated. For example,
        input (0, 2, 0, 1) will generate E*T_10*E*T_11. The indices are
        given by N = l^2 + l - q, where l is the rank and q is the projection.
    spins : tuple
        Spin quantum numbers. Must match the length of `op_def`.
    include_unit : bool
        Specifies whether unit operators are included in the product operator.

    Returns
    -------
    op : numpy.ndarray
        Product operator in the Zeeman eigenbasis.
    """

    # Initialize the product operator
    op = 1

    # Iterate through the operator definition
    for spin, oper in zip(spins, op_def):

        # Exclude unit operators if requested
        if include_unit or oper != 0:

            # Get the rank and projection
            l, q = idx_to_lq(oper)

            # Add to the product operator
            op = np.kron(op, op_T(spin, l, q))

    return op

@lru_cache(maxsize=4096)
def sop_prod(spin_system: SpinSystem, op_def: tuple, side: str) -> csc_array:
    """
    Generates a product superoperator corresponding to the product operator
    defined by `op_def`.
    
    This function is called frequently and is cached for high performance.

    Parameters
    ----------
    spin_system : SpinSystem
        The spin system for which the superoperator is generated.
    op_def : tuple
        Specifies the product operator to be generated. For example,
        input (0, 2, 0, 1) will generate E*T_10*E*T_11. The indices are
        given by N = l^2 + l - q, where l is the rank and q is the projection.
    side : str
        Specifies the type of superoperator:
        - 'comm' -- commutation superoperator
        - 'left' -- left superoperator
        - 'right' -- right superoperator

    Returns
    -------
    sop : csc_array
        Superoperator defined by `op_def`.
    """

    # If commutation superoperator, calculate left and right superoperators and return their difference
    if side == 'comm':
        sop = sop_prod(spin_system, op_def, 'left') \
            - sop_prod(spin_system, op_def, 'right')
        return sop
    
    # Extract necessary information from the spin system
    basis = spin_system.basis.arr
    dim = spin_system.basis.dim
    spins = spin_system.spins

    # Find indices of the spins participating in the operator
    idx_spins = np.nonzero(op_def)[0]

    # Return the unit operator if no spins participate in the operator
    if len(idx_spins) == 0:
        sop = sop_E(spin_system)
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
    sop = csc_array((sop_vals, (sop_j, sop_k)), shape=(dim, dim))

    return sop

def operator(spin_system: SpinSystem, operator: str) -> np.ndarray:
    """
    Generates an operator for the `spin_system` in Hilbert space from the user-specified `operators` string.

    Parameters
    ----------
    spin_system : SpinSystem
        The spin system for which the operator is generated.
    operator : str
        Defines the operator to be generated. The operator string must follow the rules below:

        - Cartesian and ladder operators: I(component,index). Example: I(x,4) --> Creates x-operator for spin at index 4.
        - Spherical tensor operators: T(l,q,index). Example: T(1,-1,3) --> Creates operator with l=1, q=-1 for spin at index 3.
        - Product operators have `*` in between the single-spin operators: I(z,0) * I(z,1)
        - Sums of operators have `+` in between the operators: I(x,0) + I(x,1)
        - The unit operator is not typed. Example: I(z,1) will generate E*I_z in case of a two-spin system. 
        - Whitespace will be ignored in the input.

    Returns
    -------
    op : numpy.ndarray
        The requested operator.
    """

    # Get the dimension of the operator
    dim = np.prod(spin_system.mults)

    # Initialize the operator
    op = np.zeros((dim, dim), dtype=complex)

    # Get the operator definitions and coefficients
    op_defs, coeffs = parse_operator_string(spin_system, operator)

    # Construct the operator
    for op_def, coeff in zip(op_defs, coeffs):
        op = op + coeff * op_prod(op_def, spin_system.spins)

    return op

def superoperator(spin_system: SpinSystem, operator: str, side: str='comm') -> csc_array:
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

    Returns
    -------
    sop : csc_array
        The requested superoperator.
    """

    # Extract necessary information from the spin system
    dim = spin_system.basis.dim

    # Initialize the superoperator
    sop = csc_array((dim, dim), dtype=complex)

    # Get the operator definitions and coefficients
    op_defs, coeffs = parse_operator_string(spin_system, operator)

    # Add to the operator
    for op_def, coeff in zip(op_defs, coeffs):
        sop = sop + coeff * sop_prod(spin_system, op_def, side)

    return sop

@lru_cache(maxsize=4096)
def sop_T_coupled(spin_system: SpinSystem, l: int, q: int, spin_1: int, spin_2: int=None) -> csc_array:
    """
    Computes the product superoperator corresponding to the coupled spherical tensor
    operator of rank `l` and projection `q`, derived from two spherical tensor operators of rank 1.

    This function is frequently called and is cached for high performance.

    Parameters
    ----------
    spin_system : SpinSystem
        The spin system for which the superoperator is generated.
    l : int
        Rank of the coupled operator.
    q : int
        Projection of the coupled operator.
    spin_1 : int
        Index of the first spin.
    spin_2 : int, optional
        Index of the second spin. Leave empty for linear single-spin interactions (e.g., shielding).

    Returns
    -------
    sop : csc_array
        Coupled spherical tensor superoperator of rank `l` and projection `q`.
    """

    # Extract the necessary information from the spin system
    dim = spin_system.basis.dim
    nspins = spin_system.size
    
    # Initialize the operator
    sop = csc_array((dim, dim), dtype=complex)

    # Handle two-spin bilinear interactions
    if isinstance(spin_2, int):

        # Loop over the projections of the rank-1 spherical tensors
        for q1 in range(-1, 2):
            for q2 in range(-1, 2):

                # Get the product operator definition corresponding to the coupled operator
                op_def = np.zeros(nspins, dtype=int)
                op_def[spin_1] = 2 - q1
                op_def[spin_2] = 2 - q2
                op_def = tuple(op_def)

                # Use the coupling of angular momenta equation
                sop += la.CG_coeff(1, q1, 1, q2, l, q) * sop_prod(spin_system, op_def, 'comm')

    # Handle linear single-spin interactions
    else:

        # The only non-zero component for the second spherical tensor is (1, 0) = 1
        for q1 in range(-1, 2):

            # Get the product operator definition corresponding to the coupled operator
            op_def = np.zeros(nspins, dtype=int)
            op_def[spin_1] = 2 - q1
            op_def = tuple(op_def)

            # Use the coupling of angular momenta equation
            sop += la.CG_coeff(1, q1, 1, 1, l, q) * sop_prod(spin_system, op_def, 'comm')

    return sop