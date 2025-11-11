"""
This module provides functions for calculating Liouville-space superoperators
either in full or truncated basis set.
"""
# Referencing SpinSystem class
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from spinguin._core._spin_system import SpinSystem

# Imports
import numpy as np
import scipy.sparse as sp
import time
from functools import lru_cache
from itertools import product
from typing import Literal
from spinguin._core import _la
from spinguin._core._utils import idx_to_lq, parse_operator_string
from spinguin._core._operators import op_T
from spinguin._core._parameters import parameters

@lru_cache(maxsize=16)
def structure_coefficients(
    spin: float,
    side: Literal["left", "right"]
) -> np.ndarray:
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
                (T_j.conj().T @ T_j).trace() * (T_k.conj().T @ T_k).trace()
            )

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
                    raise ValueError(
                        "The 'side' parameter must be either 'left' or 'right'."
                    )

    return c

def sop_E(dim: int) -> np.ndarray | sp.csc_array:
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
    if parameters.sparse_superoperator:
        unit = sp.eye_array(dim, format='csc')
    else:
        unit = np.eye(dim)

    return unit

@lru_cache(maxsize=4096)
def _sop_prod(
    op_def_bytes: bytes,
    basis_bytes: bytes,
    spins_bytes: bytes,
    side: Literal["comm", "left", "right"],
    sparse: bool=True
) -> np.ndarray | sp.csc_array:

    # If commutation superoperator, calculate left and right superoperators and
    # return their difference
    if side == 'comm':
        sop = \
            _sop_prod(op_def_bytes, basis_bytes, spins_bytes, 'left', sparse) \
            - _sop_prod(op_def_bytes, basis_bytes, spins_bytes, 'right', sparse)
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
        sop = sop_E(dim)
        return sop

    # Initialize lists for storing non-zero structure coefficients and their
    # indices
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
            ind_j, ind_k = _la.find_common_rows(op_def_j, op_def_k)

            # Append the products of structure coefficients and the indices to
            # the lists
            sop_vals.append(prod_c_jk[m] * np.ones(len(ind_j)))
            sop_j.append(j_op[ind_j])
            sop_k.append(k_op[ind_k])

    # Concatenate the arrays
    # NOTE: Sufficient to use 32-bit integers
    sop_j = np.concatenate(sop_j, dtype=np.int32)
    sop_k = np.concatenate(sop_k, dtype=np.int32)
    sop_vals = np.concatenate(sop_vals, dtype=complex)

    # Construct the superoperator
    sop = sp.csc_array((sop_vals, (sop_j, sop_k)), shape=(dim, dim))
    if not sparse:
        sop = sop.toarray()

    return sop

def sop_prod(
    op_def: np.ndarray,
    basis: np.ndarray,
    spins: np.ndarray,
    side: Literal["comm", "left", "right"]
) -> np.ndarray | sp.csc_array:
    """
    Generates a product superoperator corresponding to the product operator
    defined by `op_def`.
    
    This function is called frequently and is cached for high performance.

    TODO: Vastaava cache pohdinta kuin sop_T_coupled() -funktion kanssa.

    Parameters
    ----------
    op_def : ndarray
        Specifies the product operator to be generated. For example,
        input `np.array([0, 2, 0, 1])` will generate `E*T_10*E*T_11`. The
        indices are given by `N = l^2 + l - q`, where `l` is the rank and `q` is
        the projection.
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
    sop : ndarray or csc_array
        Superoperator defined by `op_def`.
    """
    
    # Convert types suitable for hashing
    op_def_bytes = op_def.tobytes()
    basis_bytes = basis.tobytes()
    spins_bytes = spins.tobytes()
    
    # Create the superoperator and ensure a different instance is returned
    sop = _sop_prod(
        op_def_bytes,
        basis_bytes,
        spins_bytes,
        side,
        parameters.sparse_superoperator
    ).copy()

    return sop

def sop_from_string(
    operator: str,
    basis: np.ndarray,
    spins: np.ndarray,
    side: Literal["comm", "left", "right"]
) -> np.ndarray | sp.csc_array:
    """
    Generates a superoperator from the user-specified `operators` string.

    Parameters
    ----------
    operator : str
        Defines the operator to be generated. The operator string must
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
        
        Special case: An empty `operator` string is considered as unit operator.

        Whitespace will be ignored in the input.

        NOTE: Indexing starts from 0!
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
    sop : ndarray or csc_array
        The requested superoperator.
    """

    # Obtain basis dimension and number of spins
    dim = basis.shape[0]
    nspins = spins.shape[0]

    # Initialize the superoperator
    if parameters.sparse_superoperator:
        sop = sp.csc_array((dim, dim), dtype=complex)
    else:
        sop = np.zeros((dim, dim), dtype=complex)

    # Get the operator definitions and coefficients
    op_defs, coeffs = parse_operator_string(operator, nspins)

    # Add to the operator
    for op_def, coeff in zip(op_defs, coeffs):
        sop = sop + coeff * sop_prod(op_def, basis, spins, side)

    return sop

@lru_cache(maxsize=4096)
def _sop_T_coupled(
    basis_bytes: bytes,
    spins_bytes: bytes,
    l: int,
    q: int,
    spin_1: int,
    spin_2: int=None,
    sparse: bool=True
) -> np.ndarray | sp.csc_array:

    # Obtain the hashed elements
    spins = np.frombuffer(spins_bytes, dtype=float)
    basis = np.frombuffer(basis_bytes, dtype=int).reshape(-1, spins.shape[0])

    # Obtain the basis dimension and number of spins
    dim = basis.shape[0]
    nspins = spins.shape[0]
    
    # Initialize the operator
    if sparse:
        sop = sp.csc_array((dim, dim), dtype=complex)
    if not sparse:
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
                sop += _la.CG_coeff(1, q1, 1, q2, l, q) * \
                       sop_prod(op_def, basis, spins, 'comm')

    # Handle linear single-spin interactions
    else:

        # Only non-zero component for the second spherical tensor is (1, 0) = 1
        for q1 in range(-1, 2):

            # Get the product operator definition corresponding to the coupled
            # operator
            op_def = np.zeros(nspins, dtype=int)
            op_def[spin_1] = 2 - q1

            # Use the coupling of angular momenta equation
            sop += _la.CG_coeff(1, q1, 1, 1, l, q) * \
                   sop_prod(op_def, basis, spins, 'comm')

    return sop

def sop_T_coupled(
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

    This function is frequently called and is cached for high performance.

    TODO: Mieti, onko cache tarpeellinen. Nyt hyöty tulee vain, kun lasketaan R
    useaan kertaan samalle systeemille (esim. eri magneettikentissä). Voisi
    mahdollisesti olla asetus ohjelmassa (cachet päälle / pois).

    Parameters
    ----------
    spin_system : SpinSystem
        The spin system for which the coupled T superoperator is going to be
        generated.
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
    
    # Convert to bytes to make hashing possible
    basis_bytes = spin_system.basis.basis.tobytes()
    spins_bytes = spin_system.spins.tobytes()
    
    # Ensure a different instance is returned
    sop = _sop_T_coupled(
        basis_bytes, spins_bytes, l, q, spin_1, spin_2
    ).copy()

    return sop

def sop_to_truncated_basis(
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

def superoperator(
    spin_system: SpinSystem,
    operator: str | list | np.ndarray | tuple,
    side: Literal["comm", "left", "right"] = "comm"
) -> np.ndarray | sp.csc_array:
    """
    Generates a Liouville-space superoperator for the `spin_system` from the
    user-specified `operator` string.

    Parameters
    ----------
    spin_system : SpinSystem
        Spin system for which the superoperator is going to be generated.
    operator : str or list or ndarray or tuple
        Defines the superoperator to be generated.
        
        **The operator can be defined with two different approaches:**

        - A string. Example: `I(z,0) * I(z,1)`
        - An array of integers. Example: `[2, 2]`
        
        **The string input must follow the rules below:**

        - Cartesian and ladder operators: `I(component,index)` or 
          `I(component)`, with the indexing starting from zero. Examples:

            - `I(x,4)` --> Creates *x*-operator for spin at index 4.
            - `I(x)`--> Creates *x*-operator for all spins.

        - Spherical tensor operators: `T(l,q,index)` or `T(l,q)`. Examples:

            - `T(1,-1,3)` --> \
              Creates operator with *l*=1, *q*=-1 for spin at index 3.
            - `T(1, -1)` --> \
              Creates operator with *l*=1, *q*=-1 for all spins.
            
        - Product operators have `*` in between the single-spin operators:
          `I(z,0) * I(z,1)`
        - Sums of operators have `+` in between the operators:
          `I(x,0) + I(x,1)`
        - Unit operators are ignored in the input. Interpretation of these
          two is identical: `E * I(z,1)` and `I(z,1)`
        
        Special case: An empty `operator` string is considered as unit
        operator.

        Whitespace will be ignored in the input.

        **The array input is defined as follows:**

        - Each spin is given an integer *N* in the array.
        - Each integer corresponds to a spherical tensor operator of rank *l*
          and projection *q*: *N* = *l*^2 + *l* - *q*

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
        
    # Construct the superoperator using the string input
    if isinstance(operator, str):
        sop = sop_from_string(
            operator = operator,
            basis = spin_system.basis.basis,
            spins = spin_system.spins,
            side = side,
        )

    # Construct the superoperator using the array input
    elif isinstance(operator, (list, np.ndarray, tuple)):

        # Convert the input to NumPy array
        operator = np.asarray(operator)

        # Check that the dimensions match
        if not operator.ndim == 1:
            raise ValueError("The input array must be one-dimensional")
        if not operator.shape[0] == spin_system.nspins:
            raise ValueError(
                "Length of the operator array must match the number of spins."
            )
        
        # Construct the superoperator
        sop = sop_prod(
            op_def = operator,
            basis = spin_system.basis.basis,
            spins = spin_system.spins,
            side = side
        )

    # Otherwise raise an error
    else:
        raise ValueError("Invalid input type for 'operator'")
        
    return sop

def clear_cache_structure_coefficients():
    """
    Clears the cache of the `structure_coefficients()` function.
    """
    # Clear the cache
    structure_coefficients.cache_clear()

def clear_cache_sop_prod():
    """
    Clears the cache of the `_sop_prod()` function.
    """
    # Clear the cache
    _sop_prod.cache_clear()

def clear_cache_sop_T_coupled():
    """
    Clears the cache of the `_sop_T_coupled()` function.
    """
    # Clear the cache
    _sop_T_coupled.cache_clear()