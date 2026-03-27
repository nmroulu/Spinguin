"""
superoperators.py

Provides utilities for constructing Liouville-space superoperators in either
the full basis or a truncated basis.
"""

# Reference the SpinSystem class for type checking.
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from spinguin._core._spin_system import SpinSystem

# Imports
from functools import lru_cache
from itertools import product
from typing import Literal

import numpy as np
import scipy.sparse as sp

from spinguin._core import _la
from spinguin._core._operators import op_T
from spinguin._core._parameters import parameters
from spinguin._core._utils import idx_to_lq, parse_operator_string


SuperoperatorLike = np.ndarray | sp.csc_array
OperatorInput = str | list | np.ndarray | tuple


def _require_basis(
    spin_system: SpinSystem,
    action: str,
) -> None:
    """
    Ensure that the basis has been built before superoperator construction.

    Parameters
    ----------
    spin_system : SpinSystem
        Spin system whose basis is required.
    action : str
        Description of the attempted operation for error reporting.

    Returns
    -------
    None
        Validation is performed for its side effect only.
    """

    # Reject operations that require a built basis when none is present.
    if spin_system.basis.basis is None:
        raise ValueError(f"Please build the basis before {action}.")


def _allocate_superoperator(
    dim: int,
    sparse: bool,
) -> SuperoperatorLike:
    """
    Allocate a square superoperator matrix in the requested storage format.

    Parameters
    ----------
    dim : int
        Dimension of the basis set.
    sparse : bool
        If `True`, allocate sparse CSC storage; otherwise allocate a dense
        NumPy array.

    Returns
    -------
    ndarray or csc_array
        Zero-initialised superoperator matrix.
    """

    # Allocate sparse or dense storage according to the requested format.
    if sparse:
        return sp.csc_array((dim, dim), dtype=complex)

    return np.zeros((dim, dim), dtype=complex)


def _finalise_superoperator(
    sop: SuperoperatorLike,
    sparse: bool,
) -> SuperoperatorLike:
    """
    Return a superoperator in the requested output storage format.

    Parameters
    ----------
    sop : ndarray or csc_array
        Superoperator to be finalised.
    sparse : bool
        Desired storage format of the output.

    Returns
    -------
    ndarray or csc_array
        Final superoperator.
    """

    # Convert sparse matrices to dense arrays when requested.
    if not sparse and sp.issparse(sop):
        return sop.toarray()

    return sop


@lru_cache(maxsize=16)
def structure_coefficients(
    spin: float,
    side: Literal["left", "right"],
) -> np.ndarray:
    """
    Compute the normalised structure coefficients for a single spin.

    The construction follows Eq. 24 of the following paper, with the additional
    basis normalisation used internally here:

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

    # Determine the Hilbert-space multiplicity of the spin.
    mult = int(2 * spin + 1)

    # Allocate the full structure-coefficient tensor.
    c = np.zeros((mult ** 2, mult ** 2, mult ** 2), dtype=complex)

    # Loop over the left basis index.
    for j in range(mult ** 2):

        # Construct the spherical tensor corresponding to index `j`.
        l_j, q_j = idx_to_lq(j)
        T_j = op_T(spin, l_j, q_j)

        # Loop over the right basis index.
        for k in range(mult ** 2):

            # Construct the spherical tensor corresponding to index `k`.
            l_k, q_k = idx_to_lq(k)
            T_k = op_T(spin, l_k, q_k)

            # Compute the normalisation factor of the operator product basis.
            norm = np.sqrt(
                (T_j.conj().T @ T_j).trace() * (T_k.conj().T @ T_k).trace()
            )

            # Loop over the output basis index.
            for i in range(mult ** 2):

                # Construct the spherical tensor corresponding to index `i`.
                l_i, q_i = idx_to_lq(i)
                T_i = op_T(spin, l_i, q_i)

                # Evaluate the structure coefficient for the selected side.
                if side == 'left':
                    c[i, j, k] = (T_j.conj().T @ T_i @ T_k).trace() / norm
                elif side == 'right':
                    c[i, j, k] = (T_j.conj().T @ T_k @ T_i).trace() / norm
                else:
                    raise ValueError(
                        "The 'side' parameter must be either 'left' or 'right'."
                    )

    return c


def sop_E(
    dim: int,
) -> SuperoperatorLike:
    """
    Return the unit superoperator.

    Parameters
    ----------
    dim : int
        Dimension of the basis set.

    Returns
    -------
    unit : ndarray or csc_array
        A sparse array corresponding to the unit operator.
    """

    # Construct the identity in the configured storage format.
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
    sparse: bool=True,
) -> SuperoperatorLike:
    """
    Construct a product superoperator from hashable byte representations.

    Parameters
    ----------
    op_def_bytes : bytes
        Product-operator definition encoded as bytes.
    basis_bytes : bytes
        Basis-set array encoded as bytes.
    spins_bytes : bytes
        Spin-quantum-number array encoded as bytes.
    side : {'comm', 'left', 'right'}
        Type of superoperator to be constructed.
    sparse : bool, default=True
        If `True`, return a sparse CSC array.

    Returns
    -------
    ndarray or csc_array
        Requested product superoperator.
    """

    # Construct the commutation superoperator as left minus right action.
    if side == 'comm':
        return (
            _sop_prod(op_def_bytes, basis_bytes, spins_bytes, 'left', sparse)
            - _sop_prod(op_def_bytes, basis_bytes, spins_bytes, 'right', sparse)
        )

    # Recover the hashed operator definition, basis, and spin array.
    op_def = np.frombuffer(op_def_bytes, dtype=int)
    basis = np.frombuffer(basis_bytes, dtype=int).reshape(-1, op_def.shape[0])
    spins = np.frombuffer(spins_bytes, dtype=float)

    # Determine the basis dimension.
    dim = basis.shape[0]

    # Identify the spins that participate in the product operator.
    idx_spins = np.nonzero(op_def)[0]

    # Restrict the basis to the participating spins only.
    sub_basis = basis[:, idx_spins]

    # Return the identity superoperator for the unit operator.
    if len(idx_spins) == 0:
        return _finalise_superoperator(sop_E(dim), sparse)

    # Collect non-zero structure coefficients and their tensor indices.
    c_jk = []
    j = []
    k = []

    # Gather the single-spin structure coefficients for each active spin.
    for n in idx_spins:

        # Select the slice corresponding to the current single-spin operator.
        c_jk_n = structure_coefficients(spins[n], side)[op_def[n], :, :]

        # Identify the non-zero entries of the coefficient matrix.
        nonzero_jk = np.nonzero(c_jk_n)

        # Append the coefficients and the associated basis indices.
        c_jk.append(c_jk_n[nonzero_jk])
        j.append(nonzero_jk[0])
        k.append(nonzero_jk[1])

    # Form all products of structure coefficients and their basis labels.
    prod_c_jk = np.array([np.prod(c_jk_n) for c_jk_n in product(*c_jk)])
    op_defs_j = np.array(list(product(*j)))
    op_defs_k = np.array(list(product(*k)))

    # Collect the matrix values and indices of the final superoperator.
    sop_vals = []
    sop_j = []
    sop_k = []

    # Match each active-spin operator pair against the full basis.
    for m in range(prod_c_jk.shape[0]):

        # Find basis rows matching the current active-spin operator labels.
        j_op = np.where(np.all(sub_basis == op_defs_j[m], axis=1))[0]
        k_op = np.where(np.all(sub_basis == op_defs_k[m], axis=1))[0]

        # Continue only if both operator definitions are present in the basis.
        if j_op.shape[0] != 0 and k_op.shape[0] != 0:

            # Recover the full operator definitions from the basis.
            op_def_j = basis[j_op, :]
            op_def_k = basis[k_op, :]

            # Retain only the spectator-spin parts of the operator labels.
            op_def_j = np.delete(op_def_j, idx_spins, axis=1)
            op_def_k = np.delete(op_def_k, idx_spins, axis=1)

            # Spectator-spin definitions must match for a non-zero matrix element.
            ind_j, ind_k = _la.find_common_rows(op_def_j, op_def_k)

            # Append the contributions and the corresponding matrix indices.
            sop_vals.append(prod_c_jk[m] * np.ones(len(ind_j)))
            sop_j.append(j_op[ind_j])
            sop_k.append(k_op[ind_k])

    # Concatenate the collected sparse matrix data.
    # NOTE: Thirty-two-bit integer indices are sufficient here.
    sop_j = np.concatenate(sop_j, dtype=np.int32)
    sop_k = np.concatenate(sop_k, dtype=np.int32)
    sop_vals = np.concatenate(sop_vals, dtype=complex)

    # Construct the sparse superoperator matrix.
    sop = sp.csc_array((sop_vals, (sop_j, sop_k)), shape=(dim, dim))

    return _finalise_superoperator(sop, sparse)


def sop_prod(
    op_def: np.ndarray,
    basis: np.ndarray,
    spins: np.ndarray,
    side: Literal["comm", "left", "right"],
) -> SuperoperatorLike:
    """
    Generate a product superoperator from an operator definition array.

    This function is called frequently and is cached for high performance.

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

    # Convert the inputs to byte strings so they can be cached.
    op_def_bytes = op_def.tobytes()
    basis_bytes = basis.tobytes()
    spins_bytes = spins.tobytes()

    # Build the cached superoperator and return an independent copy.
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
    side: Literal["comm", "left", "right"],
) -> SuperoperatorLike:
    """
    Generate a superoperator from an operator string.

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

    # Determine the basis dimension and the number of spins.
    dim = basis.shape[0]
    nspins = spins.shape[0]

    # Allocate the output superoperator.
    sop = _allocate_superoperator(dim, parameters.sparse_superoperator)

    # Parse the operator string into basis definitions and coefficients.
    op_defs, coeffs = parse_operator_string(operator, nspins)

    # Accumulate the contribution of each term in the operator string.
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
    sparse: bool=True,
) -> SuperoperatorLike:
    """
    Construct a coupled spherical-tensor superoperator from cached inputs.

    Parameters
    ----------
    basis_bytes : bytes
        Basis-set array encoded as bytes.
    spins_bytes : bytes
        Spin-quantum-number array encoded as bytes.
    l : int
        Rank of the coupled tensor.
    q : int
        Projection of the coupled tensor.
    spin_1 : int
        Index of the first spin.
    spin_2 : int, default=None
        Index of the second spin, or `None` for a linear interaction.
    sparse : bool, default=True
        If `True`, return a sparse CSC array.

    Returns
    -------
    ndarray or csc_array
        Coupled spherical-tensor superoperator.
    """

    # Recover the spin and basis arrays from their cached byte representation.
    spins = np.frombuffer(spins_bytes, dtype=float)
    basis = np.frombuffer(basis_bytes, dtype=int).reshape(-1, spins.shape[0])

    # Determine the basis dimension and the number of spins.
    dim = basis.shape[0]
    nspins = spins.shape[0]

    # Allocate the coupled superoperator.
    sop = _allocate_superoperator(dim, sparse)

    # Handle bilinear two-spin interactions.
    if isinstance(spin_2, int):

        # Sum over the projections of the coupled rank-1 tensors.
        for q1 in range(-1, 2):
            for q2 in range(-1, 2):

                # Build the product-operator definition of the coupled term.
                op_def = np.zeros(nspins, dtype=int)
                op_def[spin_1] = 2 - q1
                op_def[spin_2] = 2 - q2

                # Combine the term using the angular-momentum coupling rule.
                sop += _la.CG_coeff(1, q1, 1, q2, l, q) * \
                       sop_prod(op_def, basis, spins, 'comm')

    # Handle linear single-spin interactions.
    else:

        # Only the `(1, 0)` component of the second tensor contributes.
        for q1 in range(-1, 2):

            # Build the product-operator definition of the coupled term.
            op_def = np.zeros(nspins, dtype=int)
            op_def[spin_1] = 2 - q1

            # Combine the term using the angular-momentum coupling rule.
            sop += _la.CG_coeff(1, q1, 1, 0, l, q) * \
                   sop_prod(op_def, basis, spins, 'comm')

    return sop


def sop_T_coupled(
    spin_system: SpinSystem,
    l: int,
    q: int,
    spin_1: int,
    spin_2: int=None,
) -> SuperoperatorLike:
    """
    Compute the coupled spherical-tensor superoperator of rank `l`.

    This function is frequently called and is cached for high performance.

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

    # Ensure that the basis is available.
    _require_basis(spin_system, "constructing coupled superoperators")

    # Convert the basis and spin arrays to bytes for caching.
    basis_bytes = spin_system.basis.basis.tobytes()
    spins_bytes = spin_system.spins.tobytes()

    # Build the cached superoperator and return an independent copy.
    sop = _sop_T_coupled(
        basis_bytes,
        spins_bytes,
        l,
        q,
        spin_1,
        spin_2,
        parameters.sparse_superoperator,
    ).copy()

    return sop


def sop_to_truncated_basis(
    index_map: list,
    sop: SuperoperatorLike,
) -> SuperoperatorLike:
    """
    Transform a superoperator to a truncated basis.

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

    # Retain only the rows and columns selected by the truncation map.
    return sop[np.ix_(index_map, index_map)]


def superoperator(
    spin_system: SpinSystem,
    operator: OperatorInput,
    side: Literal["comm", "left", "right"]="comm",
) -> SuperoperatorLike:
    """
    Generate a Liouville-space superoperator for a spin system.

    Parameters
    ----------
    spin_system : SpinSystem
        Spin system for which the superoperator is going to be generated.
    operator : str or list or ndarray or tuple
        Defines the superoperator to be generated.
        
        **The operator can be defined with two different approaches:**

        - A string. Example: `I(z,0) * I(z,1)`
        - An array of integers. Example: `[2, 2]`
        
        **The operator string must follow the rules below:**

        - Cartesian or ladder operator at specific index or for all spins::

            operator = "I(component, index)"
            operator = "I(component)"

        - Spherical tensor operator at specific index or for all spins::

            operator = "T(l, q, index)"
            operator = "T(l, q)"

        - Product operators::

            operator = "I(component1, index1) * I(component2, index2)"

        - Sum of operators::

            operator = "I(component1, index1) + I(component2, index2)"
            
        - Unit operators are ignored in the input. These are identical::

            operator = "E * I(component, index)"
            operator = "I(component, index)"
        
        Special case: An empty `operator` string is considered as unit operator.

        Whitespace will be ignored in the input.

        **The array input is defined as follows:**

        - Each spin is given an integer *N* in the array.
        - Each integer corresponds to a spherical tensor operator of rank *l*
          and projection *q*: *N* = *l*^2 + *l* - *q*

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

    # Ensure that the basis is available.
    _require_basis(spin_system, "constructing superoperators")

    # Construct the superoperator from a string specification.
    if isinstance(operator, str):
        sop = sop_from_string(
            operator=operator,
            basis=spin_system.basis.basis,
            spins=spin_system.spins,
            side=side,
        )

    # Construct the superoperator from an explicit operator-definition array.
    elif isinstance(operator, (list, np.ndarray, tuple)):

        # Convert the input to a NumPy array.
        operator = np.asarray(operator)

        # Validate the dimensionality and length of the operator definition.
        if not operator.ndim == 1:
            raise ValueError("The input array must be one-dimensional")
        if not operator.shape[0] == spin_system.nspins:
            raise ValueError(
                "Length of the operator array must match the number of spins."
            )

        # Construct the requested product superoperator.
        sop = sop_prod(
            op_def=operator,
            basis=spin_system.basis.basis,
            spins=spin_system.spins,
            side=side,
        )

    # Reject unsupported operator specifications.
    else:
        raise ValueError("Invalid input type for 'operator'")

    return sop


def clear_cache_structure_coefficients() -> None:
    """
    Clear the cache of `structure_coefficients()`.
    """

    # Clear the cached structure-coefficient tensors.
    structure_coefficients.cache_clear()


def clear_cache_sop_prod() -> None:
    """
    Clear the cache of `_sop_prod()`.
    """

    # Clear the cached product superoperators.
    _sop_prod.cache_clear()


def clear_cache_sop_T_coupled() -> None:
    """
    Clear the cache of `_sop_T_coupled()`.
    """

    # Clear the cached coupled spherical-tensor superoperators.
    _sop_T_coupled.cache_clear()