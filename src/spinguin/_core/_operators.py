"""
This module provides functions for calculating quantum mechanical spin operators
in Hilbert space. It includes functions for single-spin operators as well as
many-spin product operators.
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
from spinguin._core._la import comm, CG_coeff
from spinguin._core._utils import idx_to_lq, parse_operator_string
from spinguin._core._parameters import parameters

def op_E(S: float) -> np.ndarray | sp.csc_array:
    """
    Generates the unit operator for a given spin quantum number `S`.

    Parameters
    ----------
    S : float
        Spin quantum number.

    Returns
    -------
    E : ndarray or csc_array
        An array representing the unit operator.
    """
    # Generate a unit operator of the correct dimension
    dim = int(2 * S + 1)
    if parameters.sparse_operator:
        E = sp.eye_array(dim, format="csc", dtype=int)
    else:
        E = np.eye(dim, dtype=int)

    return E

def op_Sx(S: float) -> np.ndarray | sp.csc_array:
    """
    Generates the spin operator Sx for a given spin quantum number `S`.

    Parameters
    ----------
    S : float
        Spin quantum number.

    Returns
    -------
    Sx : ndarray or csc_array
        An array representing the x-component spin operator.
    """
    # Calculate Sx using the raising and lowering operators
    Sx = 1 / 2 * (op_Sp(S) + op_Sm(S))

    return Sx

def op_Sy(S: float) -> np.ndarray | sp.csc_array:
    """
    Generates the spin operator Sy for a given spin quantum number `S`.

    Parameters
    ----------
    S : float
        Spin quantum number.

    Returns
    -------
    Sy : ndarray or csc_array
        An array representing the y-component spin operator.
    """
    # Calculate Sy using the raising and lowering operators
    Sy = 1 / (2j) * (op_Sp(S) - op_Sm(S))

    return Sy

def op_Sz(S: float) -> np.ndarray | sp.csc_array:
    """
    Generates the spin operator Sz for a given spin quantum number `S`.

    Parameters
    ----------
    S : float
        Spin quantum number.

    Returns
    -------
    Sz : ndarray or csc_array
        An array representing the z-component spin operator.
    """
    # Get the possible spin magnetic quantum numbers (from largest to smallest)
    m = -np.arange(-S, S + 1)

    # Initialize the operator
    if parameters.sparse_operator:
        Sz = sp.lil_array((len(m), len(m)), dtype=float)
    else:
        Sz = np.zeros((len(m), len(m)), dtype=float)

    # Populate the diagonal elements
    for i in range(len(m)):
        Sz[i, i] = m[i]

    # Convert to CSC if sparse
    if parameters.sparse_operator:
        Sz = Sz.tocsc()

    return Sz

def op_Sp(S: float) -> np.ndarray | sp.csc_array:
    """
    Generates the spin raising operator for a given spin quantum number `S`.

    Parameters
    ----------
    S : float
        Spin quantum number.

    Returns
    -------
    Sp : ndarray or csc_array
        An array representing the raising operator.
    """
    # Get the possible spin magnetic quantum numbers
    m = np.arange(-S, S + 1)

    # Initialize the operator
    if parameters.sparse_operator:
        Sp = sp.lil_array((len(m), len(m)), dtype=float)
    else:
        Sp = np.zeros((len(m), len(m)), dtype=float)

    # Populate the off-diagonal elements
    for i in range(len(m) - 1):
        Sp[i, i + 1] = np.sqrt(S * (S + 1) - m[i] * (m[i] + 1))

    # Convert to CSC if sparse
    if parameters.sparse_operator:
        Sp = Sp.tocsc()

    return Sp

def op_Sm(S: float) -> np.ndarray | sp.csc_array:
    """
    Generates the spin lowering operator for a given spin quantum number `S`.

    Parameters
    ----------
    S : float
        Spin quantum number.

    Returns
    -------
    Sm : ndarray or csc_array
        An array representing the lowering operator.
    """
    # Get the possible spin magnetic quantum numbers
    m = np.arange(-S, S + 1)

    # Initialize the operator
    if parameters.sparse_operator:
        Sm = sp.lil_array((len(m), len(m)), dtype=complex)
    else:
        Sm = np.zeros((len(m), len(m)), dtype=complex)

    # Populate the off-diagonal elements
    for i in range(1, len(m)):
        Sm[i, i - 1] = np.sqrt(S * (S + 1) - m[i] * (m[i] - 1))

    # Convert to CSC if sparse
    if parameters.sparse_operator:
        Sm = Sm.tocsc()

    return Sm

@lru_cache(maxsize=1024)
def _op_T(
    S: float,
    l: int,
    q: int,
    sparse: bool
) -> np.ndarray | sp.csc_array:

    # Calculate the operator with maximum projection q = l
    if sparse:
        T = (-1)**l * 2**(-l / 2) * sp.linalg.matrix_power(op_Sp(S), l)
    else:
        T = (-1)**l * 2**(-l / 2) * np.linalg.matrix_power(op_Sp(S), l)

    # Perform the necessary number of lowerings
    for i in range(l - q):

        # Get the current q
        q = l - i

        # Perform the lowering
        T = comm(op_Sm(S), T) / np.sqrt(l * (l + 1) - q * (q - 1))

    return T

def op_T(S: float, l: int, q: int) -> np.ndarray | sp.csc_array:
    """
    Generates the numerical spherical tensor operator for a given spin quantum
    number `S`, rank `l`, and projection `q`. The operator is obtained by
    sequential lowering of the maximum projection operator.

    Source: Kuprov (2023) - Spin: From Basic Symmetries to Quantum Optimal
    Control, page 94.

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
    T : ndarray or csc_array
        An array representing the spherical tensor operator.
    """

    # Create the operator and ensure a separate copy is returned
    T = _op_T(S, l, q, parameters.sparse_operator).copy()

    return T

def op_T_coupled(
    l: int,  q: int,
    l1: int, s1: float,
    l2: int, s2: float,
) -> np.ndarray | sp.csc_array:
    """
    Computes the coupled irreducible spherical tensor of rank `l` and projection
    `q` from two irreducible spherical tensors of ranks `l1` and `l2`.

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
    T : ndarray or csc_array
        Coupled spherical tensor operator of rank `l` and projection `q`.
    """
    # Initialize the operator
    dim = int((2 * s1 + 1) * (2 * s2 + 1))
    if parameters.sparse_operator:
        T = sp.csc_array((dim, dim), dtype=float)
    else:
        T = np.zeros((dim, dim), dtype=float)

    # Iterate over the projections
    for q1 in range(-l1, l1 + 1):
        for q2 in range(-l2, l2 + 1):

            # Analogously to the coupling of angular momenta
            if parameters.sparse_operator:
                T = T + CG_coeff(l1, q1, l2, q2, l, q) * sp.kron(
                    op_T(s1, l1, q1),
                    op_T(s2, l2, q2),
                    format="csc"
                )
            else:
                T = T + CG_coeff(l1, q1, l2, q2, l, q) * np.kron(
                    op_T(s1, l1, q1), op_T(s2, l2, q2)
                )

    return T

def op_prod(
    op_def: np.ndarray,
    spins: np.ndarray,
    include_unit: bool=True
) -> np.ndarray | sp.csc_array:
    """
    Generates a product operator defined by `op_def` in the Zeeman eigenbasis.

    Parameters
    ----------
    op_def : ndarray
        Specifies the product operator to be generated. For example, input
        `np.array([0, 2, 0, 1])` will generate `E*T_10*E*T_11`. The indices are
        given by `N = l^2 + l - q`, where `l` is the rank and `q` is the
        projection.
    spins : ndarray
        Spin quantum numbers. Must match the length of `op_def`.
    include_unit : bool, default=True
        Specifies whether unit operators are included in the product operator.

    Returns
    -------
    op : ndarray or csc_array
        Product operator in the Zeeman eigenbasis.
    """

    # Convert input to NumPy
    op_def = np.asarray(op_def)
    spins = np.asarray(spins)

    # Initialize the product operator
    if parameters.sparse_operator:
        op = sp.csc_array([[1]], dtype=float)
    else:
        op = np.array([[1]], dtype=float)

    # Iterate through the operator definition
    for spin, oper in zip(spins, op_def):

        # Exclude unit operators if requested
        if include_unit or oper != 0:

            # Get the rank and projection
            l, q = idx_to_lq(oper)

            # Add to the product operator
            if parameters.sparse_operator:
                op = sp.kron(op, op_T(spin, l, q), format="csc")
            else:
                op = np.kron(op, op_T(spin, l, q))

    return op

def op_from_string(
    spins: np.ndarray,
    operator: str
) -> np.ndarray | sp.csc_array:
    """
    Generates an operator for the `spin_system` in Hilbert space from the user-
    specified `operators` string.

    Parameters
    ----------
    spins : ndarray
        A one-dimensional array containing the spin quantum numbers of the spin
        system.
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

    Returns
    -------
    op : ndarray or csc_array
        An array representing the requested operator.
    """

    # Extract information from the spins
    nspins = spins.shape[0]
    dim = int(np.prod(2*spins + 1))

    # Initialize the operator
    if parameters.sparse_operator:
        op = sp.csc_array((dim, dim), dtype=float)
    else:
        op = np.zeros((dim, dim), dtype=float)

    # Get the operator definitions and coefficients
    op_defs, coeffs = parse_operator_string(operator, nspins)

    # Construct the operator
    for op_def, coeff in zip(op_defs, coeffs):
        op = op + coeff * op_prod(op_def, spins, include_unit=True)

    return op

def operator(
    spin_system: SpinSystem,
    operator: str | list | np.ndarray | tuple
) -> np.ndarray | sp.csc_array:
    """
    Generates an operator for the `spin_system` in Hilbert space from the
    user-specified `operator` string.

    Parameters
    ----------
    spin_system : SpinSystem
        Spin system for which the operator is going to be generated.
    operator : str or list or ndarray or tuple
        Defines the operator to be generated.
        
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

    Returns
    -------
    op : ndarray or csc_array
        An array representing the requested operator.
    """
    # Construct the operator using string input
    if isinstance(operator, str):
        op = op_from_string(
            spins = spin_system.spins,
            operator = operator
        )

    # Construct the operator using array input
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

        # Construct the operator
        op = op_prod(
            spins = spin_system.spins,
            op_def = operator,
            include_unit = True
        )

    # Otherwise raise an error
    else:
        raise ValueError("Invalid input type for 'operator'")
    
    return op

def clear_cache_op_T():
    """
    Clears the cache of the `_op_T()` function.
    """
    # Clear the cache
    _op_T.cache_clear()