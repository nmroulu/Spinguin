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