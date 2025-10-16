"""
This module provides functions for calculating quantum mechanical spin operators
in Hilbert space. It includes functions for single-spin operators as well as
many-spin product operators.
"""

# Type checking
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import numpy as np
    import scipy.sparse as sp

# Imports
from spinguin._core._operators import (
    op_E as _op_E,
    op_Sx as _op_Sx,
    op_Sy as _op_Sy,
    op_Sz as _op_Sz,
    op_Sp as _op_Sp,
    op_Sm as _op_Sm,
    op_T as _op_T,
    op_T_coupled as _op_T_coupled,
)
from spinguin._core._config import config

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
    E = _op_E(S, config.sparse_operator)
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
    Sx = _op_Sx(S, config.sparse_operator)
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
    Sy = _op_Sy(S, config.sparse_operator)
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
    Sz = _op_Sz(S, config.sparse_operator)
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
    Sp = _op_Sp(S, config.sparse_operator)
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
    Sm = _op_Sm(S, config.sparse_operator)
    return Sm

def op_T(S: float,
         l: int,
         q: int) -> np.ndarray | sp.csc_array:
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
    T = _op_T(S, l, q, config.sparse_operator)
    return T

def op_T_coupled(l: int,  q: int,
                 l1: int, s1: float,
                 l2: int, s2: float) -> np.ndarray | sp.csc_array:
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
    T = _op_T_coupled(l, q, l1, s1, l2, s2, config.sparse_operator)
    return T