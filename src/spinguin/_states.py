"""
states.py

This module provides functions for creating state vectors.
"""

# For referencing the SpinSystem class
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from spinguin._spin_system import SpinSystem

# Imports
import numpy as np
import scipy.constants as const
from scipy.sparse import lil_array, csc_array, issparse
from spinguin import _la
from spinguin._operators import op_prod
from spinguin._hamiltonian import hamiltonian
from spinguin._basis import parse_operator_string, state_idx
from typing import Union
from functools import lru_cache

def unit_state(spin_system: SpinSystem, sparse: bool=False, normalized: bool=True) -> Union[np.ndarray, csc_array]:
    """
    Returns a unit state vector. This is equivalent to the density matrix, which has
    ones on the diagonal. Because the basis set is normalized, the coefficient of the
    unit operator in the state vector is equal to the norm of the unit operator.

    Parameters
    ----------
    spin_system : SpinSystem
        The spin system for which the unit state vector is generated.
    sparse : bool
        If False (default), returns a NumPy array. If True, returns a SciPy csc_array.
    normalized : bool
        Default: True. If set to True, the function will return a state vector that
        represents the trace-normalized density matrix. If False, returns a state vector
        that corresponds to the identity operator.

    Returns
    -------
    rho : numpy.ndarray or csc_array
        State vector corresponding to the unit state.
    """

    # Extract necessary information from the spin system
    dim = spin_system.basis.dim
    mults = spin_system.mults

    # Initialize the state vector
    rho = lil_array((dim, 1), dtype=complex)

    # Assign unit state coefficient
    if normalized:
        rho[0, 0] = 1 / np.sqrt(np.prod(mults))
    else:
        rho[0, 0] = np.sqrt(np.prod(mults))

    # Convert to csc_array if requesting sparse
    if sparse:
        rho = rho.tocsc()

    # Convert to NumPy if requesting dense
    else:
        rho = rho.toarray()

    return rho

@lru_cache(maxsize=128)
def state(spin_system: SpinSystem, operator: str, sparse: bool=False) -> Union[np.ndarray, csc_array]:
    """
    Generates a state vector from the given operator string. The output of the state
    function corresponds to a density matrix, which is expressed as a linear combination
    of the basis set operators. The output of this function is a column vector, which
    contains the coefficients.
    
    Normalization:
    The basis set operators are constructed from products of single-spin spherical tensor
    operators and they are normalized. Therefore, requesting a state that corresponds
    to any operator O will result in a coefficient of norm(O) for the state.

    Parameters
    ----------
    spin_system : SpinSystem
        The spin system for which the state vector is generated.
    operator : str
        Defines the state to be generated. The operator string must follow the rules below:

        - Cartesian and ladder operators: I(component,index). Example: I(x,4) --> Creates x-operator for spin at index 4.
        - Spherical tensor operators: T(l,q,index). Example: T(1,-1,3) --> Creates operator with l=1, q=-1 for spin at index 3.
        - Product operators have `*` in between the single-spin operators: I(z,0) * I(z,1)
        - Sums of operators have `+` in between the operators: I(x,0) + I(x,1)
        - The unit operator is not typed. Example: I(z,2) will generate E*I_z in case of a two-spin system. 
        - Whitespace will be ignored in the input.
    sparse : bool
        If False (default), returns a NumPy array. If True, returns a SciPy csc_array.

    Returns
    -------
    rho : numpy.ndarray or csc_array
        State vector corresponding to the requested state.
    """

    # Extract the necessary information from the spin system
    dim = spin_system.basis.dim
    mults = spin_system.mults
    spins = spin_system.spins

    # Initialize the state vector
    rho = lil_array((dim, 1), dtype=complex)

    # Get the operator definition and coefficients
    op_defs, coeffs = parse_operator_string(spin_system, operator)

    # Get the state indices
    idxs = [state_idx(spin_system, op_def) for op_def in op_defs]

    # Assign the state
    for idx, coeff, op_def in zip(idxs, coeffs, op_defs):

        # Find indices of the active and inactive spins
        idx_active = np.where(np.array(op_def) != 0)[0]
        idx_inactive = np.where(np.array(op_def) == 0)[0]

        # Calculate the norm of the active operator part if there are active spins
        if len(idx_active) != 0:
            op_norm = np.linalg.norm(op_prod(op_def, spins, include_unit=False), ord='fro')

        # Otherwise set it to one
        else:
            op_norm = 1
        
        # Calculate the norm of the unit operator part
        unit_norm = np.sqrt(np.prod(mults[idx_inactive]))

        # Total norm of the operator
        norm = op_norm * unit_norm

        # Set the properly normalized coefficient
        rho[idx, 0] = coeff * norm

    # Convert to csc_array if requesting sparse
    if sparse:
        rho = rho.tocsc()

    # Convert to NumPy if requesting dense
    else:
        rho = rho.toarray()

    return rho

def rho_to_zeeman(spin_system: SpinSystem, rho: Union[np.ndarray, csc_array]) -> np.ndarray:
    """
    Takes the state vector defined in the normalized spherical tensor basis
    and converts it into the Zeeman eigenbasis. Useful for error checking.

    Parameters
    ----------
    spin_system : SpinSystem
        The spin system for which the conversion is performed.
    rho : numpy.ndarray or csc_array
        State vector defined in the normalized spherical tensor basis.

    Returns
    -------
    rho_zeeman : numpy.ndarray
        Spin density matrix defined in the Zeeman eigenbasis.
    """

    # Extract the necessary information from the spin system
    mults = spin_system.mults
    basis = spin_system.basis.arr
    spins = spin_system.spins

    # Get the density matrix size
    size = np.prod(mults)

    # Initialize the spin density matrix
    rho_zeeman = np.zeros((size, size), dtype=complex)
    
    # Obtain indices of the non-zero coefficients from the state vector
    idx_nonzero = rho.nonzero()[0]

    # Loop over the nonzero indices
    for idx in idx_nonzero:

        # Get the corresponding operator definition
        op_def = basis[idx]

        # Get the normalized product operator in the Zeeman eigenbasis with normalization
        oper = op_prod(op_def, spins, include_unit=True)
        oper = oper / np.linalg.norm(oper, ord='fro')
        
        # Add to the total density matrix
        rho_zeeman += rho[idx, 0] * oper
    
    return rho_zeeman

def equilibrium_state(spin_system: SpinSystem, T: float, B: float, sparse: bool = False, zero_value: float = 1e-18) -> Union[np.ndarray, csc_array]:
    """
    Returns the state vector corresponding to thermal equilibrium.

    Parameters
    ----------
    spin_system : SpinSystem
        The spin system for which the thermal equilibrium state is generated.
    T : float
        Temperature in Kelvin.
    B : float
        Magnetic field in Tesla.
    sparse : bool
        If False (default), returns a NumPy array. If True, returns a SciPy csc_array.
    zero_value : float
        Default: 1e-18. Used to estimate the convergence of the matrix exponential.

    Returns
    -------
    rho_eq : numpy.ndarray or csc_array
        Thermal equilibrium state vector.
    """

    # Extract the necessary information from the spin system
    mults = spin_system.mults

    # Build the left Hamiltonian superoperator
    H = hamiltonian(spin_system, B, 'left', disable_outputs=True)

    # Get the matrix exponential corresponding to the Boltzmann distribution
    P = _la.expm(-const.hbar / (const.k * T) * H, zero_value)

    # Obtain the thermal equilibrium by propagating the unit state
    unit = unit_state(spin_system, sparse=sparse, normalized=False)
    rho_eq = P @ unit

    # Normalize such that the trace of the corresponding density matrix is one
    rho_eq = rho_eq / (rho_eq[0, 0] * np.sqrt(np.prod(mults)))

    return rho_eq

def alpha_state(spin_system: SpinSystem, index: int, sparse: bool = False) -> Union[np.ndarray, csc_array]:
    """
    Generates the alpha state for a given spin-1/2 nucleus. Unit state is assigned to the
    other spins.

    Parameters
    ----------
    spin_system : SpinSystem
        The spin system for which the alpha state is generated.
    index : int
        Index of the spin that has the alpha state.
    sparse : bool
        If False (default), returns a NumPy array. If True, returns a SciPy csc_array.

    Returns
    -------
    rho : numpy.ndarray or csc_array
        State vector corresponding to the alpha state of the given spin index.
    """

    # Obtain the necessary information from the spin system
    mults = spin_system.mults
    dim = np.prod(mults)

    # Get states
    E = unit_state(spin_system, sparse=sparse, normalized=False)
    I_z = state(spin_system, f"I(z, {index})", sparse=sparse)

    # Make the alpha state
    rho = 1 / dim * E + 2 / dim * I_z

    return rho

def beta_state(spin_system: SpinSystem, index: int, sparse: bool = False) -> Union[np.ndarray, csc_array]:
    """
    Generates the beta state for a given spin-1/2 nucleus. Unit state is assigned to the
    other spins.

    Parameters
    ----------
    spin_system : SpinSystem
        The spin system for which the beta state is generated.
    index : int
        Index of the spin that has the beta state.
    sparse : bool
        If False (default), returns a NumPy array. If True, returns a SciPy csc_array.

    Returns
    -------
    rho : numpy.ndarray or csc_array
        State vector corresponding to the beta state of the given spin index.
    """

    # Obtain the necessary information from the spin system
    mults = spin_system.mults
    dim = np.prod(mults)

    # Get states
    E = unit_state(spin_system, sparse=sparse, normalized=False)
    I_z = state(spin_system, f"I(z, {index})", sparse=sparse)

    # Make the beta state
    rho = 1 / dim * E - 2 / dim * I_z

    return rho

def singlet_state(spin_system: SpinSystem, index_1: int, index_2: int, sparse: bool = False) -> Union[np.ndarray, csc_array]:
    """
    Generates the singlet state between two spin-1/2 nuclei. Unit state is assigned to the
    other spins.

    Parameters
    ----------
    spin_system : SpinSystem
        The spin system for which the singlet state is generated.
    index_1 : int
        Index of the first spin in the singlet state.
    index_2 : int
        Index of the second spin in the singlet state.
    sparse : bool
        If False (default), returns a NumPy array. If True, returns a SciPy csc_array.

    Returns
    -------
    rho : numpy.ndarray or csc_array
        State vector corresponding to the singlet state.
    """

    # Obtain the necessary information from the spin system
    mults = spin_system.mults
    dim = np.prod(mults)

    # Get states
    E = unit_state(spin_system, sparse=sparse, normalized=False)
    IzIz = state(spin_system, f"I(z,{index_1}) * I(z, {index_2})", sparse=sparse)
    IpIm = state(spin_system, f"I(+,{index_1}) * I(-, {index_2})", sparse=sparse)
    ImIp = state(spin_system, f"I(-,{index_1}) * I(+, {index_2})", sparse=sparse)

    # Make the singlet
    rho = 1 / dim * E - 4 / dim * IzIz - 2 / dim * (IpIm + ImIp)

    return rho

def triplet_zero_state(spin_system: SpinSystem, index_1: int, index_2: int, sparse: bool = False) -> Union[np.ndarray, csc_array]:
    """
    Generates the triplet zero state between two spin-1/2 nuclei. Unit state is assigned to the
    other spins.

    Parameters
    ----------
    spin_system : SpinSystem
        The spin system for which the triplet zero state is generated.
    index_1 : int
        Index of the first spin in the triplet zero state.
    index_2 : int
        Index of the second spin in the triplet zero state.
    sparse : bool
        If False (default), returns a NumPy array. If True, returns a SciPy csc_array.

    Returns
    -------
    rho : numpy.ndarray or csc_array
        State vector corresponding to the triplet zero state.
    """

    # Obtain the necessary information from the spin system
    mults = spin_system.mults
    dim = np.prod(mults)

    # Get states
    E = unit_state(spin_system, sparse=sparse, normalized=False)
    IzIz = state(spin_system, f"I(z,{index_1}) * I(z, {index_2})", sparse=sparse)
    IpIm = state(spin_system, f"I(+,{index_1}) * I(-, {index_2})", sparse=sparse)
    ImIp = state(spin_system, f"I(-,{index_1}) * I(+, {index_2})", sparse=sparse)

    # Make the triplet zero
    rho = 1 / dim * E - 4 / dim * IzIz + 2 / dim * (IpIm + ImIp)

    return rho

def triplet_plus_state(spin_system: SpinSystem, index_1: int, index_2: int, sparse: bool = False) -> Union[np.ndarray, csc_array]:
    """
    Generates the triplet plus state between two spin-1/2 nuclei. Unit state is assigned to the
    other spins.

    Parameters
    ----------
    spin_system : SpinSystem
        The spin system for which the triplet plus state is generated.
    index_1 : int
        Index of the first spin in the triplet plus state.
    index_2 : int
        Index of the second spin in the triplet plus state.
    sparse : bool
        If False (default), returns a NumPy array. If True, returns a SciPy csc_array.

    Returns
    -------
    rho : numpy.ndarray or csc_array
        State vector corresponding to the triplet plus state.
    """

    # Obtain the necessary information from the spin system
    mults = spin_system.mults
    dim = np.prod(mults)

    # Get states
    E = unit_state(spin_system, sparse=sparse, normalized=False)
    IzE = state(spin_system, f"I(z, {index_1})", sparse=sparse)
    EIz = state(spin_system, f"I(z, {index_2})", sparse=sparse)
    IzIz = state(spin_system, f"I(z,{index_1}) * I(z, {index_2})", sparse=sparse)

    # Make the triplet plus
    rho = 1 / dim * E + 2 / dim * IzE + 2 / dim * EIz + 4 / dim * IzIz

    return rho

def triplet_minus_state(spin_system: SpinSystem, index_1: int, index_2: int, sparse: bool = False) -> Union[np.ndarray, csc_array]:
    """
    Generates the triplet minus state between two spin-1/2 nuclei. Unit state is assigned to the
    other spins.

    Parameters
    ----------
    spin_system : SpinSystem
        The spin system for which the triplet minus state is generated.
    index_1 : int
        Index of the first spin in the triplet minus state.
    index_2 : int
        Index of the second spin in the triplet minus state.
    sparse : bool
        If False (default), returns a NumPy array. If True, returns a SciPy csc_array.

    Returns
    -------
    rho : numpy.ndarray or csc_array
        State vector corresponding to the triplet minus state.
    """

    # Obtain the necessary information from the spin system
    mults = spin_system.mults
    dim = np.prod(mults)

    # Get states
    E = unit_state(spin_system, sparse=sparse, normalized=False)
    IzE = state(spin_system, f"I(z, {index_1})", sparse=sparse)
    EIz = state(spin_system, f"I(z, {index_2})", sparse=sparse)
    IzIz = state(spin_system, f"I(z,{index_1}) * I(z, {index_2})", sparse=sparse)

    # Make the triplet minus
    rho = 1 / dim * E - 2 / dim * IzE - 2 / dim * EIz + 4 / dim * IzIz

    return rho

def measure(spin_system: SpinSystem, rho: Union[np.ndarray, csc_array], operator: str) -> complex:
    """
    Computes the expectation value of the specified operator for a given state vector.
    Assumes that the state vector `rho` represents a trace-normalized density matrix.

    Parameters
    ----------
    spin_system : SpinSystem
        The spin system for which the measurement is performed.
    rho : numpy.ndarray or csc_array
        State vector that describes the density matrix.
    operators : str or tuple
        Defines the operator whose expectation value is to be measured.
        Can be either a string or a tuple of strings.
        - str :
            Generates an "operator" for each spin specified in `indices`, measures the
            expectation value for each of them, and returns the sum of the expectation
            values.
            For example: 'I_z'
        - tuple :
            Generates a "product operator" and measures its expectation value. Each spin
            that participates in the product operator is defined in `indices`. Must match
            the length of `indices`.
            For example: ('I_z', 'I_z')

    Returns
    -------
    ex : complex
        Expectation value.
    """

    # Get the "operator" to be measured
    oper = state(spin_system, operator, sparse=issparse(rho))

    # Perform the measurement
    ex = (oper.conj().T @ rho).trace()

    return ex