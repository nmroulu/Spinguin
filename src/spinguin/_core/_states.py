"""
This module provides functions for creating state vectors.
"""
# Referencing SpinSystem class
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from spinguin._core._spin_system import SpinSystem

# Imports
import numpy as np
import scipy.sparse as sp
import scipy.constants as const
import time
from functools import lru_cache
from spinguin._core._la import expm
from spinguin._core._operators import op_prod
from spinguin._core._utils import parse_operator_string, state_idx
from spinguin._core._hide_prints import HidePrints
from spinguin._core._parameters import parameters
from spinguin._core._hamiltonian import sop_H

def _unit_state(
    basis: np.ndarray,
    spins: np.ndarray,
    normalized: bool=True
) -> np.ndarray | sp.csc_array:
    """
    Returns a unit state vector. This is equivalent to the density matrix, which
    has ones on the diagonal. Because the basis set is normalized, the
    coefficient of the unit operator in the state vector is equal to the norm of
    the unit operator.

    Parameters
    ----------
    basis : ndarray
        A 2-dimensional array containing the basis set that contains sequences
        of integers describing the Kronecker products of irreducible spherical
        tensors.
    spins : ndarray
        A 1-dimensional array specifying the spin quantum numbers of the system.
    normalized : bool, default=True
        If set to True, the function will return a state vector that represents
        the trace-normalized density matrix. If False, returns a state vector
        that corresponds to the identity operator.

    Returns
    -------
    rho : ndarray or csc_array
        State vector corresponding to the unit state.
    """

    # Obtain the basis dimension
    dim = basis.shape[0]

    # Acquire the spin multiplicities
    mults = np.array([int(2 * S + 1) for S in spins], dtype=int)

    # Initialize the state vector
    if parameters.sparse_state:
        rho = sp.lil_array((dim, 1), dtype=complex)
    else:
        rho = np.zeros((dim, 1), dtype=complex)

    # Assign unit state coefficient
    if normalized:
        rho[0, 0] = 1 / np.sqrt(np.prod(mults))
    else:
        rho[0, 0] = np.sqrt(np.prod(mults))

    # Convert to csc_array if requesting sparse
    if parameters.sparse_state:
        rho = rho.tocsc()

    return rho

@lru_cache(maxsize=8192)
def _state_from_op_def(
    basis_bytes : bytes,
    spins_bytes : bytes,
    op_def_bytes : bytes,
    sparse: bool
) -> np.ndarray | sp.csc_array:
    
    # Obtain the hashed elements
    spins = np.frombuffer(spins_bytes, dtype=float)
    basis = np.frombuffer(basis_bytes, dtype=int).reshape(-1, spins.shape[0])
    op_def = np.frombuffer(op_def_bytes, dtype=int)

    # Obtain the basis dimension and spin multiplicities
    dim = basis.shape[0]
    mults = (2*spins + 1).astype(int)

    # Initialize the state vector
    if sparse:
        rho = sp.lil_array((dim, 1), dtype=complex)
    else:
        rho = np.zeros((dim, 1), dtype=complex)

    # Get the state index
    idx = state_idx(basis, op_def)

    # Find indices of the active and inactive spins
    idx_active = np.where(np.array(op_def) != 0)[0]
    idx_inactive = np.where(np.array(op_def) == 0)[0]

    # Calculate the norm of the active operator part if there are active
    # spins
    if len(idx_active) != 0:
        if parameters.sparse_operator:
            op_norm = sp.linalg.norm(
                op_prod(op_def, spins, include_unit=False), ord='fro'
            )
        else:
            op_norm = np.linalg.norm(
                op_prod(op_def, spins, include_unit=False), ord='fro'
            )

    # Otherwise set it to one
    else:
        op_norm = 1
    
    # Calculate the norm of the unit operator part
    unit_norm = np.sqrt(np.prod(mults[idx_inactive]))

    # Total norm of the operator
    norm = op_norm * unit_norm

    # Set the properly normalized coefficient
    rho[idx, 0] = norm

    # Convert to csc_array if requesting sparse
    if sparse:
        rho = rho.tocsc()

    return rho

def state_from_op_def(
    basis : np.ndarray,
    spins : np.ndarray,
    op_def : np.ndarray,
) -> np.ndarray | sp.csc_array:
    """
    Generates a state from the given operator definition. The output of this
    function is a column vector where the requested state has been populated.
    
    Normalization:
    The output of this function corresponds to the non-normalized operator.
    However, because the basis set operators are constructed from products of
    normalized single-spin spherical tensor operators, requesting a state that
    corresponds to any operator `O` will result in a coefficient of `norm(O)`
    for the state.

    NOTE: This function is sometimes called often and is cached for high
    performance.

    Parameters
    ----------
    basis : ndarray
        A 2-dimensional array containing the basis set that contains sequences
        of integers describing the Kronecker products of irreducible spherical
        tensors.
    spins : ndarray
        A 1-dimensional array specifying the spin quantum numbers of the system.
    op_def : ndarray
        An array of integers that specify the product operator.

    Returns
    -------
    rho : ndarray or csc_array
        State vector corresponding to the requested state.
    """

    # Convert types suitable for hashing
    basis_bytes = basis.tobytes()
    spins_bytes = spins.tobytes()
    op_def_bytes = op_def.tobytes()

    # Create the state and ensure that a different instance is returned
    rho = _state_from_op_def(
        basis_bytes,
        spins_bytes,
        op_def_bytes,
        parameters.sparse_state
    ).copy()

    return rho

@lru_cache(maxsize=128)
def _state_from_string(
    basis_bytes: bytes,
    spins_bytes: bytes,
    operator: str,
    sparse: bool
) -> np.ndarray | sp.csc_array:

    # Obtain the hashed elements
    spins = np.frombuffer(spins_bytes, dtype=float)
    basis = np.frombuffer(basis_bytes, dtype=int).reshape(-1, spins.shape[0]) 

    # Obtain the basis dimension, number of spins and spin multiplicities
    dim = basis.shape[0]
    nspins = spins.shape[0]
    mults = (2*spins + 1).astype(int)

    # Initialize the state vector
    if sparse:
        rho = sp.lil_array((dim, 1), dtype=complex)
    else:
        rho = np.zeros((dim, 1), dtype=complex)

    # Get the operator definition and coefficients
    op_defs, coeffs = parse_operator_string(operator, nspins)

    # Get the state indices
    idxs = [state_idx(basis, op_def) for op_def in op_defs]

    # Assign the state
    for idx, coeff, op_def in zip(idxs, coeffs, op_defs):

        # Find indices of the active and inactive spins
        idx_active = np.where(np.array(op_def) != 0)[0]
        idx_inactive = np.where(np.array(op_def) == 0)[0]

        # Calculate the norm of the active operator part if there are active
        # spins
        if len(idx_active) != 0:
            if parameters.sparse_operator:
                op_norm = sp.linalg.norm(
                    op_prod(op_def, spins, include_unit=False), ord='fro'
                )
            else:
                op_norm = np.linalg.norm(
                    op_prod(op_def, spins, include_unit=False), ord='fro'
                )

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

    return rho

def state_from_string(
    basis: np.ndarray,
    spins: np.ndarray,
    operator: str
) -> np.ndarray | sp.csc_array:
    """
    This function returns a column vector representing the density matrix as a
    linear combination of spin operators. Each element of the vector corresponds
    to the coefficient of a specific spin operator in the expansion.
    
    Normalization:
    The output of this function uses a normalised basis built from normalised
    products of single-spin spherical tensor operators. However, the
    coefficients are scaled so that the resulting linear combination represents
    the non-normalised version of the requested operator.

    NOTE: This function is sometimes called often and is cached for high
    performance.

    Parameters
    ----------
    basis : ndarray
        A 2-dimensional array containing the basis set that contains sequences
        of integers describing the Kronecker products of irreducible spherical
        tensors.
    spins : ndarray
        A 1-dimensional array specifying the spin quantum numbers of the system.
    operator : str
        Defines the state to be generated. The operator string must follow the
        rules below:

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
    rho : ndarray or csc_array
        State vector corresponding to the requested state.
    """
    
    # Convert types suitable for hashing
    basis_bytes = basis.tobytes()
    spins_bytes = spins.tobytes()

    # Create the state and ensure that a different instance is returned
    rho = _state_from_string(
        basis_bytes,
        spins_bytes,
        operator,
        parameters.sparse_state
    ).copy()

    return rho

def _state_to_zeeman(
    basis: np.ndarray,
    spins: np.ndarray,
    rho: np.ndarray | sp.csc_array
) -> np.ndarray | sp.csc_array:
    """
    Takes the state vector defined in the normalized spherical tensor basis
    and converts it into the Zeeman eigenbasis. Useful for error checking.

    Parameters
    ----------
    basis : ndarray
        A 2-dimensional array containing the basis set that contains sequences
        of integers describing the Kronecker products of irreducible spherical
        tensors.
    spins : ndarray
        A 1-dimensional array specifying the spin quantum numbers of the system.
    rho : ndarray or csc_array
        State vector defined in the normalized spherical tensor basis.

    Returns
    -------
    rho_zeeman : ndarray or csc_array
        Spin density matrix defined in the Zeeman eigenbasis.
    """

    # Obtain the spin multiplicities
    mults = (2*spins + 1).astype(int)

    # Get the dimension of density matrix
    dim = np.prod(mults)

    # Initialize the spin density matrix
    if parameters.sparse_operator:
        rho_zeeman = sp.csc_array((dim, dim), dtype=complex)
    else:
        rho_zeeman = np.zeros((dim, dim), dtype=complex)
    
    # Obtain indices of the non-zero coefficients from the state vector
    idx_nonzero = rho.nonzero()[0]

    # Loop over the nonzero indices
    for idx in idx_nonzero:

        # Get the corresponding operator definition
        op_def = basis[idx]

        # Get the normalized product operator in the Zeeman eigenbasis with
        # normalization
        oper = op_prod(op_def, spins, include_unit=True)
        if parameters.sparse_operator:
            oper = oper / sp.linalg.norm(oper, ord='fro')
        else:
            oper = oper / np.linalg.norm(oper, ord='fro')
        
        # Add to the total density matrix
        rho_zeeman += rho[idx, 0] * oper
    
    return rho_zeeman

def _equilibrium_state(
    basis: np.ndarray,
    spins: np.ndarray,
    H_left: np.ndarray | sp.csc_array,
    T : float,
    zero_value: float=1e-18
) -> np.ndarray | sp.csc_array:
    """
    Returns the state vector corresponding to thermal equilibrium.

    Parameters
    ----------
    basis : ndarray
        A 2-dimensional array containing the basis set that contains sequences
        of integers describing the Kronecker products of irreducible spherical
        tensors.
    spins : ndarray
        A 1-dimensional array specifying the spin quantum numbers of the system.
    H_left : ndarray
        Left-side coherent Hamiltonian superoperator.
    T : float
        Temperature of the spin bath in Kelvin.
    zero_value : float, default=1e-18
        This threshold value is used to estimate the convergence of Taylor
        series in matrix exponential, and to eliminate value smaller than this
        threshold while squaring the matrix during the matrix exponential.

    Returns
    -------
    rho_eq : ndarray or csc_array
        Thermal equilibrium state vector.
    """

    # Extract the necessary information from the spin system
    mults = (2*spins + 1).astype(int)

    # Get the matrix exponential corresponding to the Boltzmann distribution
    with HidePrints():
        P = expm(-const.hbar / (const.k * T) * H_left, zero_value)

    # Obtain the thermal equilibrium by propagating the unit state
    unit = _unit_state(basis, spins, normalized=False)
    rho_eq = P @ unit

    # Ensure the correct sparsity (might have changed during multiplication)
    if parameters.sparse_state and not sp.issparse(rho_eq):
        rho_eq = sp.csc_array(rho_eq)

    # Normalize such that the trace of the corresponding density matrix is one
    rho_eq = rho_eq / (rho_eq[0, 0] * np.sqrt(np.prod(mults)))

    return rho_eq

def _alpha_state(
    basis: np.ndarray,
    spins: np.ndarray,
    index: int
) -> np.ndarray | sp.csc_array:
    """
    Generates the alpha state for a given spin-1/2 nucleus. Unit state is
    assigned to the other spins.

    Parameters
    ----------
    basis : ndarray
        A 2-dimensional array containing the basis set that contains sequences
        of integers describing the Kronecker products of irreducible spherical
        tensors.
    spins : ndarray
        A 1-dimensional array specifying the spin quantum numbers of the system.
    index : int
        Index of the spin that has the alpha state.

    Returns
    -------
    rho : ndarray or csc_array
        State vector corresponding to the alpha state of the given spin index.
    """

    # Calculate the dimension of the full Liouville space
    mults = (2*spins+1).astype(int)
    dim = np.prod(mults)

    # Get states
    E = _unit_state(basis, spins, normalized=False)
    I_z = state_from_string(basis, spins, f"I(z, {index})")

    # Make the alpha state
    rho = 1 / dim * E + 2 / dim * I_z

    return rho

def _beta_state(
    basis: np.ndarray,
    spins: np.ndarray,
    index: int
) -> np.ndarray | sp.csc_array:
    """
    Generates the beta state for a given spin-1/2 nucleus. Unit state is
    assigned to the other spins.

    Parameters
    ----------
    basis : ndarray
        A 2-dimensional array containing the basis set that contains sequences
        of integers describing the Kronecker products of irreducible spherical
        tensors.
    spins : ndarray
        A 1-dimensional array specifying the spin quantum numbers of the system.
    index : int
        Index of the spin that has the beta state.

    Returns
    -------
    rho : ndarray or csc_array
        State vector corresponding to the beta state of the given spin index.
    """

    # Calculate the dimension of the full Liouville space
    mults = (2*spins+1).astype(int)
    dim = np.prod(mults)

    # Get states
    E = _unit_state(basis, spins, normalized=False)
    I_z = state_from_string(basis, spins, f"I(z, {index})")

    # Make the beta state
    rho = 1 / dim * E - 2 / dim * I_z

    return rho

def _singlet_state(
    basis: np.ndarray,
    spins: np.ndarray,
    index_1: int,
    index_2: int
) -> np.ndarray | sp.csc_array:
    """
    Generates the singlet state between two spin-1/2 nuclei. Unit state is
    assigned to the other spins.

    Parameters
    ----------
    basis : ndarray
        A 2-dimensional array containing the basis set that contains sequences
        of integers describing the Kronecker products of irreducible spherical
        tensors.
    spins : ndarray
        A 1-dimensional array specifying the spin quantum numbers of the system.
    index_1 : int
        Index of the first spin in the singlet state.
    index_2 : int
        Index of the second spin in the singlet state.

    Returns
    -------
    rho : ndarray or csc_array
        State vector corresponding to the singlet state.
    """

    # Calculate the dimension of the full Liouville space
    mults = (2*spins+1).astype(int)
    dim = np.prod(mults)

    # Get states
    E = _unit_state(basis, spins, normalized=False)
    IzIz = state_from_string(basis, spins, f"I(z,{index_1}) * I(z, {index_2})")
    IpIm = state_from_string(basis, spins, f"I(+,{index_1}) * I(-, {index_2})")
    ImIp = state_from_string(basis, spins, f"I(-,{index_1}) * I(+, {index_2})")

    # Make the singlet
    rho = 1 / dim * E - 4 / dim * IzIz - 2 / dim * (IpIm + ImIp)

    return rho

def _triplet_zero_state(
    basis: np.ndarray,
    spins: np.ndarray,
    index_1: int,
    index_2: int
) -> np.ndarray | sp.csc_array:
    """
    Generates the triplet zero state between two spin-1/2 nuclei. Unit state is
    assigned to the other spins.

    Parameters
    ----------
    basis : ndarray
        A 2-dimensional array containing the basis set that contains sequences
        of integers describing the Kronecker products of irreducible spherical
        tensors.
    spins : ndarray
        A 1-dimensional array specifying the spin quantum numbers of the system.
    index_1 : int
        Index of the first spin in the triplet zero state.
    index_2 : int
        Index of the second spin in the triplet zero state.

    Returns
    -------
    rho : ndarray or csc_array
        State vector corresponding to the triplet zero state.
    """

    # Calculate the dimension of the full Liouville space
    mults = (2*spins+1).astype(int)
    dim = np.prod(mults)

    # Get states
    E = _unit_state(basis, spins, normalized=False)
    IzIz = state_from_string(basis, spins, f"I(z,{index_1}) * I(z, {index_2})")
    IpIm = state_from_string(basis, spins, f"I(+,{index_1}) * I(-, {index_2})")
    ImIp = state_from_string(basis, spins, f"I(-,{index_1}) * I(+, {index_2})")

    # Make the triplet zero
    rho = 1 / dim * E - 4 / dim * IzIz + 2 / dim * (IpIm + ImIp)

    return rho

def _triplet_plus_state(
    basis: np.ndarray,
    spins: np.ndarray,
    index_1: int,
    index_2: int
) -> np.ndarray | sp.csc_array:
    """
    Generates the triplet plus state between two spin-1/2 nuclei. Unit state is
    assigned to the other spins.

    Parameters
    ----------
    basis : ndarray
        A 2-dimensional array containing the basis set that contains sequences
        of integers describing the Kronecker products of irreducible spherical
        tensors.
    spins : ndarray
        A 1-dimensional array specifying the spin quantum numbers of the system.
    index_1 : int
        Index of the first spin in the triplet plus state.
    index_2 : int
        Index of the second spin in the triplet plus state.

    Returns
    -------
    rho : ndarray or csc_array
        State vector corresponding to the triplet plus state.
    """

    # Calculate the dimension of the full Liouville space
    mults = (2*spins+1).astype(int)
    dim = np.prod(mults)

    # Get states
    E = _unit_state(basis, spins, normalized=False)
    IzE = state_from_string(basis, spins, f"I(z, {index_1})")
    EIz = state_from_string(basis, spins, f"I(z, {index_2})")
    IzIz = state_from_string(basis, spins, f"I(z,{index_1}) * I(z, {index_2})")

    # Make the triplet plus
    rho = 1 / dim * E + 2 / dim * IzE + 2 / dim * EIz + 4 / dim * IzIz

    return rho

def _triplet_minus_state(
    basis: np.ndarray,
    spins: np.ndarray,
    index_1: int,
    index_2: int
) -> np.ndarray | sp.csc_array:
    """
    Generates the triplet minus state between two spin-1/2 nuclei. Unit state is
    assigned to the other spins.

    Parameters
    ----------
    basis : ndarray
        A 2-dimensional array containing the basis set that contains sequences
        of integers describing the Kronecker products of irreducible spherical
        tensors.
    spins : ndarray
        A 1-dimensional array specifying the spin quantum numbers of the system.
    index_1 : int
        Index of the first spin in the triplet minus state.
    index_2 : int
        Index of the second spin in the triplet minus state.

    Returns
    -------
    rho : ndarray or csc_array
        State vector corresponding to the triplet minus state.
    """

    # Calculate the dimension of the full Liouville space
    mults = (2*spins+1).astype(int)
    dim = np.prod(mults)

    # Get states
    E = _unit_state(basis, spins, normalized=False)
    IzE = state_from_string(basis, spins, f"I(z, {index_1})")
    EIz = state_from_string(basis, spins, f"I(z, {index_2})")
    IzIz = state_from_string(basis, spins, f"I(z,{index_1}) * I(z, {index_2})")

    # Make the triplet minus
    rho = 1 / dim * E - 2 / dim * IzE - 2 / dim * EIz + 4 / dim * IzIz

    return rho

def _measure(
    basis: np.ndarray,
    spins: np.ndarray,
    rho: np.ndarray | sp.csc_array,
    operator: str
) -> complex:
    """
    Computes the expectation value of the specified operator for a given state
    vector. Assumes that the state vector `rho` represents a trace-normalized
    density matrix.

    Parameters
    ----------
    basis : ndarray
        A 2-dimensional array containing the basis set that contains sequences
        of integers describing the Kronecker products of irreducible spherical
        tensors.
    spins : ndarray
        A 1-dimensional array specifying the spin quantum numbers of the system.
    rho : ndarray or csc_array
        State vector that describes the density matrix.
    operator : str
        Defines the operator to be measured. The operator string must follow the
        rules below:

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
    ex : complex
        Expectation value.
    """

    # Get the "operator" to be measured
    oper = state_from_string(basis, spins, operator)

    # Perform the measurement
    ex = (oper.conj().T @ rho).trace()

    return ex

def state_to_truncated_basis(
    index_map: list,
    rho: np.ndarray | sp.csc_array
) -> np.ndarray | sp.csc_array:
    """
    Transforms a state vector to a truncated basis using the `index_map`,
    which contains indices that determine the elements that are retained
    after the transformation.

    Parameters
    ----------
    index_map : list
        Index mapping from the original basis to the truncated basis.
    rho : ndarray or csc_array
        State vector to be transformed.

    Returns
    -------
    rho_transformed : ndarray or csc_array
        State vector transformed into the truncated basis.
    """

    print("Transforming the state vector into the truncated basis.")
    time_start = time.time()

    # Perform the transformation to truncated basis
    rho_transformed = rho[index_map]

    print("Transformation completed.")
    print(f"Elapsed time: {time.time() - time_start:.4f} seconds.")
    print()

    return rho_transformed

def unit_state(
    spin_system: SpinSystem,
    normalized: bool=True
) -> np.ndarray | sp.csc_array:
    """
    Returns a unit state vector, which represents the identity operator. The
    output can be either normalised (trace equal to one) or unnormalised (raw
    identity matrix).

    Parameters
    ----------
    spin_system : SpinSystem
        Spin system to which the unit state is created.
    normalized : bool, default=True
        If set to True, the function will return a state vector that represents
        the trace-normalized density matrix. If False, returns a state vector
        that corresponds to the identity operator.

    Returns
    -------
    rho : ndarray or csc_array
        State vector corresponding to the unit state.
    """
    # Create the unit state
    rho = _unit_state(
        basis = spin_system.basis.basis,
        spins = spin_system.spins,
        normalized = normalized
    )

    return rho

def state(
    spin_system: SpinSystem,
    operator: str
) -> np.ndarray | sp.csc_array:
    """
    This function returns a column vector representing the density matrix as a
    linear combination of spin operators. Each element of the vector corresponds
    to the coefficient of a specific spin operator in the expansion.
    
    Normalization:
    The output of this function uses a normalised basis built from normalised
    products of single-spin spherical tensor operators. However, the
    coefficients are scaled so that the resulting linear combination represents
    the non-normalised version of the requested operator.

    NOTE: This function is sometimes called often and is cached for high
    performance.

    Parameters
    ----------
    spin_system : SpinSystem
        Spin system for which the state is created.
    operator : str
        Defines the state to be generated. The operator string must follow the
        rules below:

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
    rho : ndarray or csc_array
        State vector corresponding to the requested state.
    """
    # Check that the required attributes are set
    if spin_system.basis.basis is None:
        raise ValueError("Please build the basis before constructing a state.")
    
    # Build the state
    rho = state_from_string(
        basis = spin_system.basis.basis,
        spins = spin_system.spins,
        operator = operator
    )

    return rho

def state_to_zeeman(
    spin_system: SpinSystem,
    rho: np.ndarray | sp.csc_array
) -> np.ndarray | sp.csc_array:
    """
    Takes the state vector defined in the normalized spherical tensor basis
    and converts it into the Zeeman eigenbasis. Useful for error checking.

    Parameters
    ----------
    spin_system : SpinSystem
        Spin system whose state vector is going to be converted into a density
        matrix.
    rho : ndarray or csc_array
        State vector defined in the normalized spherical tensor basis.

    Returns
    -------
    rho_zeeman : ndarray or csc_array
        Spin density matrix defined in the Zeeman eigenbasis.
    """
    # Check that the required attributes are set
    if spin_system.basis.basis is None:
        raise ValueError("Please build the basis before converting the "
                         "state vector into density matrix.")
    
    # Convert the state vector into density matrix
    rho_zeeman = _state_to_zeeman(
        basis = spin_system.basis.basis,
        spins = spin_system.spins,
        rho = rho
    )
    
    return rho_zeeman

def alpha_state(
    spin_system: SpinSystem,
    index: int
) -> np.ndarray | sp.csc_array:
    """
    Generates the alpha state for a given spin-1/2 nucleus. Unit state is
    assigned to the other spins.

    Parameters
    ----------
    spin_system: SpinSystem
        Spin system for which the alpha state is created.
    index : int
        Index of the spin that has the alpha state.

    Returns
    -------
    rho : ndarray or csc_array
        State vector corresponding to the alpha state of the given spin index.
    """
    # Check that the required attributes are set
    if spin_system.basis.basis is None:
        raise ValueError("Please build the basis before constructing the "
                         "alpha state.")
    
    # Create the alpha state
    rho = _alpha_state(
        basis = spin_system.basis.basis,
        spins = spin_system.spins,
        index = index
    )

    return rho

def beta_state(
    spin_system: SpinSystem,
    index: int
) -> np.ndarray | sp.csc_array:
    """
    Generates the beta state for a given spin-1/2 nucleus. Unit state is
    assigned to the other spins.

    Parameters
    ----------
    spin_system: SpinSystem
        Spin system for which the beta state is created.
    index : int
        Index of the spin that has the beta state.

    Returns
    -------
    rho : ndarray or csc_array
        State vector corresponding to the beta state of the given spin index.
    """
    # Check that the required attributes are set
    if spin_system.basis.basis is None:
        raise ValueError("Please build the basis before constructing the "
                         "beta state.")
    
    # Create the beta state
    rho = _beta_state(
        basis = spin_system.basis.basis,
        spins = spin_system.spins,
        index = index
    )

    return rho

def equilibrium_state(spin_system: SpinSystem) -> np.ndarray | sp.csc_array:
    """
    Creates the thermal equilibrium state for the spin system.

    Parameters
    ----------
    spin_system : SpinSystem
        Spin system for which the equilibrium state is going to be created.

    Returns
    -------
    rho : ndarray or csc_array
        Equilibrium state vector.
    """
    # Check that the required attributes are set
    if spin_system.basis.basis is None:
        raise ValueError("Please build the basis before constructing the "
                         "equilibrium state.")
    if parameters.magnetic_field is None:
        raise ValueError("Please set the magnetic field before "
                         "constructing the equilibrium state.")
    if parameters.temperature is None:
        raise ValueError("Please set the temperature before "
                         "constructing the equilibrium state.")

    # Build the left Hamiltonian superoperator
    with HidePrints():
        H_left = sop_H(
            basis = spin_system.basis.basis,
            spins = spin_system.spins,
            gammas = spin_system.gammas,
            B = parameters.magnetic_field,
            chemical_shifts = spin_system.chemical_shifts,
            J_couplings = spin_system.J_couplings,
            side = "left",
            zero_value = parameters.zero_hamiltonian
        )

    # Build the equilibrium state
    rho = _equilibrium_state(
        basis = spin_system.basis.basis,
        spins = spin_system.spins,
        H_left = H_left,
        T = parameters.temperature,
        zero_value = parameters.zero_equilibrium
    )

    return rho

def singlet_state(
    spin_system: SpinSystem,
    index_1 : int,
    index_2 : int
) -> np.ndarray | sp.csc_array:
    """
    Generates the singlet state between two spin-1/2 nuclei. Unit state is
    assigned to the other spins.

    Parameters
    ----------
    spin_system : SpinSystem
        Spin system for which the singlet state is created.
    index_1 : int
        Index of the first spin in the singlet state.
    index_2 : int
        Index of the second spin in the singlet state.

    Returns
    -------
    rho : ndarray or csc_array
        State vector corresponding to the singlet state.
    """
    # Check that the required attributes are set
    if spin_system.basis.basis is None:
        raise ValueError("Please build the basis before constructing the "
                         "singlet state.")

    # Build the singlet state
    rho = _singlet_state(
        basis = spin_system.basis.basis,
        spins = spin_system.spins,
        index_1 = index_1,
        index_2 = index_2
    )

    return rho

def triplet_zero_state(
    spin_system: SpinSystem,
    index_1: int,
    index_2: int
) -> np.ndarray | sp.csc_array:
    """
    Generates the triplet zero state between two spin-1/2 nuclei. Unit state is
    assigned to the other spins.

    Parameters
    ----------
    spin_system: SpinSystem
        Spin system for which the triplet zero state is created.
    index_1 : int
        Index of the first spin in the triplet zero state.
    index_2 : int
        Index of the second spin in the triplet zero state.

    Returns
    -------
    rho : ndarray or csc_array
        State vector corresponding to the triplet zero state.
    """
    # Check that the required attributes are set
    if spin_system.basis.basis is None:
        raise ValueError("Please build the basis before constructing the "
                         "triplet zero state.")
    
    # Make the triplet zero state
    rho = _triplet_zero_state(
        basis = spin_system.basis.basis,
        spins = spin_system.spins,
        index_1 = index_1,
        index_2 = index_2
    )

    return rho

def triplet_plus_state(
    spin_system: SpinSystem,
    index_1: int,
    index_2: int
) -> np.ndarray | sp.csc_array:
    """
    Generates the triplet plus state between two spin-1/2 nuclei. Unit state is
    assigned to the other spins.

    Parameters
    ----------
    spin_system : SpinSystem
        Spin system for which the triplet plus state is created.
    index_1 : int
        Index of the first spin in the triplet plus state.
    index_2 : int
        Index of the second spin in the triplet plus state.

    Returns
    -------
    rho : ndarray or csc_array
        State vector corresponding to the triplet plus state.
    """
    # Check that the required attributes are set
    if spin_system.basis.basis is None:
        raise ValueError("Please build the basis before constructing the "
                         "triplet plus state.")
    
    # Create the triplet plus state
    rho = _triplet_plus_state(
        basis = spin_system.basis.basis,
        spins = spin_system.spins,
        index_1 = index_1,
        index_2 = index_2
    )

    return rho

def triplet_minus_state(
    spin_system: SpinSystem,
    index_1: int,
    index_2: int
) -> np.ndarray | sp.csc_array:
    """
    Generates the triplet minus state between two spin-1/2 nuclei. Unit state is
    assigned to the other spins.

    Parameters
    ----------
    spin_system : SpinSystem
        Spin system for which the triplet minus state is created.
    index_1 : int
        Index of the first spin in the triplet minus state.
    index_2 : int
        Index of the second spin in the triplet minus state.

    Returns
    -------
    rho : ndarray or csc_array
        State vector corresponding to the triplet minus state.
    """
    # Check that the required attributes are set
    if spin_system.basis.basis is None:
        raise ValueError("Please build the basis before constructing the "
                         "triplet minus state.")
    
    # Create the triplet minus state
    rho = _triplet_minus_state(
        basis = spin_system.basis.basis,
        spins = spin_system.spins,
        index_1 = index_1,
        index_2 = index_2
    )

    return rho

def measure(
    spin_system: SpinSystem,
    rho: np.ndarray | sp.csc_array,
    operator: str
) -> complex:
    """
    Computes the expectation value of the specified operator for a given state
    vector. Assumes that the state vector `rho` represents a trace-normalized
    density matrix.

    Parameters
    ----------
    spin_system : SpinSystem
        Spin system whose state is going to be measured.
    rho : ndarray or csc_array
        State vector that describes the density matrix.
    operator : str
        Defines the operator to be measured. The operator string must follow the
        rules below:

        - Cartesian and ladder operators: `I(component,index)` or
          `I(component)`. Examples:

            - `I(x,4)` --> Creates x-operator for spin at index 4.
            - `I(x)` --> Creates x-operator for all spins.

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
    ex : complex
        Expectation value.
    """
    # Check that the required attributes are set
    if spin_system.basis.basis is None:
        raise ValueError("Please build the basis before measuring an "
                         "expectation value of an operator.")
    
    # Perform the measurement
    ex = _measure(
        basis = spin_system.basis.basis,
        spins = spin_system.spins,
        rho = rho,
        operator = operator
    )

    return ex

def clear_cache_state_from_string():
    """
    Clears the cache of the `_state_from_string()` function.
    """
    # Clear the cache
    _state_from_string.cache_clear()

def clear_cache_state_from_op_def():
    """
    Clears the cache of the `_state_from_op_def()` function.
    """
    # Clear the cache
    _state_from_op_def.cache_clear()