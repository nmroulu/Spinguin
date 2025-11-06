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
from functools import lru_cache
from spinguin._core._basis_indexing import parse_operator_string
from spinguin._core._config import config
from spinguin._core._hamiltonian import hamiltonian
from spinguin._core._hide_prints import HidePrints
from spinguin._core._la import expm
from spinguin._core._operators import operator_from_op_def
from spinguin._core._parameters import parameters

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
    
    # Calculate the dimension of the full Liouville space
    dim = np.prod(spin_system.mults)

    # Get states
    E = unit_state(spin_system, normalized=False)
    I_z = state(spin_system, f"I(z, {index})")

    # Make the alpha state
    rho = 1 / dim * E + 2 / dim * I_z

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
    
    # Calculate the dimension of the full Liouville space
    dim = np.prod(spin_system.mults)

    # Get states
    E = unit_state(spin_system, normalized=False)
    I_z = state(spin_system, f"I(z, {index})")

    # Make the beta state
    rho = 1 / dim * E - 2 / dim * I_z

    return rho

INTERACTIONDEFAULT = ["zeeman", "chemical_shift", "J_coupling"]
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
        H_left = hamiltonian(
            spin_system = spin_system,
            interactions = INTERACTIONDEFAULT,
            side = "left"
        )

    # Get the matrix exponential corresponding to the Boltzmann distribution
    with HidePrints():
        P = expm(-const.hbar / (const.k * parameters.temperature) * H_left,
                 config.zero_equilibrium)

    # Obtain the thermal equilibrium by propagating the unit state
    unit = unit_state(spin_system, normalized=False)
    rho = P @ unit

    # Normalize such that the trace of the corresponding density matrix is one
    rho = rho / (rho[0, 0] * np.sqrt(np.prod(spin_system.mults)))

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
    
    # Get the "operator" to be measured
    oper = state(spin_system, operator)

    # Perform the measurement
    ex = (oper.conj().T @ rho).trace()

    return ex


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

    # Calculate the dimension of the full Liouville space
    dim = np.prod(spin_system.mults)

    # Get states
    E = unit_state(spin_system, normalized=False)
    IzIz = state(spin_system, f"I(z,{index_1}) * I(z, {index_2})")
    IpIm = state(spin_system, f"I(+,{index_1}) * I(-, {index_2})")
    ImIp = state(spin_system, f"I(-,{index_1}) * I(+, {index_2})")

    # Make the singlet
    rho = 1 / dim * E - 4 / dim * IzIz - 2 / dim * (IpIm + ImIp)

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
    
    # Convert basis and spins suitable for hashing
    basis_bytes = spin_system.basis.basis.tobytes()
    spins_bytes = spin_system.spins.tobytes()
    
    # Build the state and ensure that a new copy is returned
    rho = _state_from_string(
        basis_bytes = basis_bytes,
        spins_bytes = spins_bytes,
        operator = operator,
        sparse = config.sparse_state
    ).copy()

    return rho

def state_from_op_def(
    basis : np.ndarray,
    spins : np.ndarray,
    op_def : np.ndarray,
    sparse : bool=False
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
    sparse : bool, default=False
        If False, returns a NumPy array. If True, returns a SciPy csc_array.

    Returns
    -------
    rho : ndarray or csc_array
        State vector corresponding to the requested state.
    """

    # Convert types suitable for hashing
    basis_bytes = basis.tobytes()
    spins_bytes = spins.tobytes()
    op_def_bytes = op_def.tobytes()

    # Ensure that a different instance is returned
    rho = _state_from_op_def(basis_bytes, spins_bytes, op_def_bytes,
                             sparse).copy()

    return rho

def state_vector_to_density_matrix(
    spin_system: SpinSystem,
    rho: np.ndarray | sp.csc_array
) -> np.ndarray | sp.csc_array:
    """
    Takes the state vector defined in the normalized spherical tensor basis
    and converts it into a density matrix defined in the Zeeman eigenbasis.
    Useful for error checking.

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

    # Get the dimension of density matrix
    dim = np.prod(spin_system.mults)

    # Initialize the spin density matrix
    if config.sparse_operator:
        rho_zeeman = sp.csc_array((dim, dim), dtype=complex)
    else:
        rho_zeeman = np.zeros((dim, dim), dtype=complex)
    
    # Obtain indices of the non-zero coefficients from the state vector
    idx_nonzero = rho.nonzero()[0]

    # Loop over the nonzero indices
    for idx in idx_nonzero:

        # Get the corresponding operator definition
        op_def = spin_system.basis.basis[idx]

        # Get the normalized product operator in the Zeeman eigenbasis with
        # normalization
        oper = op_prod(op_def, spin_system.spins, include_unit=True)
        if config.sparse_operator:
            oper = oper / sp.linalg.norm(oper, ord='fro')
        else:
            oper = oper / np.linalg.norm(oper, ord='fro')
        
        # Add to the total density matrix
        rho_zeeman += rho[idx, 0] * oper
    
    return rho_zeeman

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
    
    # Calculate the dimension of the full Liouville space
    dim = np.prod(spin_system.mults)

    # Get states
    E = unit_state(spin_system, normalized=False)
    IzE = state(spin_system, f"I(z, {index_1})")
    EIz = state(spin_system, f"I(z, {index_2})")
    IzIz = state(spin_system, f"I(z,{index_1}) * I(z, {index_2})")

    # Make the triplet minus
    rho = 1 / dim * E - 2 / dim * IzE - 2 / dim * EIz + 4 / dim * IzIz

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
    
    # Calculate the dimension of the full Liouville space
    dim = np.prod(spin_system.mults)

    # Get states
    E = unit_state(spin_system, normalized=False)
    IzE = state(spin_system, f"I(z, {index_1})")
    EIz = state(spin_system, f"I(z, {index_2})")
    IzIz = state(spin_system, f"I(z,{index_1}) * I(z, {index_2})")

    # Make the triplet plus
    rho = 1 / dim * E + 2 / dim * IzE + 2 / dim * EIz + 4 / dim * IzIz

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
    
    # Calculate the dimension of the full Liouville space
    dim = np.prod(spin_system.mults)

    # Get states
    E = unit_state(spin_system, normalized=False)
    IzIz = state(spin_system, f"I(z,{index_1}) * I(z, {index_2})")
    IpIm = state(spin_system, f"I(+,{index_1}) * I(-, {index_2})")
    ImIp = state(spin_system, f"I(-,{index_1}) * I(+, {index_2})")

    # Make the triplet zero
    rho = 1 / dim * E - 4 / dim * IzIz + 2 / dim * (IpIm + ImIp)

    return rho

def unit_state(
    spin_system: SpinSystem,
    normalized: bool=True
) -> np.ndarray | sp.csc_array:
    """
    Returns a unit state vector, which represents the identity operator. The
    output can be either normalised (trace equal to one) or unnormalised
    (identity matrix).

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
    # Obtain the basis dimension
    dim = spin_system.basis.dim

    # Acquire the spin multiplicities
    mults = spin_system.mults

    # Initialize the state vector
    if config.sparse_state:
        rho = sp.lil_array((dim, 1), dtype=complex)
    else:
        rho = np.zeros((dim, 1), dtype=complex)

    # Assign unit state coefficient
    if normalized:
        rho[0, 0] = 1 / np.sqrt(np.prod(mults))
    else:
        rho[0, 0] = np.sqrt(np.prod(mults))

    # Convert to csc_array if requesting sparse
    if config.sparse_state:
        rho = rho.tocsc()

    return rho

@lru_cache(maxsize=128)
def _state_from_string(
    basis_bytes: bytes,
    spins_bytes: bytes,
    operator: str,
    sparse: bool=False
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
            op = op_prod(op_def, spins, include_unit=False)
            if config.sparse_operator:
                op_norm = sp.linalg.norm(op, ord='fro')
            else:
                op_norm = np.linalg.norm(op, ord='fro')

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

@lru_cache(maxsize=8192)
def _state_from_op_def(
    basis_bytes : bytes,
    spins_bytes : bytes,
    op_def_bytes : bytes,
    sparse: bool=False
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
        # TODO: Benchmark sparse vs dense implementation
        op_norm = np.linalg.norm(
            op_prod(op_def, spins, include_unit=False, sparse=False),
            ord='fro')

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