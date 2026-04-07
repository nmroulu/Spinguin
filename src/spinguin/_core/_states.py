"""
State-vector utilities for Liouville-space spin dynamics.

This module provides helper functions for constructing Liouville-space state
vectors, converting them between representations, and evaluating expectation
values.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from spinguin._core._spin_system import SpinSystem

import numpy as np
import scipy.constants as const
import scipy.sparse as sp

from spinguin._core._hamiltonian import hamiltonian
from spinguin._core._hide_prints import HidePrints
from spinguin._core._la import expm
from spinguin._core._operators import op_prod
from spinguin._core._parameters import parameters
from spinguin._core._utils import parse_operator_string

def _require_basis(
    spin_system: SpinSystem,
    action: str,
) -> None:
    """
    Ensure that the basis has been built before state manipulation.

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

    Raises
    ------
    ValueError
        Raised if the basis has not yet been built.
    """

    # Reject operations that require a built basis when none is present.
    if spin_system.basis.basis is None:
        raise ValueError(f"Please build the basis before {action}.")


def _allocate_state_vector(
    dim: int,
) -> np.ndarray | sp.lil_array:
    """
    Allocate a mutable state vector in the configured storage format.

    Parameters
    ----------
    dim : int
        Length of the Liouville-space state vector.

    Returns
    -------
    ndarray or lil_array
        Mutable state vector initialised to zero.
    """

    # Allocate sparse or dense storage according to the global settings.
    if parameters.sparse_state:
        return sp.lil_array((dim, 1), dtype=complex)

    return np.zeros((dim, 1), dtype=complex)


def _finalise_state_vector(
    rho: np.ndarray | sp.lil_array | sp.csc_array,
) -> np.ndarray | sp.csc_array:
    """
    Return a state vector in the configured output storage format.

    Parameters
    ----------
    rho : ndarray or sparse array
        State vector assembled in mutable storage.

    Returns
    -------
    ndarray or csc_array
        Finalised state vector.
    """

    # Convert sparse work arrays to CSC format when sparse storage is used.
    if parameters.sparse_state and hasattr(rho, "tocsc"):
        return rho.tocsc()

    return rho


def _frobenius_norm(
    operator: np.ndarray | sp.csc_array,
) -> float:
    """
    Compute the Frobenius norm in the configured operator format.

    Parameters
    ----------
    operator : ndarray or csc_array
        Operator whose norm is required.

    Returns
    -------
    float
        Frobenius norm of the operator.
    """

    # Dispatch to the sparse or dense norm implementation as appropriate.
    if parameters.sparse_operator:
        return sp.linalg.norm(operator, ord="fro")

    return np.linalg.norm(operator, ord="fro")


def _liouville_dimension(
    spin_system: SpinSystem,
) -> int:
    """
    Return the Hilbert-space dimension of the spin system.

    Parameters
    ----------
    spin_system : SpinSystem
        Spin system whose total multiplicity product is required.

    Returns
    -------
    int
        Hilbert-space dimension of the spin system.
    """

    # Multiply the single-spin multiplicities to obtain the full dimension.
    return int(np.prod(spin_system.mults))


def _pair_longitudinal_states(
    spin_system: SpinSystem,
    index_1: int,
    index_2: int,
) -> tuple[np.ndarray | sp.csc_array, np.ndarray | sp.csc_array,
           np.ndarray | sp.csc_array]:
    """
    Construct the longitudinal basis states for a two-spin pair.

    Parameters
    ----------
    spin_system : SpinSystem
        Spin system in which the states are constructed.
    index_1 : int
        Index of the first spin.
    index_2 : int
        Index of the second spin.

    Returns
    -------
    tuple of ndarray or csc_array
        Unit state, first-spin ``Iz``, and second-spin ``Iz`` contributions.
    """

    # Build the unit and single-spin longitudinal states.
    unit = unit_state(spin_system, normalized=False)
    Iz_1 = state(spin_system, f"I(z, {index_1})")
    Iz_2 = state(spin_system, f"I(z, {index_2})")

    return unit, Iz_1, Iz_2


def _pair_correlation_states(
    spin_system: SpinSystem,
    index_1: int,
    index_2: int,
) -> tuple[
    np.ndarray | sp.csc_array,
    np.ndarray | sp.csc_array,
    np.ndarray | sp.csc_array,
    np.ndarray | sp.csc_array,
]:
    """
    Construct the pair-correlation states for a two-spin manifold.

    Parameters
    ----------
    spin_system : SpinSystem
        Spin system in which the states are constructed.
    index_1 : int
        Index of the first spin.
    index_2 : int
        Index of the second spin.

    Returns
    -------
    tuple of ndarray or csc_array
        Unit state, ``IzIz``, ``IpIm``, and ``ImIp`` contributions.
    """

    # Build the unit and two-spin correlation states.
    unit = unit_state(spin_system, normalized=False)
    IzIz = state(spin_system, f"I(z, {index_1}) * I(z, {index_2})")
    IpIm = state(spin_system, f"I(+, {index_1}) * I(-, {index_2})")
    ImIp = state(spin_system, f"I(-, {index_1}) * I(+, {index_2})")

    return unit, IzIz, IpIm, ImIp


def state_to_truncated_basis(
    index_map: list[int],
    rho: np.ndarray | sp.csc_array,
) -> np.ndarray | sp.csc_array:
    """
    Transform a state vector to a truncated basis.

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

    # Retain only those coefficients selected by the truncation map.
    return rho[index_map]


def unit_state(
    spin_system: SpinSystem,
    normalized: bool=True,
) -> np.ndarray | sp.csc_array:
    """
    Return the Liouville-space representation of the unit operator.

    The output can be either normalised to unit trace or left in the raw
    unnormalised form.

    Parameters
    ----------
    spin_system : SpinSystem
        Spin system for which the unit state is created.
    normalized : bool, default=True
        If set to ``True``, the function returns a state vector that
        represents the trace-normalised density matrix. If ``False``, it
        returns a state vector corresponding to the identity operator.

    Returns
    -------
    rho : ndarray or csc_array
        State vector corresponding to the unit state.
    """

    # Ensure that the basis is available.
    _require_basis(spin_system, "constructing the unit state")

    # Allocate the output state vector.
    rho = _allocate_state_vector(spin_system.basis.dim)

    # Set the identity-operator coefficient with the requested normalisation.
    if normalized:
        rho[0, 0] = 1 / np.sqrt(np.prod(spin_system.mults))
    else:
        rho[0, 0] = np.sqrt(np.prod(spin_system.mults))

    # Return the state in the configured output format.
    return _finalise_state_vector(rho)


def state(
    spin_system: SpinSystem,
    operator: str,
) -> np.ndarray | sp.csc_array:
    """
    Construct a state vector from an operator-string specification.

    The output uses the normalised spherical-tensor basis of the spin system,
    but the coefficients are scaled so that the resulting linear combination
    corresponds to the unnormalised version of the requested operator.

    Parameters
    ----------
    spin_system : SpinSystem
        Spin system for which the state is created.
    operator : str
        Defines the state to be generated. The operator string must follow the
        rules below:

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

        Special case: An empty ``operator`` string is considered as the unit
        operator.

        Whitespace will be ignored in the input.

        Note that indexing starts from 0.

    Returns
    -------
    rho : ndarray or csc_array
        State vector corresponding to the requested state.
    """

    # Ensure that the basis is available.
    _require_basis(spin_system, "constructing a state")

    # Allocate the output state vector.
    dim = spin_system.basis.dim
    rho = _allocate_state_vector(dim)

    # Parse the operator string into basis definitions and coefficients.
    op_defs, coeffs = parse_operator_string(operator, spin_system.nspins)

    # Map the operator definitions to basis indices.
    idxs = [spin_system.basis.indexof(op_def) for op_def in op_defs]

    # Populate the state vector one operator term at a time.
    for idx, coeff, op_def in zip(idxs, coeffs, op_defs):

        # Separate active and inactive spins in the product operator.
        op_array = np.asarray(op_def)
        idx_active = np.where(op_array != 0)[0]
        idx_inactive = np.where(op_array == 0)[0]

        # Compute the norm of the non-unit part of the operator.
        if len(idx_active) != 0:
            op_norm = _frobenius_norm(
                op_prod(op_def, spin_system.spins, include_unit=False)
            )

        # Use unit norm when the operator contains no active spin factors.
        else:
            op_norm = 1

        # Compute the norm contribution from inactive unit operators.
        unit_norm = np.sqrt(np.prod(spin_system.mults[idx_inactive]))

        # Combine the norm factors and store the coefficient.
        norm = op_norm * unit_norm
        rho[idx, 0] = coeff * norm

    # Return the state in the configured output format.
    return _finalise_state_vector(rho)


def state_to_zeeman(
    spin_system: SpinSystem,
    rho: np.ndarray | sp.csc_array,
) -> np.ndarray | sp.csc_array:
    """
    Convert a Liouville-space state vector to the Zeeman eigenbasis.

    This function is mainly intended for validation and debugging.

    Parameters
    ----------
    spin_system : SpinSystem
        Spin system whose state vector is going to be converted into a density
        matrix.
    rho : ndarray or csc_array
        State vector defined in the normalised spherical-tensor basis.

    Returns
    -------
    rho_zeeman : ndarray or csc_array
        Spin density matrix defined in the Zeeman eigenbasis.
    """

    # Ensure that the basis is available.
    _require_basis(
        spin_system,
        "converting the state vector into a density matrix",
    )

    # Determine the Hilbert-space dimension of the density matrix.
    dim = _liouville_dimension(spin_system)

    # Allocate the density matrix in the configured storage format.
    if parameters.sparse_operator:
        rho_zeeman = sp.csc_array((dim, dim), dtype=complex)
    else:
        rho_zeeman = np.zeros((dim, dim), dtype=complex)

    # Identify the non-zero coefficients of the Liouville-space state.
    idx_nonzero = rho.nonzero()[0]

    # Accumulate the corresponding normalised product operators.
    for idx in idx_nonzero:

        # Read the operator definition associated with the basis index.
        op_def = spin_system.basis.basis[idx]

        # Construct and normalise the corresponding Zeeman-basis operator.
        oper = op_prod(op_def, spin_system.spins, include_unit=True)
        oper = oper / _frobenius_norm(oper)

        # Add the weighted contribution to the density matrix.
        rho_zeeman += rho[idx, 0] * oper

    return rho_zeeman


def alpha_state(
    spin_system: SpinSystem,
    index: int,
) -> np.ndarray | sp.csc_array:
    """
    Generate the alpha state for a selected spin-1/2 nucleus.

    The remaining spins are assigned the unit state.

    Parameters
    ----------
    spin_system : SpinSystem
        Spin system for which the alpha state is created.
    index : int
        Index of the spin that has the alpha state.

    Returns
    -------
    rho : ndarray or csc_array
        State vector corresponding to the alpha state of the given spin index.
    """

    # Ensure that the basis is available.
    _require_basis(spin_system, "constructing the alpha state")

    # Determine the full Hilbert-space dimension.
    dim = _liouville_dimension(spin_system)

    # Build the unit and longitudinal single-spin states.
    unit = unit_state(spin_system, normalized=False)
    I_z = state(spin_system, f"I(z, {index})")

    # Form the alpha-state density operator.
    rho = 1 / dim * unit + 2 / dim * I_z

    return rho


def beta_state(
    spin_system: SpinSystem,
    index: int,
) -> np.ndarray | sp.csc_array:
    """
    Generate the beta state for a selected spin-1/2 nucleus.

    The remaining spins are assigned the unit state.

    Parameters
    ----------
    spin_system : SpinSystem
        Spin system for which the beta state is created.
    index : int
        Index of the spin that has the beta state.

    Returns
    -------
    rho : ndarray or csc_array
        State vector corresponding to the beta state of the given spin index.
    """

    # Ensure that the basis is available.
    _require_basis(spin_system, "constructing the beta state")

    # Determine the full Hilbert-space dimension.
    dim = _liouville_dimension(spin_system)

    # Build the unit and longitudinal single-spin states.
    unit = unit_state(spin_system, normalized=False)
    I_z = state(spin_system, f"I(z, {index})")

    # Form the beta-state density operator.
    rho = 1 / dim * unit - 2 / dim * I_z

    return rho


def equilibrium_state(
    spin_system: SpinSystem,
) -> np.ndarray | sp.csc_array:
    """
    Construct the thermal-equilibrium state of the spin system.

    Parameters
    ----------
    spin_system : SpinSystem
        Spin system for which the equilibrium state is going to be created.

    Returns
    -------
    rho : ndarray or csc_array
        Equilibrium state vector.
    """

    # Ensure that the basis is available.
    _require_basis(spin_system, "constructing the equilibrium state")

    # Ensure that the thermodynamic parameters have been configured.
    if parameters.magnetic_field is None:
        raise ValueError("Please set the magnetic field before "
                         "constructing the equilibrium state.")
    if parameters.temperature is None:
        raise ValueError("Please set the temperature before "
                         "constructing the equilibrium state.")

    # Build the Boltzmann propagator and apply it to the unit state.
    with HidePrints():
        H_left = hamiltonian(spin_system, side="left")

        # Evaluate the matrix exponential of the Boltzmann operator.
        T = parameters.temperature
        zv = parameters.zero_equilibrium
        P = expm(-const.hbar / (const.k * T) * H_left, zv)

        # Propagate the unnormalised unit state to equilibrium.
        unit = unit_state(spin_system, normalized=False)
        rho = P @ unit

        # Restore sparse storage if the matrix product densified the output.
        if parameters.sparse_state and not sp.issparse(rho):
            rho = sp.csc_array(rho)

        # Normalise the state so that the density-matrix trace equals one.
        rho = rho / (rho[0, 0] * np.sqrt(_liouville_dimension(spin_system)))

    return rho


def singlet_state(
    spin_system: SpinSystem,
    index_1: int,
    index_2: int,
) -> np.ndarray | sp.csc_array:
    """
    Generate the singlet state of a spin-1/2 pair.

    The remaining spins are assigned the unit state.

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

    # Ensure that the basis is available.
    _require_basis(spin_system, "constructing the singlet state")

    # Determine the full Hilbert-space dimension.
    dim = _liouville_dimension(spin_system)

    # Build the unit and two-spin correlation states.
    unit, IzIz, IpIm, ImIp = _pair_correlation_states(
        spin_system,
        index_1,
        index_2,
    )

    # Form the singlet-state density operator.
    rho = 1 / dim * unit - 4 / dim * IzIz - 2 / dim * (IpIm + ImIp)

    return rho


def triplet_zero_state(
    spin_system: SpinSystem,
    index_1: int,
    index_2: int,
) -> np.ndarray | sp.csc_array:
    """
    Generate the triplet-zero state of a spin-1/2 pair.

    The remaining spins are assigned the unit state.

    Parameters
    ----------
    spin_system : SpinSystem
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

    # Ensure that the basis is available.
    _require_basis(spin_system, "constructing the triplet zero state")

    # Determine the full Hilbert-space dimension.
    dim = _liouville_dimension(spin_system)

    # Build the unit and two-spin correlation states.
    unit, IzIz, IpIm, ImIp = _pair_correlation_states(
        spin_system,
        index_1,
        index_2,
    )

    # Form the triplet-zero density operator.
    rho = 1 / dim * unit - 4 / dim * IzIz + 2 / dim * (IpIm + ImIp)

    return rho


def triplet_plus_state(
    spin_system: SpinSystem,
    index_1: int,
    index_2: int,
) -> np.ndarray | sp.csc_array:
    """
    Generate the triplet-plus state of a spin-1/2 pair.

    The remaining spins are assigned the unit state.

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

    # Ensure that the basis is available.
    _require_basis(spin_system, "constructing the triplet plus state")

    # Determine the full Hilbert-space dimension.
    dim = _liouville_dimension(spin_system)

    # Build the unit, single-spin, and longitudinal correlation states.
    unit, Iz_1, Iz_2 = _pair_longitudinal_states(spin_system, index_1, index_2)
    IzIz = state(spin_system, f"I(z, {index_1}) * I(z, {index_2})")

    # Form the triplet-plus density operator.
    rho = 1 / dim * unit + 2 / dim * Iz_1 + 2 / dim * Iz_2 + 4 / dim * IzIz

    return rho


def triplet_minus_state(
    spin_system: SpinSystem,
    index_1: int,
    index_2: int,
) -> np.ndarray | sp.csc_array:
    """
    Generate the triplet-minus state of a spin-1/2 pair.

    The remaining spins are assigned the unit state.

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

    # Ensure that the basis is available.
    _require_basis(spin_system, "constructing the triplet minus state")

    # Determine the full Hilbert-space dimension.
    dim = _liouville_dimension(spin_system)

    # Build the unit, single-spin, and longitudinal correlation states.
    unit, Iz_1, Iz_2 = _pair_longitudinal_states(spin_system, index_1, index_2)
    IzIz = state(spin_system, f"I(z, {index_1}) * I(z, {index_2})")

    # Form the triplet-minus density operator.
    rho = 1 / dim * unit - 2 / dim * Iz_1 - 2 / dim * Iz_2 + 4 / dim * IzIz

    return rho


def measure(
    spin_system: SpinSystem,
    rho: np.ndarray | sp.csc_array,
    operator: str,
) -> complex:
    """
    Compute the expectation value of an operator for a given state.

    The state vector ``rho`` is assumed to represent a trace-normalised density
    matrix.

    Parameters
    ----------
    spin_system : SpinSystem
        Spin system whose state is going to be measured.
    rho : ndarray or csc_array
        State vector that describes the density matrix.
    operator : str
        Defines the operator to be measured. The operator string must follow the
        rules below:

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

        Special case: An empty ``operator`` string is considered as the unit
        operator.

        Whitespace will be ignored in the input.

        Note that indexing starts from 0.

    Returns
    -------
    ex : complex
        Expectation value.
    """

    # Ensure that the basis is available.
    _require_basis(spin_system, "measuring an expectation value of an operator")

    # Construct the operator state vector in the system basis.
    oper = state(spin_system, operator)

    # Evaluate the inner product corresponding to the expectation value.
    return (oper.conj().T @ rho).trace()