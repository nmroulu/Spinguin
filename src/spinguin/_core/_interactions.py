"""
This module provides functionality for calculating the NMR interactions.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.constants import mu_0, hbar, e

from spinguin._core._validation import require
from spinguin._core._parameters import parameters

if TYPE_CHECKING:
    from spinguin._core._spin_system import SpinSystem

def dd_constants(spin_system: SpinSystem) -> np.ndarray:
    """
    Calculate the dipole-dipole coupling constants in rad/s.

    Parameters
    ----------
    spin_system : SpinSystem
        Spin system for which the dipole-dipole coupling constants are to be
        calculated.

    Returns
    -------
    np.ndarray
        Dipole-dipole coupling constants in rad/s.
    """
    # Ensure that the Cartesian coordinates are set
    require(spin_system, "xyz", "calculating dipole-dipole coupling constants")

    # Convert the molecular coordinates from Å to metres
    xyz = spin_system.xyz * 1e-10

    # Build the inter-spin vectors and distances
    connectors = xyz[:, np.newaxis] - xyz
    r = np.linalg.norm(connectors, axis=2)

    # Evaluate the pairwise dipole-dipole coupling constants in rad/s
    dd_couplings = np.zeros(shape=(spin_system.nspins, spin_system.nspins))
    y = spin_system.gammas
    for i in range(spin_system.nspins):
        for j in range(i):
            dd_couplings[i, j] = -mu_0*y[i]*y[j]*hbar / (4*np.pi*r[i, j]**3)

    return dd_couplings

def dd_coupling_tensors(spin_system: SpinSystem) -> np.ndarray:
    """
    Calculate dipole-dipole coupling tensors between all spins in rad/s.

    Parameters
    ----------
    spin_system : SpinSystem
        Spin system for which the dipole-dipole coupling tensors are to be
        calculated.

    Returns
    -------
    ndarray
        Array of dimensions (N, N, 3, 3) containing the 3x3 tensors
        between all nuclei. Only the lower-triangular part is populated.
    """
    # Fetch the dipole-dipole coupling constants
    b = dd_constants(spin_system)

    # Convert the Cartesian coordinates from ångströms to metres.
    xyz = spin_system.xyz * 1e-10

    # Build the connector and distance arrays for all spin pairs.
    connectors = xyz[:, np.newaxis] - xyz
    r = np.linalg.norm(connectors, axis=2)

    # Allocate the full array of dipole-dipole interaction tensors.
    dd_tensors = np.zeros((spin_system.nspins, spin_system.nspins, 3, 3))

    # Fill the lower-triangular spin-pair tensors.
    for i in range(spin_system.nspins):
        for j in range(i):
            rr = np.outer(connectors[i, j], connectors[i, j])
            dd_tensors[i, j] = b[i, j]*(3*rr/r[i, j]**2 - np.eye(3))

    return dd_tensors

def _Q_prefactor(S: float, Q_moment: float) -> float:
    """
    Calculate the prefactor in nuclear quadrupolar coupling tensors.

    Parameters
    ----------
    S : float
        Spin quantum number.
    Q_moment : float
        Nuclear quadrupole moment (in units of m^2).

    Returns
    -------
    float
        Prefactor in nuclear quadrupolar coupling
        tensors in ``(rad/s) / (V/m^2)``.
    """

    # Evaluate the quadrupolar prefactor for spins with ``S >= 1``.
    if (S >= 1) and (Q_moment > 0):
        Q_prefactor = -e * Q_moment / hbar / (2 * S * (2 * S - 1))
    else:
        Q_prefactor = 0
    
    return Q_prefactor

# TODO: Confirm the sign convention of the quadrupolar interaction.
def Q_intr_tensors(spin_system: SpinSystem) -> np.ndarray:
    """
    Calculate quadrupolar interaction tensors for a spin system.

    Parameters
    ----------
    spin_system: SpinSystem
        Spin system for which the quadrupolar interaction tensors are to be
        calculated.
        
    Returns
    -------
    ndarray
        Quadrupolar interaction tensors for each nucleus in the spin system.
    """

    # Convert the electric field gradients from atomic units to ``V/m^2``.
    efg = -9.7173624292e21 * spin_system.efg

    # Evaluate the quadrupolar coupling prefactors for all spins.
    spins = spin_system.spins
    quad = spin_system.quad
    Q_prefactors = [_Q_prefactor(S, Q) for S, Q in zip(spins, quad)]

    # Calculate the quadrupolar interaction tensors
    Q_tensors = np.zeros_like(efg)
    for i in range(spin_system.nspins):
        Q_tensors[i] = Q_prefactors[i] * efg[i]

    return Q_tensors

def shielding_intr_tensors(spin_system: SpinSystem) -> np.ndarray:
    """
    Calculate shielding interaction tensors for a spin system.

    Parameters
    ----------
    spin_system: SpinSystem
        Spin system for which the shielding interaction tensors are to be
        calculated.

    Returns
    -------
    ndarray
        Array of shielding tensors in rad/s.
    """
    # Ensure that the magnetic field is set
    require(
        parameters,
        "magnetic_field",
        "calculating shielding interaction tensors"
    )

    # Convert the shielding tensors from ppm to dimensionless units.
    shielding_tensors = spin_system.shielding * 1e-6

    # Construct the bare Larmor frequencies that scale the shielding tensors.
    w0s = resonance_frequencies(spin_system, cs=False)

    # Scale each shielding tensor by the corresponding Larmor frequency.
    for i, val in enumerate(w0s):
        shielding_tensors[i] *= val

    return shielding_tensors

def resonance_frequencies(
    spin_system: SpinSystem,
    zeeman: bool=True,
    cs: bool=True
) -> np.ndarray:
    """
    Calculate the resonance frequencies of all spins in a spin system.

    Parameters
    ----------
    spin_system: SpinSystem
        Spin system for which the resonance frequencies are to be calculated.

    Returns
    -------
    np.ndarray
        Resonance frequencies of all spins in the spin system in rad/s.
    """
    # Ensure that either the Zeeman or chemical-shift interactions are included.
    if not zeeman and not cs:
        raise ValueError(
            "Either the Zeeman or chemical-shift interaction must be included."
        )

    # Require that the magnetic field is set
    require(parameters, "magnetic_field", "calculating resonance frequencies")
    B = parameters.magnetic_field

    # Calculate the requested resonance frequencies
    omegas = np.zeros(spin_system.nspins)
    if zeeman:
        # Add contribution from Zeeman interaction
        omegas -= spin_system.gammas * B
    if cs:
        # Require that the chemical shifts are set
        require(
            spin_system,
            "chemical_shifts",
            "calculating resonance frequencies with chemical shifts"
        )
        # Add contribution from chemical shift
        omegas -= spin_system.gammas * B * spin_system.chemical_shifts * 1e-6

    return omegas
