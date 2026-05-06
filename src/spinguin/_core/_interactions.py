"""
This module provides functionality for calculating the NMR interactions.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.constants import mu_0, hbar, e

from spinguin._core._validation import require

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

def Q_constant(S: float, Q_moment: float) -> float:
    """
    Calculate the nuclear quadrupolar coupling constant.

    Parameters
    ----------
    S : float
        Spin quantum number.
    Q_moment : float
        Nuclear quadrupole moment (in units of m^2).

    Returns
    -------
    Q_const : float
        Quadrupolar coupling constant in ``(rad/s) / (V/m^2)``.
    """

    # Evaluate the quadrupolar prefactor for spins with ``S >= 1``.
    if (S >= 1) and (Q_moment > 0):
        Q_const = -e * Q_moment / hbar / (2 * S * (2 * S - 1))
    else:
        Q_const = 0
    
    return Q_const

# TODO: Confirm the sign convention of the quadrupolar interaction.
def Q_intr_tensors(
    efg: np.ndarray,
    spins: np.ndarray,
    quad: np.ndarray,
) -> np.ndarray:
    """
    Calculate quadrupolar interaction tensors for a spin system.

    Parameters
    ----------
    efg : ndarray
        A 3-dimensional array specifying the electric field gradient tensors.
        Must be given in atomic units.
    spins : ndarray
        A 1-dimensional array specifying the spin quantum numbers for each
        spin.
    quad : ndarray
        A 1-dimensional array specifying the quadrupolar moments. Must be given
        in the units of m^2.
        
    Returns
    -------
    Q_tensors : ndarray
        Quadrupolar interaction tensors.
    """

    # Convert the electric field gradients from atomic units to ``V/m^2``.
    Q_tensors = -9.7173624292e21 * efg

    # Evaluate the quadrupolar coupling constants for all spins.
    Q_constants = [Q_constant(S, Q) for S, Q in zip(spins, quad)]

    # Scale each tensor by the corresponding quadrupolar coupling constant.
    for i, val in enumerate(Q_constants):
        Q_tensors[i] *= val

    return Q_tensors

def shielding_intr_tensors(
    shielding: np.ndarray,
    gammas: np.ndarray,
    B: float,
) -> np.ndarray:
    """
    Calculate shielding interaction tensors for relaxation calculations.

    Parameters
    ----------
    shielding : ndarray
        A 3-dimensional array specifying the nuclear shielding tensors for each
        nucleus. The tensors must be given in the units of ppm.
    gammas : ndarray
        A 1-dimensional array specifying the gyromagnetic ratios for
        each nucleus in the spin system. Must be given in the units
        of rad/s/T.
    B : float
        External magnetic field in units of T.

    Returns
    -------
    shielding_tensors : ndarray
        Array of shielding tensors.
    """

    # Convert the shielding tensors from ppm to dimensionless units.
    shielding_tensors = shielding * 1e-6

    # Construct the Larmor frequencies that scale the shielding tensors.
    w0s = -gammas * B

    # Scale each shielding tensor by the corresponding Larmor frequency.
    for i, val in enumerate(w0s):
        shielding_tensors[i] *= val

    return shielding_tensors