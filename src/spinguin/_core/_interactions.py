"""
This module provides functionality for calculating the NMR interactions.
"""

import numpy as np
import scipy.constants as const

def dd_constant(y1: float, y2: float) -> float:
    """
    Calculate the dipole-dipole coupling constant excluding the distance term.

    Parameters
    ----------
    y1 : float
        Gyromagnetic ratio of the first spin in units of rad/s/T.
    y2 : float
        Gyromagnetic ratio of the second spin in units of rad/s/T.

    Returns
    -------
    dd_const : float
        Dipole-dipole coupling constant in units of rad/s * m^3.
    """

    # Evaluate the gyromagnetic prefactor of the dipole-dipole coupling.
    dd_const = -const.mu_0 / (4 * np.pi) * y1 * y2 * const.hbar

    return dd_const

def dd_coupling_tensors(xyz: np.ndarray, gammas: np.ndarray) -> np.ndarray:
    """
    Calculate dipole-dipole coupling tensors between all spins.

    Parameters
    ----------
    xyz : ndarray
        A 2-dimensional array specifying the cartesian coordinates in
        the XYZ format for each nucleus in the spin system. Must be
        given in the units of Å.
    gammas : ndarray
        A 1-dimensional array specifying the gyromagnetic ratios for
        each nucleus in the spin system. Must be given in the units
        of rad/s/T.

    Returns
    -------
    dd_tensors : ndarray
        Array of dimensions (N, N, 3, 3) containing the 3x3 tensors
        between all nuclei.
    """

    # Determine the number of spins in the system.
    nspins = gammas.shape[0]

    # Convert the Cartesian coordinates from ångströms to metres.
    xyz = xyz * 1e-10

    # Build the connector and distance arrays for all spin pairs.
    connectors = xyz[:, np.newaxis] - xyz
    distances = np.linalg.norm(connectors, axis=2)

    # Allocate the full array of dipole-dipole interaction tensors.
    dd_tensors = np.zeros((nspins, nspins, 3, 3))

    # Fill the lower-triangular spin-pair tensors.
    for i in range(nspins):
        for j in range(nspins):

            # Evaluate only the lower-triangular part to avoid duplication.
            if i > j:
                rr = np.outer(connectors[i, j], connectors[i, j])
                dd_tensors[i, j] = (
                    dd_constant(gammas[i], gammas[j])
                    * (3 * rr - distances[i, j] ** 2 * np.eye(3))
                    / distances[i, j] ** 5
                )

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
        Q_const = -const.e * Q_moment / const.hbar / (2 * S * (2 * S - 1))
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