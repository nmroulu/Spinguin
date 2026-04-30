"""
Relaxation-theory functions for Spinguin spin-dynamics simulations.

This module provides helper functions for constructing phenomenological and
Redfield relaxation superoperators together with supporting routines for
interaction tensors and different motional models. 
"""

from __future__ import annotations

import time
import warnings
from typing import TYPE_CHECKING, Literal

import numpy as np
import scipy.constants as const
import scipy.sparse as sp
from scipy.optimize import fsolve

from spinguin._core._hamiltonian import hamiltonian
from spinguin._core._hide_prints import HidePrints
from spinguin._core._la import (
    auxiliary_matrix_expm,
    cartesian_tensor_to_spherical_tensor,
    decompose_matrix,
    eliminate_small,
    expm,
    rotation_matrix_to_align_axes,
)
from spinguin._core._operators import op_Sx, op_Sy, op_Sz
from spinguin._core._parameters import parameters
from spinguin._core._status import status
from spinguin._core._superoperators import sop_T_coupled, superoperator
from spinguin._core._utils import idx_to_lq, parse_operator_string

if TYPE_CHECKING:
    from spinguin._core._spin_system import SpinSystem

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

def tau_c_l(tau_c: float | np.ndarray, l: int) -> float | np.ndarray:
    """
    Calculate the rank-dependent rotational correlation time from the
    "molecular tumbling" correlation time, which is commonly used in
    NMR literature but actually refers to the rank-2 correlation time.

    This function is used to convert the input molecular correlation
    time to the rank-dependent correlation times required for the relaxation
    superoperator construction. 

    The expression applies to anisotropic rotationally modulated interactions
    with ``l > 0``.

    Source: Eq. 70 from Hilla & Vaara: Rela2x: Analytic and automatic NMR
    relaxation theory
    https://doi.org/10.1016/j.jmr.2024.107828

    Parameters
    ----------
    tau_c : float or ndarray
        Rotational correlation time.
    l : int
        Interaction rank.

    Returns
    -------
    t_cl : float or ndarray
        Rotational correlation time for the given rank.

    Raises
    ------
    ValueError
        Raised if ``l`` is zero.
    """

    # Evaluate the rank-dependent correlation time for anisotropic motion.
    if l != 0:
        t_cl = 6 * tau_c / (l * (l + 1))

    # Reject the undefined isotropic ``l = 0`` case explicitly.
    else:
        raise ValueError("Rank l must be different from 0 in tau_c_l().")
    
    return t_cl
    
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

def rotational_diffusion_constant_SED(
    T: float,
    eta: float,
    r: float | np.ndarray,
) -> float | np.ndarray:
    """
    Calculate the rotational diffusion constant from the SED relation.

    Parameters
    ----------
    T : float
        Temperature in Kelvin.
    eta : float
        Solvent viscosity in Pa·s.
    r : float or ndarray
        Effective radius of the molecule in Ångströms (Å).

    Returns
    -------
    D_r : float or ndarray
        Rotational diffusion constant in units of 1/s.
    """
    # Convert the effective radius from ångströms to metres.
    r_meters = r * 1e-10

    # Evaluate the Stokes-Einstein-Debye rotational diffusion constant.
    D_r = const.k * T / (8 * np.pi * eta * r_meters**3)

    return D_r

def rotational_correlation_time(
    l: int,
    D_r: float | np.ndarray,
) -> float | np.ndarray:
    """
    Calculate the rotational correlation time for a given rank.

    Parameters
    ----------
    l : int
        Interaction rank.
    D_r : float or ndarray
        Rotational diffusion constant in units of 1/s.

    Returns
    -------
    tau_c : float or ndarray
        Rotational correlation time for the given rank in seconds.
    """
    # Evaluate the rotational correlation time for the given rank.
    tau_c = 1 / (l * (l + 1) * D_r)

    return tau_c

def rotational_correlation_time_SED(
    T: float,
    eta: float,
    r: float | np.ndarray,
    l: int,
) -> float | np.ndarray:
    """
    Calculate the rotational correlation time from the SED relation.

    Parameters
    ----------
    T : float
        Temperature in Kelvin.
    eta : float
        Solvent viscosity in Pa·s.
    r : float or ndarray
        Effective radius of the molecule in Ångströms (Å).
    l : int
        Rank of the tensor.

    Returns
    -------
    tau_c : float or ndarray
        Rotational correlation time for the given rank in seconds.
    """
    # Calculate the rotational diffusion constant using the SED relation.
    D_r = rotational_diffusion_constant_SED(T, eta, r)

    # Convert the diffusion constant to a rank-dependent correlation time.
    tau_c = rotational_correlation_time(l, D_r)

    return tau_c

def center_of_mass(
    masses: np.ndarray,
    coords: np.ndarray,
) -> np.ndarray:
    """
    Calculate the centre of mass of a molecule.

    Parameters
    ----------
    masses : ndarray
        A 1-dimensional array specifying the atomic masses of each atom
        in the molecule.
        Must be given in atomic mass units (u).
    coords : ndarray
        A 2-dimensional array specifying the cartesian coordinates in
        the XYZ format for each atom in the molecule.
        Must be given in the units of Å.

    Returns
    -------
    center_of_mass : ndarray
        Centre-of-mass coordinates in units of Å.
    """
    # Compute the total molecular mass.
    total_mass = np.sum(masses)

    # Evaluate the mass-weighted Cartesian mean position.
    return np.sum(masses[:, np.newaxis] * coords, axis=0) / total_mass

def moment_of_inertia_tensor(
    masses: np.ndarray,
    coords: np.ndarray,
) -> np.ndarray:
    """
    Calculate the moment of inertia tensor of a molecule.

    Parameters
    ----------
    masses : ndarray
        A 1-dimensional array specifying the atomic masses of each atom
        in the molecule.
        Must be given in atomic mass units (u).
    coords : ndarray
        A 2-dimensional array specifying the cartesian coordinates in
        the XYZ format for each atom in the molecule.
        Must be given in the units of Å.

    Returns
    -------
    I : ndarray
        Moment of inertia tensor in units of u·Å².
    """

    # Determine the number of atoms in the molecule.
    natoms = masses.shape[0]

    # Determine the centre of mass of the molecule.
    center_of_mass_coords = center_of_mass(masses, coords)

    # Shift the Cartesian coordinates to the centre-of-mass frame.
    coords_centered = coords - center_of_mass_coords

    # Allocate the moment of inertia tensor.
    I = np.zeros((3, 3))

    # Accumulate the Cartesian tensor elements atom by atom.
    for i in range(natoms):
        x, y, z = coords_centered[i]
        m = masses[i]

        # Update the tensor elements from the current atom.
        I[0, 0] += m * (y**2 + z**2)
        I[1, 1] += m * (x**2 + z**2)
        I[2, 2] += m * (x**2 + y**2)
        I[0, 1] -= m * x * y
        I[0, 2] -= m * x * z
        I[1, 2] -= m * y * z

    # Complete the symmetric lower-triangular tensor elements.
    I[1, 0] = I[0, 1]
    I[2, 0] = I[0, 2]
    I[2, 1] = I[1, 2]

    return I

def moment_of_inertia_equivalent_ellipsoid(
    masses: np.ndarray,
    coords: np.ndarray,
) -> np.ndarray:
    """
    Calculate the semi-axes of the mass-equivalent ellipsoid.

    The semi-axes are determined from the equations

        I_1 = (1/5) * M * (a_y^2 + a_z^2)
        I_2 = (1/5) * M * (a_x^2 + a_z^2)
        I_3 = (1/5) * M * (a_x^2 + a_y^2)

    where ``I_1``, ``I_2``, and ``I_3`` are the eigenvalues of the moment of
    inertia tensor, ``M`` is the total mass of the molecule, and ``a_x``,
    ``a_y``, and ``a_z`` are the semi-axes of the ellipsoid.

    Parameters
    ----------
    masses : ndarray
        A 1-dimensional array specifying the atomic masses of each atom
        in the molecule, given in atomic mass units (u).
    coords : ndarray
        A 2-dimensional array specifying the Cartesian coordinates in
        the XYZ format for each atom in the molecule, given in units of Å.

    Returns
    -------
    semi_axes : ndarray
        Semi-axes of the mass-equivalent ellipsoid in units of Å, calculated
        as described above.
    """
    # Calculate the moment of inertia tensor and its eigenvalues.
    I = moment_of_inertia_tensor(masses, coords)
    eigenvalues, _ = np.linalg.eigh(I)

    # Calculate the total molecular mass.
    total_mass = np.sum(masses)

    # Define the nonlinear system for the equivalent ellipsoid semi-axes.
    def equations(semi_axes: np.ndarray) -> list[float]:
        a_x, a_y, a_z = semi_axes
        eq1 = (1/5) * total_mass * (a_y**2 + a_z**2) - eigenvalues[0]
        eq2 = (1/5) * total_mass * (a_x**2 + a_z**2) - eigenvalues[1]
        eq3 = (1/5) * total_mass * (a_x**2 + a_y**2) - eigenvalues[2]
        return [eq1, eq2, eq3]

    # Build an initial guess from a sphere with the same largest inertia.
    initial_guess = np.sqrt(5/2 * eigenvalues[2] / total_mass) * np.ones(3)

    # TODO: Check this
    # Solve the nonlinear system while suppressing numerical warnings.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        a_x, a_y, a_z = fsolve(
            equations,
            initial_guess,
            maxfev=10000,
            xtol=1e-15,
        )

    return np.array([a_x, a_y, a_z])

def Perrin_integrals(
    a_x: float,
    a_y: float,
    a_z: float,
    num_points: int = int(1e6),
) -> tuple[float, float, float]:
    """
    Calculate the Perrin integrals for an ellipsoid.

    A variable substitution is used to map the infinite integration domain to
    the interval ``[0, 1)``.

    Parameters
    ----------
    a_x : float
        Semi-axis a_x of the ellipsoid in units of Å.
    a_y : float
        Semi-axis a_y of the ellipsoid in units of Å.
    a_z : float
        Semi-axis a_z of the ellipsoid in units of Å.
    num_points : int, optional
        Number of points for the integration. Default is 1e6.

    Returns
    -------
    P : float
        Perrin P integral for the ellipsoid.
    Q : float
        Perrin Q integral for the ellipsoid.
    R : float
        Perrin R integral for the ellipsoid.
    """
    # Map the semi-infinite integration range to the interval ``[0, 1)``.
    t = np.linspace(0, 1, num_points, endpoint=False)
    s = t / (1 - t)
    jacobian = 1 / (1 - t)**2

    # Evaluate the three Perrin integrands including the Jacobian factor.
    integrand_P = \
        1 / np.sqrt((a_x**2 + s) ** 3 * (a_y**2 + s) * (a_z**2 + s)) * jacobian
    integrand_Q = \
        1 / np.sqrt((a_y**2 + s) ** 3 * (a_z**2 + s) * (a_x**2 + s)) * jacobian
    integrand_R = \
        1 / np.sqrt((a_z**2 + s) ** 3 * (a_x**2 + s) * (a_y**2 + s)) * jacobian

    # Integrate the three transformed functions over the finite interval.
    P = np.trapz(integrand_P, t)
    Q = np.trapz(integrand_Q, t)
    R = np.trapz(integrand_R, t)

    return P, Q, R

def Perrin_factors(
    a_x: float,
    a_y: float,
    a_z: float,
) -> np.ndarray:
    """
    Calculate the Perrin factors for an ellipsoid.

    Parameters
    ----------
    a_x : float
        Semi-axis a_x of the ellipsoid in units of Å.
    a_y : float
        Semi-axis a_y of the ellipsoid in units of Å.
    a_z : float
        Semi-axis a_z of the ellipsoid in units of Å.

    Returns
    -------
    factors : ndarray
        Array containing the Perrin factors for the x, y, and z axes.
    """
    # Evaluate the Perrin integrals for the ellipsoid.
    P, Q, R = Perrin_integrals(a_x, a_y, a_z)

    # Convert the integrals to rotational friction factors along each axis.
    f_x = (a_y**2 + a_z**2) / (a_y**2 * Q + a_z**2 * R)
    f_y = (a_x**2 + a_z**2) / (a_z**2 * R + a_x**2 * P)
    f_z = (a_x**2 + a_y**2) / (a_x**2 * P + a_y**2 * Q)

    return np.array([f_x, f_y, f_z])

def rotational_diffusion_constants_Perrin(
    T: float,
    eta: float,
    a_x: float,
    a_y: float,
    a_z: float,
) -> np.ndarray:
    """
    Calculate Perrin rotational diffusion constants along the principal axes.

    Parameters
    ----------
    T : float
        Temperature in Kelvin.
    eta : float
        Solvent viscosity in Pa·s.
    a_x : float
        Semi-axis a_x of the ellipsoid in units of Å.
    a_y : float
        Semi-axis a_y of the ellipsoid in units of Å.
    a_z : float
        Semi-axis a_z of the ellipsoid in units of Å.

    Returns
    -------
    D_rs : ndarray
        Rotational diffusion constants along the principal axes in units of 1/s.
    """
    # Evaluate the Perrin factors for the ellipsoid.
    perrin_factors = Perrin_factors(a_x, a_y, a_z)

    # Convert the Perrin factors from ``Å^3`` to ``m^3``.
    perrin_factors *= 1e-30

    # Convert the Perrin factors to rotational diffusion constants.
    D_rs = (const.k * T / perrin_factors) / (16 * np.pi * eta / 3)

    return D_rs

def rotational_correlation_times_Perrin(
    masses: np.ndarray,
    coords: np.ndarray,
    T: float,
    eta: float,
    l: int,
) -> np.ndarray:
    """
    Calculate Perrin rotational correlation times for a given tensor rank.

    Parameters
    ----------
    masses : ndarray
        A 1-dimensional array specifying the atomic masses of each atom
        in the molecule. Must be given in atomic mass units (u).
    coords : ndarray
        A 2-dimensional array specifying the Cartesian coordinates in
        the XYZ format for each atom in the molecule. Must be given in
        the units of Å.
    T : float
        Temperature in Kelvin.
    eta : float
        Solvent viscosity in Pa·s.
    l : int
        Rank of the tensor (must be greater than 0).

    Returns
    -------
    tau_cs : ndarray
        Rotational correlation times along the principal axes in seconds.
        The output is an array of shape (3,) corresponding to the three
        principal axes of the ellipsoid.
    """
    # Calculate the semi-axes of the mass-equivalent ellipsoid.
    semi_axes = moment_of_inertia_equivalent_ellipsoid(masses, coords)

    # Add a minimum hydrodynamic thickness to avoid unrealistically thin axes.
    # TODO: Calibrate this
    semi_axes += 1.0

    # Calculate the principal-axis rotational diffusion constants.
    D_rs = rotational_diffusion_constants_Perrin(
        T,
        eta,
        semi_axes[0],
        semi_axes[1],
        semi_axes[2],
    )

    # Convert the diffusion constants to rank-dependent correlation times.
    tau_cs = rotational_correlation_time(l, D_rs)

    return tau_cs

def _rot_diff_gen(spin_system: SpinSystem) -> dict:
    """
    Generate the rotational diffusion generator for a spin system.

    Parameters
    ----------
    spin_system : SpinSystem
        Spin system for which the rotational diffusion generator is going to
        be generated.
    
    Returns
    -------
    G : dict
        Rotational diffusion generator for ranks l = 1 and l = 2 (keys). The
        values are 3x3 and 5x5 NumPy arrays, respectively.
    """
    # Report the start of the rotational-diffusion calculation.
    status("Calculating the rotational diffusion generator...")
    time_start = time.time()

    # Obtain the rotational correlation time specification.
    tau_c = spin_system.relaxation.tau_c
    
    # Initialise the dictionary that stores the generators by rank.
    G = {}

    # Build the rotational diffusion generator for each relevant rank.
    for l in [1, 2]:

        # Construct the rank-``l`` angular momentum operators.
        Jx = op_Sx(l)
        Jy = op_Sy(l)
        Jz = op_Sz(l)

        # Convert sparse operators to dense arrays for matrix products.
        if parameters.sparse_operator:
            Jx = Jx.toarray()
            Jy = Jy.toarray()
            Jz = Jz.toarray()

        # Convert the input correlation times to the current tensor rank.
        tau_cl = tau_c_l(tau_c, l)

        # Assemble the isotropic rotational diffusion generator.
        if np.isscalar(tau_c):
            D = 1 / (tau_cl * l * (l + 1))
            G[l] = D * (Jx @ Jx + Jy @ Jy + Jz @ Jz)

        # Assemble the symmetric-top rotational diffusion generator.
        elif len(tau_c) == 2:
            D1 = 1 / (tau_cl[0] * l * (l + 1))
            D2 = 1 / (tau_cl[1] * l * (l + 1))
            G[l] = D1 * Jz @ Jz + D2 * (Jx @ Jx + Jy @ Jy)

        # Assemble the fully anisotropic rotational diffusion generator.
        else:
            D1 = 1 / (tau_cl[0] * l * (l + 1))
            D2 = 1 / (tau_cl[1] * l * (l + 1))
            D3 = 1 / (tau_cl[2] * l * (l + 1))
            G[l] = D1 * Jx @ Jx + D2 * Jy @ Jy + D3 * Jz @ Jz

    # Report the completion of the rotational-diffusion calculation.
    status(f"Completed in {time.time() - time_start:.4f} seconds.\n")

    return G

def _process_interaction_tensor(
    V: np.ndarray,
    R: np.ndarray,
    dge: dict,
    anti: bool,
) -> dict:
    """
    Decompose an interaction tensor into rotational-diffusion components.

    The first key of the returned dictionary is the interaction rank ``l`` and
    the second key is the component ``p`` associated with the eigenvectors of
    the rotational diffusion generator. Ranks with negligible norm are omitted.

    Parameters
    ----------
    V : ndarray
        Cartesian interaction tensor.
    R : ndarray
        Rotation operator that rotates the interaction tensor to the rotational
        principal axis frame. For isotropic rotational diffusion, R is a unit
        matrix.
    dge : dict
        Eigenvectors of the rotational diffusion generator for ranks l = 1 and
        l = 2 (keys).
    anti : bool
        Defines whether to return antisymmetric part.

    Returns
    -------
    V_lp : dict
        Interaction tensor as a dictionary of dictionaries. The first keys are
        the interaction rank ``l``, and the second keys are the components
        ``p``.
    """
    # Initialise the dictionary that stores the rank-resolved tensor parts.
    V_lp = {}

    # Decompose the Cartesian tensor into antisymmetric and symmetric parts.
    _, V1, V2 = decompose_matrix(V)

    # Retain only tensor parts whose norm exceeds the interaction threshold.
    if anti and np.linalg.norm(V1, ord=2) > parameters.zero_interaction:
        V_lp[1] = {}
    if np.linalg.norm(V2, ord=2) > parameters.zero_interaction:
        V_lp[2] = {}

    # Return immediately if no tensor part survives the threshold.
    if len(V_lp) == 0:
        return V_lp

    # Rotate the tensor into the rotational principal-axis frame.
    V = R @ V @ R.T

    # Express the rotated tensor in spherical tensor notation.
    V = cartesian_tensor_to_spherical_tensor(V)

    # Project the tensor components onto the diffusion-generator eigenbasis.
    for l in V_lp.keys():
        for p in range(2 * l + 1):
            V_lp[l][p] = sum(
                V[l, m] * dge[l][l - m, p]
                for m in range(-l, l + 1)
            )

    return V_lp

def _process_interactions(spin_system: SpinSystem, dge: dict) -> dict:
    """
    Collect and rank-sort all relevant interaction tensors.

    Interaction tensors whose norm falls below the global threshold are
    disregarded.

    Parameters
    ----------
    spin_system : SpinSystem
        SpinSystem whose interactions are to be processed.
    dge : dict
        Eigenvectors of the rotational diffusion generator for ranks l = 1 and
        l = 2 (keys).

    Returns
    -------
    interactions : dict
        A dictionary where the interactions are organized by rank ``l``. The
        values are lists containing all interactions with meaningful
        strength. The interactions are tuples in the format::
        
            (interaction, spin_1, spin_2, tensor).
    """
    # Report the start of the interaction processing.
    status("Processing interactions for relaxation...")
    time_start = time.time()

    # Determine whether isotropic or anisotropic rotational diffusion is used.
    iso = np.isscalar(spin_system.relaxation.tau_c)

    # Build the rotation from the laboratory frame to the rotational axes.
    if iso:
        R = np.eye(3)
    else:
        masses = spin_system.relaxation.molecule.masses
        coords = spin_system.relaxation.molecule.xyz
        moi_tensor = moment_of_inertia_tensor(masses, coords)
        _, rot_principal_axes = np.linalg.eigh(moi_tensor)
        lab_frame = np.eye(3)
        R = rotation_matrix_to_align_axes(lab_frame, rot_principal_axes.T)

    # Initialise the dictionary of interactions grouped by tensor rank.
    interactions = {1: [], 2: []}

    # Process all dipole-dipole interaction tensors.
    if spin_system.xyz is not None:

        # Get the dipole-dipole coupling tensors.
        dd_tensors = dd_coupling_tensors(spin_system.xyz, spin_system.gammas)

        # Add all non-negligible dipole-dipole interaction components.
        for spin_1 in range(spin_system.nspins):
            for spin_2 in range(spin_1):
                V = dd_tensors[spin_1, spin_2]
                V = _process_interaction_tensor(V, R, dge, anti=False)
                for l in V.keys():
                    interactions[l].append(("DD", spin_1, spin_2, V[l]))

    # Process all shielding interaction tensors.
    if spin_system.shielding is not None:

        # Get the shielding interaction tensors.
        sh_tensors = shielding_intr_tensors(
            spin_system.shielding,
            spin_system.gammas,
            parameters.magnetic_field
        )

        # Choose whether the antisymmetric CSA contribution is retained.
        anti = spin_system.relaxation.antisymmetric
        
        # Add all non-negligible shielding interaction components.
        for spin_1 in range(spin_system.nspins):
            V = sh_tensors[spin_1]
            V = _process_interaction_tensor(V, R, dge, anti=anti)
            for l in V.keys():
                interactions[l].append(("CSA", spin_1, None, V[l]))

    # Process all quadrupolar interaction tensors.
    if spin_system.efg is not None:

        # Get the quadrupolar interaction tensors.
        q_tensors = Q_intr_tensors(
            spin_system.efg,
            spin_system.spins,
            spin_system.quad
        )

        # Add all non-negligible quadrupolar interaction components.
        for spin_1 in range(spin_system.nspins):
            if spin_system.spins[spin_1] > 1 / 2:
                V = q_tensors[spin_1]
                V = _process_interaction_tensor(V, R, dge, anti=False)
                for l in V.keys():
                    interactions[l].append(("Q", spin_1, None, V[l]))

    # Remove ranks for which no interaction survived the threshold.
    for l in [1, 2]:
        if len(interactions[l]) == 0:
            del interactions[l]

    # Report the completion of the interaction processing.
    status(f"Completed in {time.time() - time_start:.4f} seconds.\n")

    return interactions

def _get_sop_T(
    spin_system: SpinSystem,
    l: int,
    q: int,
    interaction_type: Literal["CSA", "Q", "DD"],
    spin_1: int,
    spin_2: int | None = None,
) -> np.ndarray | sp.csc_array:
    """
    Calculate the coupled spherical tensor superoperator for one interaction.

    Parameters
    ----------
    spin_system : SpinSystem
        SpinSystem object containing the basis and spins information.
    l : int
        Operator rank.
    q : int
        Operator projection.
    interaction_type : {'CSA', 'Q', 'DD'}
        Describes the interaction type. Possible options are "CSA", "Q", and
        "DD", which stand for chemical shift anisotropy, quadrupolar coupling,
        and dipole-dipole coupling, respectively.
    spin_1 : int
        Index of the first spin.
    spin_2 : int, optional
        Index of the second spin. Leave empty for single-spin interactions
        (e.g., CSA).

    Returns
    -------
    sop : ndarray or csc_array
        Coupled spherical tensor superoperator of rank ``l`` and projection
        ``q``.
    """

    # Construct the superoperator for single-spin linear interactions.
    if interaction_type == "CSA":
        sop = sop_T_coupled(spin_system, l, q, spin_1)

    # Construct the superoperator for single-spin quadratic interactions.
    elif interaction_type == "Q":
        sop = superoperator(spin_system, f"T({l},{q},{spin_1})")

    # Construct the superoperator for two-spin bilinear interactions.
    elif interaction_type == "DD":
        sop = sop_T_coupled(spin_system, l, q, spin_1, spin_2)

    # Reject unsupported interaction labels explicitly.
    else:
        raise ValueError(
            f"Invalid interaction type '{interaction_type}' for relaxation "
            "superoperator. Possible options are 'CSA', 'Q', and 'DD'."
        )

    return sop

# TODO
def sop_R_random_field() -> None:
    """
    Placeholder for a random-field relaxation superoperator.
    """

    # This functionality has not yet been implemented.
    return None

def _sop_R_phenomenological(
    basis: np.ndarray,
    R1: np.ndarray,
    R2: np.ndarray,
) -> np.ndarray | sp.csc_array:
    """
    Construct the phenomenological relaxation superoperator.

    Parameters
    ----------
    basis : ndarray
        A 2-dimensional array containing the basis set that contains sequences
        of integers describing the Kronecker products of irreducible spherical
        tensors.
    R1 : ndarray
        A one-dimensional array containing the longitudinal relaxation rates
        in 1/s for each spin. For example: ``np.array([1.0, 2.0, 2.5])``.
    R2 : ndarray
        A one-dimensional array containing the transverse relaxation rates
        in 1/s for each spin. For example: ``np.array([2.0, 4.0, 5.0])``.

    Returns
    -------
    sop_R : ndarray or csc_array
        Relaxation superoperator.
    """

    # Report the start of the phenomenological relaxation construction.
    time_start = time.time()
    status("Constructing the phenomenological relaxation superoperator...")

    # Determine the basis dimension.
    dim = basis.shape[0]

    # Allocate the diagonal relaxation superoperator.
    if parameters.sparse_superoperator:
        sop_R = sp.lil_array((dim, dim))
    else:
        sop_R = np.zeros((dim, dim))

    # Assign a longitudinal or transverse relaxation rate to each basis state.
    for idx, state in enumerate(basis):

        # Initialise the total relaxation rate of the current basis state.
        R_state = 0
        
        # Sum the spin-specific rate contributions for the current state.
        for spin, operator in enumerate(state):

            # Skip unit-operator entries in the basis-state description.
            if operator != 0:

                # Extract the tensor projection of the current basis element.
                _, q = idx_to_lq(operator)
            
                # Add a longitudinal contribution for ``q = 0``.
                if q == 0:
                    R_state += R1[spin]

                # Otherwise add a transverse contribution.
                else:
                    R_state += R2[spin]

        # Insert the total rate onto the diagonal element of the state.
        sop_R[idx, idx] = R_state

    # Convert the sparse work array to CSC format if needed.
    if parameters.sparse_superoperator:
        sop_R = sop_R.tocsc()

    # Report the completion of the phenomenological construction.
    status(f"Completed in {time.time() - time_start:.4f} seconds.\n")

    return sop_R

def _sop_R_sr2k(
    spin_system: SpinSystem,
    R: sp.csc_array,
) -> np.ndarray | sp.csc_array:
    """
    Calculate scalar relaxation of the second kind from Abragam's formula.

    Parameters
    ----------
    spin_system : SpinSystem
        SpinSystem for which the SR2K contribution is to be calculated.
    R : ndarray or csc_array
        Relaxation superoperator without scalar relaxation of the second kind.

    Returns
    -------
    R : ndarray or csc_array
        Relaxation superoperator containing only the contribution from scalar
        relaxation of the second kind.
    """

    # Report the start of the SR2K calculation.
    status("Processing scalar relaxation of the second kind...")
    time_start = time.time()

    # Build a lookup dictionary for rapid basis-state indexing.
    basis_lookup = {
        tuple(row): idx
        for idx, row in enumerate(spin_system.basis.basis)
    }

    # Initialise the target-spin longitudinal and transverse rates.
    R1 = np.zeros(spin_system.nspins)
    R2 = np.zeros(spin_system.nspins)

    # Identify all quadrupolar nuclei in the spin system.
    quadrupolar = [
        i
        for i, spin in enumerate(spin_system.spins)
        if spin > 0.5
    ]
    
    # Accumulate SR2K contributions from each quadrupolar nucleus.
    for quad in quadrupolar:

        # Find the operator definitions of the longitudinal and transverse
        # states
        op_def_z, _ = parse_operator_string(f"I(z, {quad})", spin_system.nspins)
        op_def_p, _ = parse_operator_string(f"I(+, {quad})", spin_system.nspins)

        # Convert operator definitions to tuples for dictionary lookup.
        op_def_z = tuple(op_def_z[0])
        op_def_p = tuple(op_def_p[0])

        # Locate the longitudinal and transverse basis states.
        idx_long = basis_lookup[op_def_z]
        idx_trans = basis_lookup[op_def_p]

        # Extract the effective longitudinal and transverse relaxation times.
        T1 = 1 / R[idx_long, idx_long]
        T2 = 1 / R[idx_trans, idx_trans]

        # Retain only the real components of the relaxation times.
        T1 = np.real(T1)
        T2 = np.real(T2)

        # Evaluate the Larmor frequency of the quadrupolar nucleus.
        omega_quad = (
            spin_system.gammas[quad]
            * parameters.magnetic_field
            * (1 + spin_system.chemical_shifts[quad] * 1e-6)
        )

        # Get the spin quantum number of the quadrupolar nucleus.
        S = spin_system.spins[quad]

        # Add the SR2K contribution induced on each target spin.
        for target, gamma in enumerate(spin_system.gammas):

            # Skip the degenerate case of identical gyromagnetic ratios.
            if not spin_system.gammas[quad] == gamma:

                # Evaluate the Larmor frequency of the target spin.
                omega_target = (
                    spin_system.gammas[target]
                    * parameters.magnetic_field
                    * (1 + spin_system.chemical_shifts[target] * 1e-6)
                )

                # Convert the scalar coupling from hertz to rad/s.
                J = 2 * np.pi * spin_system.J_couplings[quad][target]

                # Add the longitudinal and transverse SR2K rate contributions.
                R1[target] += (
                    (J**2) * S * (S + 1) / 3
                    * (2 * T2) / (1 + (omega_target - omega_quad)**2 * T2**2)
                )
                R2[target] += (
                    (J**2) * S * (S + 1) / 3
                    * (T1 + T2 / (1 + (omega_target - omega_quad)**2 * T2**2))
                )

    # Convert the accumulated rates to a phenomenological superoperator.
    with HidePrints():
        sop_R = _sop_R_phenomenological(spin_system.basis.basis, R1, R2)

    # Report the completion of the SR2K calculation.
    status(f"Completed in {time.time() - time_start:.4f} seconds.\n")
    
    return sop_R

def _ldb_thermalization(
    R: np.ndarray | sp.csc_array,
    H_left: np.ndarray | sp.csc_array,
    T: float,
) -> np.ndarray | sp.csc_array:
    """
    Apply Levitt-di Bari thermalization to a relaxation superoperator.

    Parameters
    ----------
    R : ndarray or csc_array
        Relaxation superoperator to be thermalized.
    H_left : ndarray or csc_array
        Left-side coherent Hamiltonian superoperator.
    T : float
        Temperature of the spin bath in Kelvin.
    
    Returns
    -------
    R : ndarray or csc_array
        Thermalized relaxation superoperator.
    """
    # Report the start of the thermalization step.
    status("Applying thermalization to the relaxation superoperator...")
    time_start = time.time()

    # Build the Boltzmann-distribution matrix exponential.
    with HidePrints():
        P = expm(
            const.hbar / (const.k * T) * H_left,
            parameters.zero_thermalization,
        )

    # Right-multiply the relaxation superoperator by the thermalization map.
    R = R @ P

    # Report the completion of the thermalization step.
    status(f"Completed in {time.time() - time_start:.4f} seconds.\n")

    return R


def _validate_relaxation_inputs(spin_system: SpinSystem) -> None:
    """
    Validate the input data required for relaxation calculations.

    Parameters
    ----------
    spin_system : SpinSystem
        Spin system for which the relaxation superoperator is requested.
    """

    # Require that the relaxation theory has been specified explicitly.
    if spin_system.relaxation.theory is None:
        raise ValueError(
            "Please specify relaxation theory before constructing the "
            "relaxation superoperator."
        )

    # Require that the basis has already been built.
    if spin_system.basis.basis is None:
        raise ValueError(
            "Please build basis before constructing the relaxation "
            "superoperator."
        )

    # Validate the phenomenological relaxation inputs.
    if spin_system.relaxation.theory == "phenomenological":
        if spin_system.relaxation.T1 is None:
            raise ValueError(
                "Please set T1 times before constructing the relaxation "
                "superoperator."
            )
        if spin_system.relaxation.T2 is None:
            raise ValueError(
                "Please set T2 times before constructing the relaxation "
                "superoperator."
            )

    # Validate the Redfield relaxation inputs.
    elif spin_system.relaxation.theory == "redfield":
        if spin_system.relaxation.tau_c is None:
            raise ValueError(
                "Please set the correlation time before constructing the "
                "Redfield relaxation superoperator."
            )
        if parameters.magnetic_field is None:
            raise ValueError(
                "Please set the magnetic field before constructing the "
                "Redfield relaxation superoperator."
            )

    # Validate the inputs required for scalar relaxation of the second kind.
    if spin_system.relaxation.sr2k and parameters.magnetic_field is None:
        raise ValueError(
            "Please set the magnetic field before applying scalar "
            "relaxation of the second kind."
        )

    # Validate the inputs required for thermalization.
    if spin_system.relaxation.thermalization:
        if parameters.magnetic_field is None:
            raise ValueError(
                "Please set the magnetic field when applying thermalization."
            )
        if parameters.temperature is None:
            raise ValueError(
                "Please define temperature when applying thermalization."
            )

def relaxation(spin_system: SpinSystem) -> np.ndarray | sp.csc_array:
    """
    Create the relaxation superoperator using the requested level
    of theory.

    Requires that the following spin system properties are set:

    - spin_system.relaxation.theory : must be specified
    - spin_system.basis : must be built

    If ``phenomenological`` relaxation theory is requested, the following must
    be set:

    - spin_system.relaxation.T1
    - spin_system.relaxation.T2

    If ``redfield`` relaxation theory is requested, the following must be set:

    - spin_system.relaxation.tau_c
    - parameters.magnetic_field

    If ``sr2k`` is requested, the following must be set:

    - parameters.magnetic_field

    If ``thermalization`` is requested, the following must be set:

    - parameters.magnetic_field
    - parameters.temperature

    Parameters
    ----------
    spin_system : SpinSystem
        Spin system for which the relaxation superoperator is going to be
        generated.

    Returns
    -------
    R : ndarray or csc_array
        Relaxation superoperator.
    """

    # Validate that all required inputs have been provided.
    _validate_relaxation_inputs(spin_system)

    # Construct the phenomenological relaxation superoperator if requested.
    if spin_system.relaxation.theory == "phenomenological":
        R = _sop_R_phenomenological(
            basis=spin_system.basis.basis,
            R1=spin_system.relaxation.R1,
            R2=spin_system.relaxation.R2,
        )

    # Construct the Redfield relaxation superoperator if requested.
    elif spin_system.relaxation.theory == "redfield":
        R = _sop_R_redfield(spin_system)
    
    # Add scalar relaxation of the second kind if requested.
    if spin_system.relaxation.sr2k:
        R += _sop_R_sr2k(spin_system, R)
        
    # Apply Levitt-di Bari thermalization if requested.
    if spin_system.relaxation.thermalization:
        
        # Build the left Hamiltonian superoperator.
        with HidePrints():
            H_left = hamiltonian(spin_system, side="left")
            
        # Thermalize the relaxation superoperator.
        R = _ldb_thermalization(
            R=R,
            H_left=H_left,
            T=parameters.temperature,
        )

    return R

def _get_all_sop_T(spin_system: SpinSystem, interactions: dict) -> dict:
    """
    Build all coupled spherical tensor superoperators needed for relaxation.

    Parameters
    ----------
    spin_system : SpinSystem
        Spin system for which the superoperators are going to be built.
    interactions : dict
        A dictionary that contains the interaction ranks as the keys. The values
        are lists where each interaction is represented by a tuple. The keys do
        not exist if there are no interactions of that specific rank.

    Returns
    -------
    sop_Ts : dict
        A dictionary that contains the coupled spherical tensor superoperators.
        The keys are tuples: (l, q, itype, spin1, spin2).
    """
    # Report the start of the coupled-superoperator construction.
    status("Building the coupled spherical tensor operators...")
    time_start = time.time()

    # Initialise the dictionary of coupled spherical tensor superoperators.
    sop_Ts = {}

    # Build the required superoperators for each interaction rank.
    for l in interactions.keys():

        # Iterate over the interactions.
        for interaction in interactions[l]:

            # Extract the interaction information.
            itype = interaction[0]
            spin1 = interaction[1]
            spin2 = interaction[2]

            # Build the coupled superoperator for each tensor projection.
            for q in range(-l, l + 1):

                # Construct the coupled spherical tensor superoperator.
                sop_T = _get_sop_T(spin_system, l, q, itype, spin1, spin2)

                # Store the superoperator in the lookup dictionary.
                sop_Ts[(l, q, itype, spin1, spin2)] = sop_T

    # Report the completion of the coupled-superoperator construction.
    status(f"Completed in {time.time() - time_start:.4f} seconds.\n")
    return sop_Ts

def _sop_R_redfield(spin_system: SpinSystem) -> sp.csc_array:
    """
    Calculate the relaxation superoperator using Redfield theory.

    Parameters
    ----------
    spin_system : SpinSystem
        SpinSystem for which the relaxation superoperator is to be calculated.

    Returns
    -------
    R : csc_array
        Relaxation superoperator calculated using the Redfield theory.
    """
    # Report the start of the Redfield relaxation calculation.
    time_start_R = time.time()
    status("Constructing the Redfield relaxation superoperator...\n")

    # Extract the basis dimension and the relative integration tolerance.
    dim = spin_system.basis.dim
    relative_error = spin_system.relaxation.relative_error

    # Calculate the rotational diffusion generator.
    G = _rot_diff_gen(spin_system)

    # Diagonalise the rotational diffusion generators for each rank.
    G_eval = {}
    G_evec = {}
    for l in [1, 2]:
        G_eval[l], G_evec[l] = np.linalg.eig(G[l])

    # Process the interaction tensors in the diffusion-generator basis.
    interactions = _process_interactions(spin_system, G_evec)

    # Build all coupled spherical tensor superoperators.
    sop_Ts = _get_all_sop_T(spin_system, interactions)
    
    # Build the coherent Hamiltonian superoperator.
    with HidePrints():
        H = hamiltonian(spin_system)

    # Construct the top-left block of the auxiliary matrix.
    top_left = 1j * H

    # Initialise the Redfield relaxation superoperator accumulator.
    status("Calculating the relaxation superoperator terms...")
    time_start = time.time()
    R = sp.csc_array((dim, dim), dtype=complex)

    # Accumulate all Redfield terms over ranks, eigenmodes, and projections.
    for l in interactions.keys():

        # Loop over the diffusion-generator eigenmodes.
        for p in range(2 * l + 1):

            # Set the finite integration limit for the auxiliary-matrix method.
            t_max = np.log(1 / relative_error) * (1 / G_eval[l][p])

            # Build the diagonal block containing the diffusion eigenvalue.
            G_eval_diag = G_eval[l][p] * sp.eye_array(dim, format="csc")

            # Construct the bottom-right block of the auxiliary matrix.
            bottom_right = 1j * H - G_eval_diag

            # Loop over the tensor projections of the current rank.
            for q in range(-l, l + 1):

                # Report the current Redfield-term indices.
                status(f"l = {l}, p = {p}, q = {q}")

                # Construct the three-index superoperator for the current mode.
                sop_X_lpq = sp.csc_array((dim, dim), dtype=complex)
                for interaction in interactions[l]:

                    # Extract the interaction information.
                    itype = interaction[0]
                    spin1 = interaction[1]
                    spin2 = interaction[2]
                    V_lpu = interaction[3][p]

                    # Retrieve the precomputed coupled spherical tensor term.
                    sop_T_u = sop_Ts[(l, q, itype, spin1, spin2)]

                    # Add the weighted contribution to the mode-specific sum.
                    sop_X_lpq = sop_X_lpq + V_lpu * sop_T_u.conj().T

                # Evaluate the Redfield integral with the auxiliary matrix.
                aux_expm = auxiliary_matrix_expm(
                    A=top_left,
                    B=sop_X_lpq.conj().T,
                    C=bottom_right,
                    t=t_max,
                    zero_value=parameters.zero_aux,
                )
                aux_top_l = aux_expm[:dim, :dim]
                aux_top_r = aux_expm[:dim, dim:]
                integral = aux_top_l.conj().T @ aux_top_r

                # Add the current Redfield term to the total superoperator.
                R = R + (1 / (2*l + 1)) * sop_X_lpq @ integral

    # Report the completion of the Redfield-term accumulation.
    status(f"Completed in {time.time() - time_start:.4f} seconds.\n")

    # Remove the dynamic frequency shifts unless they were requested.
    if not spin_system.relaxation.dynamic_frequency_shift:
        status("Removing dynamic frequency shifts...")
        time_start = time.time()
        R = R.real
        status(f"Completed in {time.time() - time_start:.4f} seconds.\n")

    # Remove numerically negligible matrix elements.
    status("Eliminating small values from the relaxation superoperator...")
    time_start = time.time()
    eliminate_small(R, parameters.zero_relaxation)
    status(f"Completed in {time.time() - time_start:.4f} seconds.\n")
    
    # Report the total time required for the Redfield construction.
    status(
        f"Redfield relaxation superoperator constructed in "
        f"{time.time() - time_start_R:.4f} seconds.\n"
    )

    return R