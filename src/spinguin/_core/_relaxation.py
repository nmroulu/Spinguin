"""
This module provides functions for calculating relaxation superoperators.
"""
# Referencing SpinSystem class
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from spinguin._core._spin_system import SpinSystem

# Imports
import time
import numpy as np
import scipy.constants as const
import scipy.sparse as sp
from spinguin._core._operators import op_Sx, op_Sy, op_Sz
from spinguin._core._superoperators import sop_T_coupled, superoperator
from spinguin._core._la import (
    eliminate_small,
    cartesian_tensor_to_spherical_tensor,
    auxiliary_matrix_expm,
    expm,
    decompose_matrix,
    rotation_matrix_to_align_axes
)
from spinguin._core._utils import idx_to_lq, lq_to_idx, parse_operator_string
from spinguin._core._hide_prints import HidePrints
from spinguin._core._parameters import parameters
from spinguin._core._hamiltonian import hamiltonian
from spinguin._core._status import status
from typing import Literal
from scipy.optimize import fsolve

def dd_constant(y1: float, y2: float) -> float:
    """
    Calculates the dipole-dipole coupling constant (excluding the distance).

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

    # Calculate the constant
    dd_const = -const.mu_0 / (4 * np.pi) * y1 * y2 * const.hbar

    return dd_const

def Q_constant(S: float, Q_moment: float) -> float:
    """
    Calculates the nuclear quadrupolar coupling constant in (rad/s) / (V/m^2).
    
    Parameters
    ----------
    S : float
        Spin quantum number.
    Q_moment : float
        Nuclear quadrupole moment (in units of m^2).

    Returns
    -------
    Q_const : float
        Quadrupolar coupling constant.
    """

    # Calculate the quadrupolar coupling constant
    if (S >= 1) and (Q_moment > 0):
        Q_const = -const.e * Q_moment / const.hbar / (2 * S * (2 * S - 1))
    else:
        Q_const = 0
    
    return Q_const

def tau_c_l(tau_c: float | np.ndarray, l: int) -> float | np.ndarray:
    """
    Calculates the rotational correlation time for a given rank `l`. 
    Applies only for anisotropic rotationally modulated interactions (l > 0).

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
    """

    # Calculate the rotational correlation time for anisotropic interactions
    if l != 0:
        t_cl = 6 * tau_c / (l * (l + 1))

    # For isotropic interactions raise an error
    else:
        raise ValueError('Rank l must be different from 0 in tau_c_l.')
    
    return t_cl
    
def dd_coupling_tensors(xyz: np.ndarray, gammas: np.ndarray) -> np.ndarray:
    """
    Calculates the dipole-dipole coupling tensor between all nuclei
    in the spin system.

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

    # Deduce the number of spins in the system
    nspins = gammas.shape[0]

    # Convert the molecular coordinates to SI units
    xyz = xyz * 1e-10

    # Get the connector and distance arrays
    connectors = xyz[:, np.newaxis] - xyz
    distances = np.linalg.norm(connectors, axis=2)

    # Initialize the array of tensors
    dd_tensors = np.zeros((nspins, nspins, 3, 3))

    # Go through each spin pair
    for i in range(nspins):
        for j in range(nspins):

            # Only the lower triangular part is computed
            if i > j:
                rr = np.outer(connectors[i, j], connectors[i, j])
                dd_tensors[i, j] = dd_constant(gammas[i], gammas[j]) * \
                                   (3 * rr - distances[i, j]**2 * np.eye(3)) / \
                                   distances[i, j]**5

    return dd_tensors

def shielding_intr_tensors(shielding: np.ndarray,
                           gammas: np.ndarray, B: float) -> np.ndarray:
    """
    Calculates the shielding interaction tensors for a spin system.

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
    shielding_tensors: ndarray
        Array of shielding tensors.
    """

    # Convert from ppm to dimensionless
    shielding_tensors = shielding * 1e-6

    # Create Larmor frequencies ("shielding constants" for relaxation)
    # TODO: Check the sign of the Larmor frequency (Perttu?)
    w0s = -gammas * B

    # Multiply with the Larmor frequencies
    for i, val in enumerate(w0s):
        shielding_tensors[i] *= val

    return shielding_tensors

# TODO: Check the sign (Perttu?)
def Q_intr_tensors(efg: np.ndarray,
                   spins: np.ndarray,
                   quad: np.ndarray) -> np.ndarray:
    """
    Calculates the quadrupolar interaction tensors for a spin system.

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
    Q_tensors: ndarray
        Quadrupolar interaction tensors.
    """

    # Convert from a.u. to V/m^2
    Q_tensors = -9.7173624292e21 * efg

    # Create quadrupolar coupling constants
    Q_constants = [Q_constant(S, Q) for S, Q in zip(spins, quad)]

    # Multiply the tensors with the quadrupolar coupling constants
    for i, val in enumerate(Q_constants):
        Q_tensors[i] *= val

    return Q_tensors

def rotational_diffusion_constant_SED(T: float, 
                                      eta: float, 
                                      r: float | np.ndarray) -> float | np.ndarray:
    """
    Calculates the rotational diffusion constant using the Stokes-Einstein-Debye
    (SED) relation.

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
    # Convert r from Å to meters
    r_meters = r * 1e-10

    # Calculate the rotational diffusion constant
    D_r = const.k * T / (8 * np.pi * eta * r_meters**3)
    return D_r

def rotational_correlation_time(l: int, 
                                D_r: float | np.ndarray) -> float | np.ndarray:
    """
    Calculates the rotational correlation time for a given rank `l`.

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
    # Calculate the rotational correlation time
    tau_c = 1 / (l * (l + 1) * D_r)
    return tau_c

def rotational_correlation_time_SED(T: float,
                                    eta: float,
                                    r: float | np.ndarray,
                                    l: int) -> float | np.ndarray:
    """
    Calculates the rotational correlation time using the Stokes-Einstein-Debye
    (SED) relation for a given tensor rank.

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
    # Calculate the rotational diffusion constant using SED
    D_r = rotational_diffusion_constant_SED(T, eta, r)

    # Calculate the rotational correlation time for the given rank
    tau_c = rotational_correlation_time(l, D_r)

    return tau_c

def center_of_mass(masses: np.ndarray, 
                   coords: np.ndarray) -> np.ndarray:
    """
    Calculates the center of mass for a molecule.

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
        Center of mass coordinates in units of Å.
    """
    total_mass = np.sum(masses)
    return np.sum(masses[:, np.newaxis] * coords, axis=0) / total_mass

def moment_of_inertia_tensor(masses: np.ndarray, 
                             coords: np.ndarray) -> np.ndarray:
    """
    Calculates the moment of inertia tensor for a molecule.

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

    # Number of atoms
    natoms = masses.shape[0]

    # Calculate the center of mass
    center_of_mass_coords = center_of_mass(masses, coords)

    # Center the coordinates at the center of mass
    coords_centered = coords - center_of_mass_coords

    # Initialize the moment of inertia tensor
    I = np.zeros((3, 3))

    # Go through each atom
    for i in range(natoms):
        x, y, z = coords_centered[i]
        m = masses[i]

        # Update the moment of inertia tensor
        I[0, 0] += m * (y**2 + z**2)
        I[1, 1] += m * (x**2 + z**2)
        I[2, 2] += m * (x**2 + y**2)
        I[0, 1] -= m * x * y
        I[0, 2] -= m * x * z
        I[1, 2] -= m * y * z

    # Fill the symmetric elements
    I[1, 0] = I[0, 1]
    I[2, 0] = I[0, 2]
    I[2, 1] = I[1, 2]

    return I

def moment_of_inertia_equivalent_ellipsoid(masses: np.ndarray,
                                           coords: np.ndarray) -> np.ndarray:
    """
    Calculates the three semi-axes a_x, a_y, and a_z of an mass-equivalent 
    ellipsoid that has the same moment of inertia tensor eigenvalues as the molecule.
    The semi-axes are determined by solving the equations:
        
        I_1 = (1/5) * M * (a_y^2 + a_z^2)
        I_2 = (1/5) * M * (a_x^2 + a_z^2)
        I_3 = (1/5) * M * (a_x^2 + a_y^2)

    where I_1, I_2, and I_3 are the eigenvalues of the moment of inertia tensor,
    M is the total mass of the molecule, and a_x, a_y, a_z are the semi-axes of the 
    ellipsoid.

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
    # Compute the moment of inertia tensor and its eigenvalues
    I = moment_of_inertia_tensor(masses, coords)
    eigenvalues, _ = np.linalg.eigh(I)

    # Calculate the total mass of the molecule
    total_mass = np.sum(masses)

    # Calculate the semi-axes by solving the system of equations using Scipy's fsolve
    def equations(vars):
        a_x, a_y, a_z = vars
        eq1 = (1/5) * total_mass * (a_y**2 + a_z**2) - eigenvalues[0]
        eq2 = (1/5) * total_mass * (a_x**2 + a_z**2) - eigenvalues[1]
        eq3 = (1/5) * total_mass * (a_x**2 + a_y**2) - eigenvalues[2]
        return [eq1, eq2, eq3]

    initial_guess = [2.0, 2.0, 2.0]  # Initial guess for the semi-axes in Å
    a_x, a_y, a_z = fsolve(equations, initial_guess, maxfev=10000, xtol=1e-15)

    return np.array([a_x, a_y, a_z])

def Perrin_integrals(a_x: float, 
                     a_y: float, 
                     a_z: float, 
                     upper_limit: float = 1e4, 
                     num_points: int = int(1e6)) -> tuple[float, float, float]:
    """
    Calculates the Perrin integrals P, Q, and R for an ellipsoid with semi-axes a_x, a_y, and a_z.

    Parameters
    ----------
    a_x : float
        Semi-axis a_x of the ellipsoid in units of Å.
    a_y : float
        Semi-axis a_y of the ellipsoid in units of Å.
    a_z : float
        Semi-axis a_z of the ellipsoid in units of Å.
    upper_limit : float, optional
        Upper limit for the integration. Default is 1e4.
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
    # Axis to integrate over
    s = np.linspace(0, upper_limit, num_points)  # More points for better accuracy

    # Calculate the integrands
    integrand_P = 1 / np.sqrt((a_x**2 + s)**3 * (a_y**2 + s) * (a_z**2 + s))
    integrand_Q = 1 / np.sqrt((a_y**2 + s)**3 * (a_z**2 + s) * (a_x**2 + s))
    integrand_R = 1 / np.sqrt((a_z**2 + s)**3 * (a_x**2 + s) * (a_y**2 + s))

    # Perform the integrations using the trapezoidal rule
    P = np.trapz(integrand_P, s)
    Q = np.trapz(integrand_Q, s)
    R = np.trapz(integrand_R, s)

    return P, Q, R

def Perrin_factors(a_x: float, 
                   a_y: float, 
                   a_z: float) -> np.ndarray:
    """
    Calculates the Perrin factors for an ellipsoid with semi-axes a_x, a_y, and a_z.

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
    # Calculate the Perrin integrals for the ellipsoid
    P, Q, R = Perrin_integrals(a_x, a_y, a_z)

    # Calculate the Perrin factors along each axis
    f_x = (a_y**2 + a_z**2) / (a_y**2 * Q + a_z**2 * R)
    f_y = (a_x**2 + a_z**2) / (a_z**2 * R + a_x**2 * P)
    f_z = (a_x**2 + a_y**2) / (a_x**2 * P + a_y**2 * Q)

    return np.array([f_x, f_y, f_z])

def rotational_diffusion_constants_Perrin(T: float,
                                         eta: float,
                                         a_x: float,
                                         a_y: float,
                                         a_z: float) -> np.ndarray:
    """
    Calculates the rotational diffusion constants along the principal axes of an ellipsoid
    using the Perrin factors.

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
    # Calculate the Perrin factors for the ellipsoid
    perrin_factors = Perrin_factors(a_x, a_y, a_z)

    # Convert to correct units
    perrin_factors *= 1e-30  # Convert from Å^3 to m^3

    # Calculate the rotational diffusion constants along the principal axes
    D_rs = (const.k * T / perrin_factors) / (16 * np.pi * eta / 3)

    return D_rs

def rotational_correlation_times_Perrin(masses: np.ndarray,
                                        coords: np.ndarray,
                                        T: float,
                                        eta: float,
                                        l: int,
                                        scale: float = 1.0) -> np.ndarray:
    """
    Calculates the rotational correlation times along the principal axes
    of rotation using the Perrin factors for a given tensor rank.

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
    scale : float, optional
        Scaling factor for the rotational diffusion constants. Default is 1.0.
        Useful for adjusting the correlation times to better match experimental data.

    Returns
    -------
    tau_cs : ndarray
        Rotational correlation times along the principal axes in seconds.
        The output is an array of shape (3,) corresponding to the three
        principal axes of the ellipsoid.
    """
    # Calculate the semi-axes of the mass-equivalent ellipsoid
    semi_axes = moment_of_inertia_equivalent_ellipsoid(masses, coords)

    # NOTE: 
    # Add hydrodynamic thickness to semi-axes to account for the effects
    # of solvent interactions and vibrational motion on the effective size
    # of the molecule. Necessary for flat molecules where one of the semi-axes
    # can be very small. 
    semi_axes += 1.0  # Minimum thickness

    # Calculate the rotational diffusion constants along the principal axes using Perrin factors
    D_rs = rotational_diffusion_constants_Perrin(T, eta, semi_axes[0], semi_axes[1], semi_axes[2])

    # Apply scaling factor to the rotational diffusion constants
    D_rs *= scale

    # Calculate the rotational correlation times along the principal axes
    tau_cs = rotational_correlation_time(l, D_rs)

    return tau_cs

def _rot_diff_gen(spin_system: SpinSystem) -> dict:
    """
    Generates the rotational diffusion generator for the spin system.

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
    status("Calculating rotational diffusion generator...")
    time_start = time.time()

    # Obtain the rotational correlation time
    tau_c = spin_system.relaxation.tau_c
    
    # Initialise a dictionary for the rotational diffusion generator
    G = {}

    # Calculate the rotational diffusion generator for each rank l
    for l in [1, 2]:

        # Get the angular momentum operators
        Jx = op_Sx(l)
        Jy = op_Sy(l)
        Jz = op_Sz(l)

        # Convert to NumPy if not already
        if parameters.sparse_operator:
            Jx = Jx.toarray()
            Jy = Jy.toarray()
            Jz = Jz.toarray()

        # Get the correlation time for the current rank
        tau_cl = tau_c_l(tau_c, l)

        # Isotropic case
        if isinstance(spin_system.relaxation.tau_c, float):
            D = 1 / (tau_cl * l * (l + 1))
            G[l] = D * (Jx @ Jx + Jy @ Jy + Jz @ Jz)

        # Symmetric top case
        elif len(spin_system.relaxation.tau_c) == 2:
            D1 = 1 / (tau_cl[0] * l * (l + 1))
            D2 = 1 / (tau_cl[1] * l * (l + 1))
            G[l] = D1 * Jz @ Jz + D2 * (Jx @ Jx + Jy @ Jy)

        # General anisotropic case
        else:
            D1 = 1 / (tau_cl[0] * l * (l + 1))
            D2 = 1 / (tau_cl[1] * l * (l + 1))
            D3 = 1 / (tau_cl[2] * l * (l + 1))
            G[l] = D1 * Jx @ Jx + D2 * Jy @ Jy + D3 * Jz @ Jz

    status(f"Completed in {time.time() - time_start:.4f} seconds.\n")

    return G

def _process_interaction_tensor(
    V: np.ndarray,
    R : np.ndarray,
    dge: dict,
    anti: bool
) -> dict:
    """
    Decomposes the interaction tensor into a dictionary of dictionaries, where
    the first key is the interaction rank `l` and the second key is the
    component `p` matching to the eigenvalues / eigenvectors of the rotational
    diffusion generator. The key for the interaction rank is omitted if the
    corresponding part of the interaction tensor is small.

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
        the interaction rank `l`, and the second keys are the components `p`.
    """
    # Create an empty dictionary to hold the interaction tensor
    V_lp = {}

    # Decompose the interaction tensor into symmetric and antisymmetric part
    _, V1, V2 = decompose_matrix(V)

    # Choose whether to include antisymmetric / symmetric parts
    if anti and np.linalg.norm(V1, ord=2) > parameters.zero_interaction:
        V_lp[1] = {}
    if np.linalg.norm(V2, ord=2) > parameters.zero_interaction:
        V_lp[2] = {}

    # If nothing is included, return
    if len(V_lp) == 0:
        return V_lp

    # Transform the tensor to the rotational principal axes frame
    V = R @ V @ R.T

    # Write the tensor in the spherical tensor notation
    V = cartesian_tensor_to_spherical_tensor(V)

    # Compute the interaction tensor terms
    for l in V_lp.keys():
        for p in range(2*l+1):
            V_lp[l][p] = sum(V[l, m] * dge[l][l-m, p] for m in range(-l, l + 1))

    return V_lp

def _process_interactions(spin_system: SpinSystem, dge: dict) -> dict:
    """
    Obtains all interaction tensors from the spin system, and organizes them by
    their interaction rank `l`. Interaction tensors whose norm is below the
    threshold specified in global parameters are disregarded.

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
        A dictionary where the interactions are organized by rank `l`. The
        values are lists containing all interactions with meaningful
        strength. The interactions are tuples in the format::
        
            (interaction, spin_1, spin_2, tensor).
    """
    status("Processing interactions for relaxation...")
    time_start = time.time()

    # Choose between isotropic and anisotropic models
    iso = isinstance(spin_system.relaxation.tau_c, float)

    # Rotation matrix that aligns the lab frame to the rotational principal axes
    if iso:
        R = np.eye(3)
    else:
        masses = spin_system.relaxation.molecule.masses
        coords = spin_system.relaxation.molecule.xyz
        moi_tensor = moment_of_inertia_tensor(masses, coords)
        _, rot_principal_axes = np.linalg.eigh(moi_tensor)
        lab_frame = np.eye(3)
        R = rotation_matrix_to_align_axes(lab_frame, rot_principal_axes.T)

    # Initialize the dictionary
    interactions = {}
    for l in [1, 2]:
        interactions[l] = []

    # Process dipole-dipole couplings
    if spin_system.xyz is not None:

        # Get the DD-coupling tensors
        dd_tensors = dd_coupling_tensors(spin_system.xyz, spin_system.gammas)

        # Go through the DD-coupling tensors
        for spin_1 in range(spin_system.nspins):
            for spin_2 in range(spin_1):
                V = dd_tensors[spin_1, spin_2]
                V = _process_interaction_tensor(V, R, dge, anti=False)
                for l in V.keys():
                    interactions[l].append(("DD", spin_1, spin_2, V[l]))

    # Process nuclear shielding
    if spin_system.shielding is not None:

        # Get the shielding tensors
        sh_tensors = shielding_intr_tensors(
            spin_system.shielding,
            spin_system.gammas,
            parameters.magnetic_field
        )

        # Choose whether to process antisymmetric part of CSA
        anti = spin_system.relaxation.antisymmetric
        
        # Go through the shielding tensors
        for spin_1 in range(spin_system.nspins):
            V = sh_tensors[spin_1]
            V = _process_interaction_tensor(V, R, dge, anti=anti)
            for l in V.keys():
                interactions[l].append(("CSA", spin_1, None, V[l]))

    # Process quadrupolar coupling
    if spin_system.efg is not None:

        # Get the quadrupole coupling tensors
        q_tensors = Q_intr_tensors(
            spin_system.efg,
            spin_system.spins,
            spin_system.quad
        )

        # Go through the quadrupole coupling tensors
        for spin_1 in range(spin_system.nspins):
            if spin_system.spins[spin_1] > 1/2:
                V = q_tensors[spin_1]
                V = _process_interaction_tensor(V, R, dge, anti=False)
                for l in V.keys():
                    interactions[l].append(("Q", spin_1, None, V[l]))

    # Remove empty keys
    for l in [1, 2]:
        if len(interactions[l]) == 0:
            del interactions[l]

    status(f"Completed in {time.time() - time_start:.4f} seconds.\n")

    return interactions

def _get_sop_T(
    spin_system: SpinSystem,
    l: int,
    q: int,
    interaction_type: Literal["CSA", "Q", "DD"],
    spin_1: int,
    spin_2: int = None
) -> np.ndarray | sp.csc_array:
    """
    Helper function for the relaxation module. Calculates the coupled product 
    superoperators for different interaction types.

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
        Coupled spherical tensor superoperator of rank `l` and projection `q`.
    """

    # Single-spin linear interaction
    if interaction_type == "CSA":
        sop = sop_T_coupled(spin_system, l, q, spin_1)

    # Single-spin quadratic interaction
    elif interaction_type == "Q":
        sop = superoperator(spin_system, f"T({l},{q},{spin_1})")

    # Two-spin bilinear interaction
    elif interaction_type == "DD":
        sop = sop_T_coupled(spin_system, l, q, spin_1, spin_2)

    # Raise an error for invalid interaction types
    else:
        raise ValueError(f"Invalid interaction type '{interaction_type}' for "
                         "relaxation superoperator. Possible options are " 
                         "'CSA', 'Q', and 'DD'.")

    return sop

def sop_R_random_field():
    """
    TODO PERTTU?
    """

def _sop_R_phenomenological(
    basis: np.ndarray,
    R1: np.ndarray,
    R2: np.ndarray
) -> np.ndarray | sp.csc_array:
    """
    Constructs the relaxation superoperator from given `R1` and `R2` values
    for each spin.

    Parameters
    ----------
    basis : ndarray
        A 2-dimensional array containing the basis set that contains sequences
        of integers describing the Kronecker products of irreducible spherical
        tensors.
    R1 : ndarray
        A one dimensional array containing the longitudinal relaxation rates
        in 1/s for each spin. For example: `np.array([1.0, 2.0, 2.5])`
    R2 : ndarray
        A one dimensional array containing the transverse relaxation rates
        in 1/s for each spin. For example: `np.array([2.0, 4.0, 5.0])`

    Returns
    -------
    sop_R : ndarray or csc_array
        Relaxation superoperator.
    """

    time_start = time.time()
    status('Constructing the phenomenological relaxation superoperator...')

    # Obtain the basis dimension
    dim = basis.shape[0]

    # Create an empty array for the relaxation superoperator
    if parameters.sparse_superoperator:
        sop_R = sp.lil_array((dim, dim))
    else:
        sop_R = np.zeros((dim, dim))

    # Loop over the basis set
    for idx, state in enumerate(basis):

        # Initialize the relaxation rate for the current state
        R_state = 0
        
        # Loop over the state
        for spin, operator in enumerate(state):

            # Continue only if the operator is not the unit state
            if operator != 0:

                # Get the projection of the state
                _, q = idx_to_lq(operator)
            
                # Check if the current spin has a longitudinal state
                if q == 0:
                    
                    # Add to the relaxation rate
                    R_state += R1[spin]

                # Otherwise, the state must be transverse
                else:

                    # Add to the relaxation rate
                    R_state += R2[spin]

        # Add to the relaxation matrix
        sop_R[idx, idx] = R_state

    # Convert to CSC array if using sparse
    if parameters.sparse_superoperator:
        sop_R = sop_R.tocsc()

    status(f"Completed in {time.time() - time_start:.4f} seconds.\n")

    return sop_R

def _sop_R_sr2k(
    spin_system: SpinSystem,
    R: sp.csc_array,
) -> np.ndarray | sp.csc_array:
    """
    Calculates the scalar relaxation of the second kind (SR2K) based on 
    Abragam's formula.

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

    status("Processing scalar relaxation of the second kind...")
    time_start = time.time()

    # Make a dictionary of the basis for fast lookup
    basis_lookup = {
        tuple(row): idx
        for idx, row in enumerate(spin_system.basis.basis)
    }

    # Initialize arrays for the relaxation rates
    R1 = np.zeros(spin_system.nspins)
    R2 = np.zeros(spin_system.nspins)

    # Obtain indices of quadrupolar nuclei in the system
    quadrupolar = []
    for i, spin in enumerate(spin_system.spins):
        if spin > 0.5:
            quadrupolar.append(i)
    
    # Loop over the quadrupolar nuclei
    for quad in quadrupolar:

        # Find the operator definitions of the longitudinal and transverse
        # states
        op_def_z, _ = parse_operator_string(f"I(z, {quad})", spin_system.nspins)
        op_def_p, _ = parse_operator_string(f"I(+, {quad})", spin_system.nspins)

        # Convert operator definitions to tuple for searching the basis
        op_def_z = tuple(op_def_z[0])
        op_def_p = tuple(op_def_p[0])

        # Find the indices of the longitudinal and transverse states
        idx_long = basis_lookup[op_def_z]
        idx_trans = basis_lookup[op_def_p]

        # Find the relaxation times of the quadrupolar nucleus
        T1 = 1 / R[idx_long, idx_long]
        T2 = 1 / R[idx_trans, idx_trans]

        # Convert to real values
        T1 = np.real(T1)
        T2 = np.real(T2)

        # Find the Larmor frequency of the quadrupolar nucleus
        omega_quad = spin_system.gammas[quad] \
                   * parameters.magnetic_field \
                   * (1 + spin_system.chemical_shifts[quad] * 1e-6)

        # Find the spin quantum number of the quadrupolar nucleus
        S = spin_system.spins[quad]

        # Loop over all spins
        for target, gamma in enumerate(spin_system.gammas):

            # Proceed only if the gyromagnetic ratios are different
            if not spin_system.gammas[quad] == gamma:

                # Find the Larmor frequency of the target spin
                omega_target = spin_system.gammas[target] \
                             * parameters.magnetic_field \
                             * (1 + spin_system.chemical_shifts[target] * 1e-6)

                # Find the scalar coupling between spins in rad/s
                J = 2 * np.pi * spin_system.J_couplings[quad][target]

                # Calculate the relaxation rates
                R1[target] += ((J**2) * S * (S + 1)) / 3 * \
                    (2 * T2) / (1 + (omega_target - omega_quad)**2 * T2**2)
                R2[target] += ((J**2) * S * (S + 1)) / 3 * \
                    (T1 + (T2 / (1 + (omega_target - omega_quad)**2 * T2**2)))

    # Get relaxation superoperator corresponding to SR2K
    with HidePrints():
        sop_R = _sop_R_phenomenological(spin_system.basis.basis, R1, R2)

    status(f"Completed in {time.time() - time_start:.4f} seconds.\n")
    
    return sop_R

def _ldb_thermalization(
    R: np.ndarray | sp.csc_array,
    H_left: np.ndarray |sp.csc_array,
    T: float
) -> np.ndarray | sp.csc_array:
    """
    Applies the Levitt-Di Bari thermalization to the relaxation superoperator.

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
    status("Applying thermalization to the relaxation superoperator...")
    time_start = time.time()

    # Get the matrix exponential corresponding to the Boltzmann distribution
    with HidePrints():
        P = expm(const.hbar/(const.k*T)*H_left, parameters.zero_thermalization)

    # Calculate the thermalized relaxation superoperator
    R = R @ P

    status(f"Completed in {time.time() - time_start:.4f} seconds.\n")

    return R

def relaxation(spin_system: SpinSystem) -> np.ndarray | sp.csc_array:
    """
    Creates the relaxation superoperator using the requested relaxation theory.

    Requires that the following spin system properties are set:

    - spin_system.relaxation.theory : must be specified
    - spin_system.basis : must be built

    If `phenomenological` relaxation theory is requested, the following must
    be set:

    - spin_system.relaxation.T1
    - spin_system.relaxation.T2

    If `redfield` relaxation theory is requested, the following must be set:

    - spin_system.relaxation.tau_c
    - parameters.magnetic_field

    If `sr2k` is requested, the following must be set:

    - parameters.magnetic_field

    If `thermalization` is requested, the following must be set:

    - parameters.magnetic_field
    - parameters.thermalization

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
    # Check that the required attributes have been set
    if spin_system.relaxation.theory is None:
        raise ValueError("Please specify relaxation theory before "
                         "constructing the relaxation superoperator.")
    if spin_system.basis.basis is None:
        raise ValueError("Please build basis before constructing the "
                         "relaxation superoperator.")
    if spin_system.relaxation.theory == "phenomenological":
        if spin_system.relaxation.T1 is None:
            raise ValueError("Please set T1 times before constructing the "
                             "relaxation superoperator.")
        if spin_system.relaxation.T2 is None:
            raise ValueError("Please set T2 times before constructing the "
                             "relaxation superoperator.")
    elif spin_system.relaxation.theory == "redfield":
        if spin_system.relaxation.tau_c is None:
            raise ValueError("Please set the correlation time before "
                             "constructing the Redfield relaxation "
                             "superoperator.")
        if parameters.magnetic_field is None:
            raise ValueError("Please set the magnetic field before "
                             "constructing the Redfield relaxation "
                             "superoperator.")
    if spin_system.relaxation.sr2k:
        if parameters.magnetic_field is None:
            raise ValueError("Please set the magnetic field before "
                             "applying scalar relaxation of the second kind.")
    if spin_system.relaxation.thermalization:
        if parameters.magnetic_field is None:
            raise ValueError("Please set the magnetic field when applying "
                             "thermalization.")
        if parameters.temperature is None:
            raise ValueError("Please define temperature when applying "
                             "thermalization.")

    # Make phenomenological relaxation superoperator
    if spin_system.relaxation.theory == "phenomenological":
        R = _sop_R_phenomenological(
            basis = spin_system.basis.basis,
            R1 = spin_system.relaxation.R1,
            R2 = spin_system.relaxation.R2,
        )

    # Make relaxation superoperator using Redfield theory
    elif spin_system.relaxation.theory == "redfield":
        R = _sop_R_redfield(spin_system)
    
    # Apply scalar relaxation of the second kind if requested
    if spin_system.relaxation.sr2k:
        R += _sop_R_sr2k(spin_system, R)
        
    # Apply thermalization if requested
    if spin_system.relaxation.thermalization:
        
        # Build the left Hamiltonian superopertor
        with HidePrints():
            H_left = hamiltonian(spin_system, side="left")
            
        # Perform the thermalization
        R = _ldb_thermalization(
            R = R,
            H_left = H_left,
            T = parameters.temperature
        )

    return R

def _get_all_sop_T(spin_system: SpinSystem, interactions: dict) -> dict:
    """
    Helper function that builds all the coupled spherical tensor superoperators.

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
    status("Building the coupled spherical tensor operators...")
    time_start = time.time()

    # Create an empty dictionary for the coupled spherical tensor operators
    sop_Ts = {}

    # Iterate over the interaction ranks l
    for l in interactions.keys():

        # Iterate over the interactions
        for interaction in interactions[l]:

            # Extract the interaction information
            itype = interaction[0]
            spin1 = interaction[1]
            spin2 = interaction[2]

            # Iterate over the projections q
            for q in range(-l, l+1):

                # Compute the coupled T superoperator
                sop_T = _get_sop_T(spin_system, l, q, itype, spin1, spin2)

                # Store the coupled T superoperator to the dictionary
                sop_Ts[(l, q, itype, spin1, spin2)] = sop_T

    status(f"Completed in {time.time() - time_start:.4f} seconds.\n")
    return sop_Ts

def _sop_R_redfield(spin_system: SpinSystem) -> sp.csc_array:
    """
    Calculates the relaxation superoperator R using the Redfield theory.

    Parameters
    ----------
    spin_system : SpinSystem
        SpinSystem for which the relaxation superoperator is to be calculated.

    Returns
    -------
    R : csc_array
        Relaxation superoperator calculated using the Redfield theory.
    """
    time_start_R = time.time()
    status('Constructing relaxation superoperator using Redfield theory...\n')

    # Extract information from the spin system
    dim = spin_system.basis.dim
    relative_error = spin_system.relaxation.relative_error

    # Calculate the rotational diffusion generator
    G = _rot_diff_gen(spin_system)

    # Get the eigenvalues and eigenvectors
    G_eval = {}
    G_evec = {}
    for l in [1, 2]:
        G_eval[l], G_evec[l] = np.linalg.eig(G[l])

    # Process the interactions
    interactions = _process_interactions(spin_system, G_evec)

    # Build coupled spherical tensor operators
    sop_Ts = _get_all_sop_T(spin_system, interactions) 
    
    # Build the coherent Hamiltonian superoperator
    with HidePrints():
        H = hamiltonian(spin_system)

    # Build the top left array of the auxiliary matrix (A)
    top_left = 1j * H

    # Start building the relaxation superoperator
    status("Calculating the Redfield superoperator terms...")
    time_start = time.time()
    R = sp.csc_array((dim, dim), dtype=complex)

    # Iterate over the ranks
    for l in interactions.keys():

        # Iterate over diffusion tensor eigenvalues/eigenvectors
        for p in range(2*l + 1):

            # Define the integration limit for the auxiliary matrix method
            t_max = np.log(1 / relative_error) * (1 / G_eval[l][p])

            # Diagonal matrix of the eigenvalues
            G_eval_diag = G_eval[l][p] * sp.eye_array(dim, format='csc')

            # Bottom right array of auxiliary matrix (C)
            bottom_right = 1j * H - G_eval_diag

            # Iterate over the projections
            for q in range(-l, l + 1):

                status(f"l = {l}, p = {p}, q = {q}")

                # Calculate the three-index operators
                sop_X_lpq = sp.csc_array((dim, dim), dtype=complex)
                for interaction in interactions[l]:

                    # Extract the interaction information
                    itype = interaction[0]
                    spin1 = interaction[1]
                    spin2 = interaction[2]
                    V_lpu = interaction[3][p]

                    # Acquire the coupled T superoperator
                    sop_T_u = sop_Ts[(l, q, itype, spin1, spin2)]

                    # Add to the sum
                    sop_X_lpq = sop_X_lpq + V_lpu * sop_T_u.conj().T

                # Calculate the Redfield integral
                aux_expm = auxiliary_matrix_expm(
                    A = top_left,
                    B = sop_X_lpq.conj().T,
                    C = bottom_right,
                    t = t_max,
                    zero_value = parameters.zero_aux
                )
                aux_top_l = aux_expm[:dim, :dim]
                aux_top_r = aux_expm[:dim, dim:]
                integral = aux_top_l.conj().T @ aux_top_r

                # Add the current term to the relaxation superoperator
                R = R + (1 / (2*l + 1)) * sop_X_lpq @ integral

    status(f"Completed in {time.time() - time_start:.4f} seconds.\n")

    # Return only real values unless dynamic frequency shifts are requested
    if not spin_system.relaxation.dynamic_frequency_shift:
        status("Removing the dynamic frequency shifts...")
        time_start = time.time()
        R = R.real
        status(f"Completed in {time.time() - time_start:.4f} seconds.\n")

    # Eliminate small values
    status("Eliminating small values from the relaxation superoperator...")
    time_start = time.time()
    eliminate_small(R, parameters.zero_relaxation)
    status(f"Completed in {time.time() - time_start:.4f} seconds.\n")
    
    status(
        "Redfield relaxation superoperator constructed in "
        f"{time.time() - time_start_R:.4f} seconds.\n"
    )

    return R