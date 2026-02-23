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
from joblib import Parallel, delayed
from scipy.special import eval_legendre
from spinguin._core._superoperators import sop_T_coupled, sop_prod
from spinguin._core._la import \
    eliminate_small, principal_axis_system, \
    cartesian_tensor_to_spherical_tensor, angle_between_vectors, norm_1, \
    auxiliary_matrix_expm, expm, read_shared_sparse, write_shared_sparse, \
    rotation_matrix_to_align_axes
from spinguin._core._utils import idx_to_lq, lq_to_idx, parse_operator_string
from spinguin._core._hide_prints import HidePrints
from spinguin._core._parameters import parameters
from spinguin._core._hamiltonian import hamiltonian
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

def G0(tensor1: np.ndarray, tensor2: np.ndarray, l: int) -> float:
    """
    Computes the time correlation function at t = 0, G(0), for two
    Cartesian tensors.

    This is the multiplicative factor in front of the exponential
    decay for the isotropic rotational diffusion model.

    Source: Eq. 70 from Hilla & Vaara: Rela2x: Analytic and automatic NMR
    relaxation theory
    https://doi.org/10.1016/j.jmr.2024.107828

    Parameters
    ----------
    tensor1 : ndarray
        Cartesian tensor 1.
    tensor2 : ndarray
        Cartesian tensor 2.
    l : int
        Common rank of the tensors.

    Returns
    -------
    G_0 : float
        Time correlation function evaluated at t = 0.
    """
    # Find the principal axis systems of the tensors
    _, eigvecs1, tensor1_pas = principal_axis_system(tensor1)
    _, eigvecs2, tensor2_pas = principal_axis_system(tensor2)

    # Find the angle between the principal axes
    angle = angle_between_vectors(eigvecs1[0], eigvecs2[0])

    # Write the tensors in the spherical tensor notation
    V1_pas = cartesian_tensor_to_spherical_tensor(tensor1_pas)
    V2_pas = cartesian_tensor_to_spherical_tensor(tensor2_pas)

    # Compute G0
    G_0 = 1 / (2 * l + 1) * eval_legendre(2, np.cos(angle)) * sum(
        [V1_pas[l, q] * np.conj(V2_pas[l, q]) for q in range(-l, l + 1)])

    return G_0

def tau_c_l(tau_c: float, l: int) -> float:
    """
    Calculates the rotational correlation time for a given rank `l`. 
    Applies only for anisotropic rotationally modulated interactions (l > 0).

    Source: Eq. 70 from Hilla & Vaara: Rela2x: Analytic and automatic NMR
    relaxation theory
    https://doi.org/10.1016/j.jmr.2024.107828

    Parameters
    ----------
    tau_c : float
        Rotational correlation time.
    l : int
        Interaction rank.

    Returns
    -------
    t_cl : float
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

############################################################################

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

############################################################################

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

############################################################################

def l2_ellipsoidal_rot_diff_gen_constants(D_1: float, 
                                          D_2: float, 
                                          D_3: float) -> tuple[float, float, float, float]:
    """
    Calculates the rank-2 rotational diffusion generator constants (see Spinguin documentation)
    given the three eigenvalues (principal values) of the rotational diffusion
    tensor for an ellipsoidal diffusor.

    Parameters
    ----------
    D_1 : float
        Rotational diffusion constant along the first principal axis in units of 1/s.
    D_2 : float
        Rotational diffusion constant along the second principal axis in units of 1/s.
    D_3 : float
        Rotational diffusion constant along the third principal axis in units of 1/s.
        
    Returns
    -------
    D: float
        Average rotational diffusion constant.
    R: float
        Rotational diffusion generator parameter R.
    K_p: float
        Rotational diffusion generator parameter K plus.
    K_m: float
        Rotational diffusion generator parameter K minus.
    """
    D = (D_1 + D_2 + D_3) / 3
    R = (D_1*D_2 + D_1*D_3 + D_2*D_3) / 3
    K_p = (np.sqrt(6) * (-2*D + D_1 + D_2 + 2*np.sqrt(D**2 - R)) / (D_1 - D_2))
    K_m = (np.sqrt(6) * (-2*D + D_1 + D_2 - 2*np.sqrt(D**2 - R)) / (D_1 - D_2))

    return D, R, K_p, K_m

def ellipsoidal_rot_diff_gen_eigvals_and_eigvecs(l: int, 
                                                 D_1: float, 
                                                 D_2: float, 
                                                 D_3: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns the eigenvalues and eigenvectors of the rotational diffusion
    generator tensor given the principal values D_1, D_2, and D_3 of the rotational
    diffusion tensor for a given rank `l`.

    Parameters
    ----------
    l : int
        Rank of the tensor undergoing rotational diffusion (1 or 2).
    D_1 : float
        Rotational diffusion constant along the first principal axis in units of 1/s.
    D_2 : float
        Rotational diffusion constant along the second principal axis in units of 1/s.
    D_3 : float
        Rotational diffusion constant along the third principal axis in units of 1/s.

    Returns
    -------
    eigenvalues : ndarray
        Eigenvalues of the rotational diffusion generator tensor.
    eigenvectors : ndarray
        Eigenvectors of the rotational diffusion generator tensor.
    """
    if l == 1:
        # Eigenvalues
        eigenvalue_1 = D_1 + D_2
        eigenvalue_2 = D_1 + D_3
        eigenvalue_3 = D_2 + D_3

        # Eigenvectors
        eigenvector_1 = np.array([0.0, 1.0, 0.0])
        eigenvector_2 = np.array([1.0, 0.0, 1.0])
        eigenvector_3 = np.array([-1.0, 0.0, 1.0])

        # Normalize the eigenvectors
        eigenvector_1 /= np.linalg.norm(eigenvector_1)
        eigenvector_2 /= np.linalg.norm(eigenvector_2)
        eigenvector_3 /= np.linalg.norm(eigenvector_3)

        # Create arrays for eigenvalues and eigenvectors
        eigenvalues = np.array([
            eigenvalue_1,
            eigenvalue_2,
            eigenvalue_3
        ])

        eigenvectors = np.array([
            eigenvector_1,
            eigenvector_2,
            eigenvector_3
        ])

    elif l == 2:
        # Handle the special case where D_1 == D_2 (symmetric top) [or D_2 == D_3 or D_1 == D_3 below]
        if np.isclose(D_1, D_2):

            # Print status message
            print("Symmetric top (D_1 == D_2) detected in ellipsoidal_rot_diff_gen_eigvals_and_eigvecs.")
            
            # Eigenvalues
            eigenvalue_1 = 2 * (D_1 + 2*D_3)
            eigenvalue_2 = 5*D_1 + D_3
            eigenvalue_3 = 6*D_3
            eigenvalue_4 = 5*D_1 + D_3
            eigenvalue_5 = 2 * (D_1 + 2*D_3)

            # Eigenvectors (all already unit norm)
            eigenvector_1 = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
            eigenvector_2 = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
            eigenvector_3 = np.array([0.0, 0.0, 0.0, 0.0, 1.0])
            eigenvector_4 = np.array([0.0, 1.0, 0.0, 0.0, 0.0])
            eigenvector_5 = np.array([0.0, 0.0, 0.0, 1.0, 0.0])

        # Otherwise, general ellipsoid case
        else:
            # Compute diffusor generator constants
            D, R, K_p, K_m = l2_ellipsoidal_rot_diff_gen_constants(D_1, D_2, D_3)

            # Eigenvalues
            eigenvalue_1 = 3 * (4*D - D_1 - D_2)
            eigenvalue_2 = 3 * (D + D_2)
            eigenvalue_3 = 3 * (D + D_1)
            eigenvalue_4 = 6 * (D - np.sqrt(D**2 - R))
            eigenvalue_5 = 6 * (D + np.sqrt(D**2 - R))

            # Eigenvectors
            eigenvector_1 = np.array([-1.0, 0.0, 0.0, 0.0, 1.0])
            eigenvector_2 = np.array([0.0, -1.0, 0.0, 1.0, 0.0])
            eigenvector_3 = np.array([0.0, 1.0, 0.0, 1.0, 0.0])
            eigenvector_4 = np.array([1.0, 0.0, K_m, 0.0, 1.0])
            eigenvector_5 = np.array([1.0, 0.0, K_p, 0.0, 1.0])

            # Normalize the eigenvectors
            eigenvector_1 /= np.linalg.norm(eigenvector_1)
            eigenvector_2 /= np.linalg.norm(eigenvector_2)
            eigenvector_3 /= np.linalg.norm(eigenvector_3)
            eigenvector_4 /= np.linalg.norm(eigenvector_4)
            eigenvector_5 /= np.linalg.norm(eigenvector_5)

        # Create arrays for eigenvalues and eigenvectors
        eigenvalues = np.array([
            eigenvalue_1,
            eigenvalue_2,
            eigenvalue_3,
            eigenvalue_4,
            eigenvalue_5
        ])

        eigenvectors = np.array([
            eigenvector_1,
            eigenvector_2,
            eigenvector_3,
            eigenvector_4,
            eigenvector_5
        ])

    else:
        raise ValueError("Only ranks l = 1 and l = 2 are supported for "
                         "ellipsoidal_rot_diff_gen_eigvals_and_eigvecs.")

    return eigenvalues, eigenvectors

def g_lp_0(tensor1: np.ndarray, 
           tensor2: np.ndarray,
           l: int, 
           p: int,
           rotational_principal_axes: np.ndarray,
           diffusion_generator_eigenvectors: np.ndarray) -> float:
    """
    Computes the p:th component of the time correlation function at t = 0
    in the case of anisotropic rotational diffusion (ellipsoidal diffusor)
    between two Cartesian tensors.

    This is the multiplicative factor in front of each exponential decay term
    for the anisotropic rotational diffusion model.
    (See Spinguin documentation.)

    Parameters
    ----------
    tensor1 : ndarray
        Cartesian tensor 1.
    tensor2 : ndarray
        Cartesian tensor 2.
    l : int
        Common rank of the tensors.
    p : int
        Component index (eigenvalue/eigenvector index of the rotational diffusion tensor).
    rotational_principal_axes : ndarray
        Principal axes of rotation (eigenvectors of the rotational diffusion tensor).
        Assumes that the axes are represented as the columns of the matrix.
    diffusion_generator_eigenvectors : ndarray
        Eigenvectors of the rotational diffusion generator.

    Returns
    -------
    g_lp_0 : float
        p:th component of the time correlation function evaluated at t = 0.
    """
    # Find the rotation matrix that aligns the laboratory frame to the rotational principal axes
    lab_frame = np.eye(3)
    R = rotation_matrix_to_align_axes(lab_frame, rotational_principal_axes.T)

    # Transform the tensors to the rotational principal axes frame
    tensor1_rpa = R @ tensor1 @ R.T
    tensor2_rpa = R @ tensor2 @ R.T

    # Write the tensors in the spherical tensor notation
    V1_rpa = cartesian_tensor_to_spherical_tensor(tensor1_rpa)
    V2_rpa = cartesian_tensor_to_spherical_tensor(tensor2_rpa)

    # Compute two different summations in g_lp_0
    sum1 = sum(V1_rpa[l, m] * diffusion_generator_eigenvectors[m, p] for m in range(-l, l + 1))
    sum2 = sum(np.conj(V2_rpa[l, m] * diffusion_generator_eigenvectors[m, p]) for m in range(-l, l + 1))

    # Compute g_lp_0
    g_lp_0 = (1 / (2 * l + 1)) * sum1 * sum2

    return g_lp_0

###############################################################################

def _process_interactions(spin_system: SpinSystem) -> dict:
    """
    Obtains all interaction tensors from the spin system, and organizes them by
    their interaction rank. Interaction tensors whose norm is below the
    threshold specified in global parameters are disregarded.

    Parameters
    ----------
    spin_system : SpinSystem
        SpinSystem whose interactions are to be processed.

    Returns
    -------
    interactions : dict
        A dictionary where the interactions are organized by rank. The values
        contain all interactions with meaningful strength. The interactions are
        tuples in the format ("interaction", spin_1, spin_2, tensor).
    """
    # Obtain the threshold from the parameters
    zv = parameters.zero_interaction

    # Initialize the lists of interaction descriptions for different ranks
    interactions = {
        1: [],
        2: []
    }

    # Process dipole-dipole couplings
    if spin_system.xyz is not None:

        # Interaction name and rank
        interaction = "DD"
        rank = 2

        # Get the DD-coupling tensors
        dd_tensors = dd_coupling_tensors(spin_system.xyz, spin_system.gammas)

        # Go through the DD-coupling tensors
        for spin_1 in range(spin_system.nspins):
            for spin_2 in range(spin_system.nspins):
                if norm_1(dd_tensors[spin_1, spin_2], ord='row') > zv:
                    interactions[rank].append((
                        interaction,
                        spin_1,
                        spin_2, 
                        dd_tensors[spin_1, spin_2]
                    ))

    # Process nuclear shielding
    if spin_system.shielding is not None:

        # Interaction name
        interaction = "CSA"

        # Get the shielding tensors
        sh_tensors = shielding_intr_tensors(
            spin_system.shielding,
            spin_system.gammas,
            parameters.magnetic_field
        )
        
        # Go through the shielding tensors
        for spin_1 in range(spin_system.nspins):
            if norm_1(sh_tensors[spin_1], ord='row') > zv:

                # Add antisymmetric part if requested
                if spin_system.relaxation.antisymmetric:
                    rank = 1
                    interactions[rank].append(
                        (interaction, spin_1, None, sh_tensors[spin_1])
                    )

                # Always add the symmetric part
                rank = 2
                interactions[rank].append(
                    (interaction, spin_1, None, sh_tensors[spin_1])
                )

    # Process quadrupolar coupling
    if spin_system.efg is not None:

        # Interaction name and rank
        interaction = "Q"
        rank = 2

        # Get the quadrupole coupling tensors
        q_tensors = Q_intr_tensors(
            spin_system.efg,
            spin_system.spins,
            spin_system.quad
        )

        # Go through the quadrupole coupling tensors
        for spin_1 in range(spin_system.nspins):
            if norm_1(q_tensors[spin_1], ord='row') > zv:
                interactions[rank].append(
                    (interaction, spin_1, None, q_tensors[spin_1])
                )

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
        op_def = np.zeros(spin_system.nspins, dtype=int)
        op_def[spin_1] = lq_to_idx(l, q)
        sop = sop_prod(op_def, spin_system.basis.basis, spin_system.spins, 'comm')

    # Two-spin bilinear interaction
    elif interaction_type == "DD":
        sop = sop_T_coupled(spin_system, l, q, spin_1, spin_2)

    # Raise an error for invalid interaction types
    else:
        raise ValueError(f"Invalid interaction type '{interaction_type}' for "
                         "relaxation superoperator. Possible options are " 
                         "'CSA', 'Q', and 'DD'.")

    return sop

def sop_R_redfield_term_isotropic(
    l: int, q: int,
    type_r: str, 
    spin_idx_r1: int, spin_idx_r2: int, 
    tensor_r: np.ndarray,
    top_l_block_shr: dict, top_r_block_shr: dict, bottom_r_block_shr: dict,
    t_max: float, 
    aux_zero: float, relaxation_zero: float,
    sop_Ts_shr: dict, interactions: dict
) -> tuple[int, int, str, int, int, sp.csc_array]:
    """
    Helper function for the Redfield relaxation theory in the case of isotropic diffusion.
    This function calculates one term of the relaxation superoperator and enables the use 
    of parallel computation.

    NOTE: This function returns some of the input parameters to display the
    progress in the computation of the total Redfield relaxation superoperator.

    Parameters
    ----------
    l : int
        Operator rank.
    q : int
        Operator projection.
    type_r : str
        Interaction type. Possible options are "CSA", "Q", and "DD".
    spin_idx_r1 : int
        Index of the first spin in the interaction.
    spin_idx_r2 : int
        Index of the second spin in the interaction. Leave empty for single-spin
        interactions (e.g., CSA).
    tensor_r : np.ndarray
        Interaction tensor for the right-hand interaction.
    top_l_block_shr : dict
        Dictionary containing the shared top left block of the auxiliary matrix.
    top_r_block_shr : dict
        Dictionary containing the shared top right block of the auxiliary
        matrix.
    bottom_r_block_shr : dict
        Dictionary containing the shared bottom right block of the auxiliary
        matrix.
    t_max : float
        Integration limit for the auxiliary matrix method.
    aux_zero : float
        Threshold for the convergence of the Taylor series when exponentiating
        the auxiliary matrix.
    relaxation_zero : float
        Values below this threshold are disregarded in the construction of the
        relaxation superoperator term.
    sop_Ts_shr : dict
        Dictionary containing the shared coupled T superoperators for different
        interactions.
    interactions : dict
        Dictionary containing the interactions organized by rank.

    Returns
    -------
    l : int
        Operator rank.
    q : int
        Operator projection.
    type_r : str
        Interaction type.
    spin_idx_r1 : int
        Index of the first spin.
    spin_idx_r2 : int
        Index of the second spin.
    sop_R_term : csc_array
        Relaxation superoperator term for the given interaction.
    """
    # Create an empty list for the SharedMemory objects
    shms = []

    # Convert the shared arrays back to CSC arrays
    top_l_block, top_l_block_shrm = read_shared_sparse(top_l_block_shr)
    top_r_block, top_r_block_shrm = read_shared_sparse(top_r_block_shr)
    bottom_r_block, bottom_r_block_shrm = read_shared_sparse(bottom_r_block_shr)
    dim = top_r_block.shape[0]

    # Store the SharedMemories
    shms.extend(top_l_block_shrm)
    shms.extend(top_r_block_shrm)
    shms.extend(bottom_r_block_shrm)
    
    # Calculate the Redfield integral using the auxiliary matrix method
    aux_expm = auxiliary_matrix_expm(top_l_block, top_r_block, bottom_r_block, t_max, aux_zero)

    # Extract top left and top right blocks
    aux_top_l = aux_expm[:dim, :dim]
    aux_top_r = aux_expm[:dim, dim:]

    # Extract the Redfield integral
    integral = aux_top_l.conj().T @ aux_top_r

    # Initialize the left coupled T superoperator
    sop_T_l = sp.csc_array((dim, dim), dtype=complex)

    # Iterate over the LEFT interactions
    for interaction_l in interactions[l]:

        # Extract the interaction information
        type_l = interaction_l[0]
        spin_idx_l1 = interaction_l[1]
        spin_idx_l2 = interaction_l[2]
        tensor_l = interaction_l[3]

        # Continue only if T is found (non-zero)
        if (l, q, type_l, spin_idx_l1, spin_idx_l2) in sop_Ts_shr:

            # Compute G0
            G_0 = G0(tensor_l, tensor_r, l)

            # Get the shared T
            sop_T_shared = sop_Ts_shr[(l, q, type_l, spin_idx_l1, spin_idx_l2)]

            # Add current term to the left operator
            sop_T, sop_T_shrm = read_shared_sparse(sop_T_shared)
            sop_T_l += G_0 * sop_T
            shms.extend(sop_T_shrm)

    # Handle negative q values by spherical tensor properties
    if q == 0:
        sop_R_term = sop_T_l.conj().T @ integral
    else:
        sop_R_term = sop_T_l.conj().T @ integral + sop_T_l @ integral.conj().T

    # Eliminate small values
    eliminate_small(sop_R_term, relaxation_zero)
    
    # Close the SharedMemory objects
    for shm in shms:
        shm.close()

    return l, q, type_r, spin_idx_r1, spin_idx_r2, sop_R_term

def sop_R_redfield_term_anisotropic(
    l: int, q: int,
    p: int, 
    rotational_principal_axes: np.ndarray,
    diffusion_generator_eigenvectors: np.ndarray,
    type_r: str, 
    spin_idx_r1: int, spin_idx_r2: int, 
    tensor_r: np.ndarray,
    top_l_block_shr: dict, top_r_block_shr: dict, bottom_r_block_shr: dict,
    t_max: float, 
    aux_zero: float, relaxation_zero: float,
    sop_Ts_shr: dict, interactions: dict
) -> tuple[int, int, str, int, int, sp.csc_array]:
    """
    Helper function for the Redfield relaxation theory in the case of anisotropic diffusion.
    This function calculates one term of the relaxation superoperator and enables the use 
    of parallel computation.

    NOTE: This function returns some of the input parameters to display the
    progress in the computation of the total Redfield relaxation superoperator.

    Parameters
    ----------
    l : int
        Operator rank.
    q : int
        Operator projection.
    p : int
        Index of the rotational diffusion generator eigenvalue/eigenvector.
    rotational_principal_axes : ndarray
        Principal axes of rotation (eigenvectors of the rotational diffusion tensor).
        Assumes that the axes are represented as the columns of the matrix.
    diffusion_generator_eigenvectors : ndarray
        Eigenvectors of the rotational diffusion generator.
    type_r : str
        Interaction type. Possible options are "CSA", "Q", and "DD".
    spin_idx_r1 : int
        Index of the first spin in the interaction.
    spin_idx_r2 : int
        Index of the second spin in the interaction. Leave empty for single-spin
        interactions (e.g., CSA).
    tensor_r : np.ndarray
        Interaction tensor for the right-hand interaction.
    top_l_block_shr : dict
        Dictionary containing the shared top left block of the auxiliary matrix.
    top_r_block_shr : dict
        Dictionary containing the shared top right block of the auxiliary
        matrix.
    bottom_r_block_shr : dict
        Dictionary containing the shared bottom right block of the auxiliary
        matrix.
    t_max : float
        Integration limit for the auxiliary matrix method.
    aux_zero : float
        Threshold for the convergence of the Taylor series when exponentiating
        the auxiliary matrix.
    relaxation_zero : float
        Values below this threshold are disregarded in the construction of the
        relaxation superoperator term.
    sop_Ts_shr : dict
        Dictionary containing the shared coupled T superoperators for different
        interactions.
    interactions : dict
        Dictionary containing the interactions organized by rank.

    Returns
    -------
    l : int
        Operator rank.
    q : int
        Operator projection.
    type_r : str
        Interaction type.
    spin_idx_r1 : int
        Index of the first spin.
    spin_idx_r2 : int
        Index of the second spin.
    sop_R_term : csc_array
        Relaxation superoperator term for the given interaction.
    """
    # Create an empty list for the SharedMemory objects
    shms = []

    # Convert the shared arrays back to CSC arrays
    top_l_block, top_l_block_shrm = read_shared_sparse(top_l_block_shr)
    top_r_block, top_r_block_shrm = read_shared_sparse(top_r_block_shr)
    bottom_r_block, bottom_r_block_shrm = read_shared_sparse(bottom_r_block_shr)
    dim = top_r_block.shape[0]

    # Store the SharedMemories
    shms.extend(top_l_block_shrm)
    shms.extend(top_r_block_shrm)
    shms.extend(bottom_r_block_shrm)

    # Calculate the Redfield integral using the auxiliary matrix method
    aux_expm = auxiliary_matrix_expm(top_l_block, top_r_block, bottom_r_block, t_max, aux_zero)

    # Extract top left and top right blocks
    aux_top_l = aux_expm[:dim, :dim]
    aux_top_r = aux_expm[:dim, dim:]

    # Extract the Redfield integral
    integral = aux_top_l.conj().T @ aux_top_r

    # Initialize the left coupled T superoperator
    sop_T_l = sp.csc_array((dim, dim), dtype=complex)

    # Iterate over the LEFT interactions
    for interaction_l in interactions[l]:

        # Extract the interaction information
        type_l = interaction_l[0]
        spin_idx_l1 = interaction_l[1]
        spin_idx_l2 = interaction_l[2]
        tensor_l = interaction_l[3]

        # Continue only if T is found (non-zero)
        if (l, q, type_l, spin_idx_l1, spin_idx_l2) in sop_Ts_shr:

            # Compute g_lp_0
            g_lp_0_val = g_lp_0(tensor_l, tensor_r, l, p, rotational_principal_axes, diffusion_generator_eigenvectors)

            # Get the shared T
            sop_T_shared = sop_Ts_shr[(l, q, type_l, spin_idx_l1, spin_idx_l2)]

            # Add current term to the left operator
            sop_T, sop_T_shrm = read_shared_sparse(sop_T_shared)
            sop_T_l += g_lp_0_val * sop_T
            shms.extend(sop_T_shrm)

    # Handle negative q values by spherical tensor properties
    if q == 0:
        sop_R_term = sop_T_l.conj().T @ integral
    else:
        sop_R_term = sop_T_l.conj().T @ integral + sop_T_l @ integral.conj().T

    # Eliminate small values
    eliminate_small(sop_R_term, relaxation_zero)

    # Close the SharedMemory objects
    for shm in shms:
        shm.close()

    return l, q, type_r, spin_idx_r1, spin_idx_r2, sop_R_term

def _sop_R_redfield(
    spin_system: SpinSystem,
    masses: np.ndarray = None,
    coords: np.ndarray = None
) -> np.ndarray | sp.csc_array:
    """
    Calculates the relaxation superoperator using Redfield relaxation theory.

    Sources:
    Eq. 54 from Hilla & Vaara: Rela2x: Analytic and automatic NMR relaxation
    theory, https://doi.org/10.1016/j.jmr.2024.107828

    Eq. 24 and 25 from Goodwin & Kuprov: Auxiliary matrix formalism for
    interaction representation transformations, optimal control, and spin
    relaxation theories, https://doi.org/10.1063/1.4928978

    Parameters
    ----------
    spin_system : SpinSystem
        SpinSystem for which the relaxation superoperator is to be calculated.
    masses : ndarray, optional
        A 1-dimensional array specifying the atomic masses of each atom
        in the molecule. Must be given in atomic mass units (u). Required
        only for anisotropic diffusion.
    coords : ndarray, optional
        A 2-dimensional array specifying the cartesian coordinates in
        the XYZ format for each atom in the molecule. Must be given in
        the units of Å. Required only for anisotropic diffusion.

    Returns
    -------
    R : ndarray or csc_array
        Relaxation superoperator.
    """

    time_start = time.time()
    print('Constructing relaxation superoperator using Redfield theory...')

    # Extract information from the spin system
    dim = spin_system.basis.dim
    tau_c = spin_system.relaxation.tau_c
    relative_error = spin_system.relaxation.relative_error

    # Process the interactions
    interactions = _process_interactions(spin_system)

    # Build the coherent Hamiltonian superoperator
    with HidePrints():
        H = hamiltonian(spin_system)

    # Initialize a list to hold all SharedMemories (for parallel processing)
    shms = []

    # Build the top left array of the auxiliary matrix
    top_left = 1j * H
    top_left, top_left_shm = write_shared_sparse(top_left)
    shms.extend(top_left_shm)

    # Print status message about the rotational diffusion model
    if isinstance(tau_c, (int, float)):
        print("Using isotropic rotational diffusion model.")
    elif isinstance(tau_c, (np.ndarray)) and len(tau_c) == 3:
        print("Using anisotropic rotational diffusion model.")

    # FIRST LOOP
    # -- PRECOMPUTE THE COUPLED T SUPEROPERATORS
    # -- CREATE THE LIST OF TASKS
    print("Building superoperators...")
    sop_Ts = {}
    tasks = []

    # Iterate over the ranks
    for l in [1, 2]:

        ### ISOTROPIC CASE
        if isinstance(tau_c, (int, float)):

            # Define the integration limit for the auxiliary matrix method
            t_max = np.log(1 / relative_error) * tau_c

            # Diagonal matrix of correlation time
            tau_c_diagonal_l = 1 / tau_c_l(tau_c, l) * sp.eye_array(dim, format='csc')

            # Bottom right array of auxiliary matrix
            bottom_right = 1j * H - tau_c_diagonal_l
            bottom_right, bottom_right_shm = write_shared_sparse(bottom_right)
            shms.extend(bottom_right_shm)

            # Iterate over the projections (negative q values are handled by
            # spherical tensor properties)
            for q in range(0, l + 1):

                # Iterate over the interactions
                for interaction in interactions[l]:

                    # Extract the interaction information
                    itype = interaction[0]
                    spin1 = interaction[1]
                    spin2 = interaction[2]
                    tensor = interaction[3]

                    # Show current status
                    if spin2 is None:
                        print(f"l: {l}, q: {q} - {itype} for spin {spin1}")
                    else:
                        print(f"l: {l}, q: {q} - {itype} for spins {spin1}-{spin2}")

                    # Compute the coupled T superoperator
                    sop_T = _get_sop_T(spin_system, l, q, itype, spin1, spin2)

                    # Continue only if T is not empty
                    if sop_T.nnz != 0:

                        # Make a shared version of the coupled T superoperator
                        sop_T_shared, sop_T_shm = write_shared_sparse(sop_T)
                        sop_Ts[(l, q, itype, spin1, spin2)] = sop_T_shared
                        shms.extend(sop_T_shm)

                        # Add to the list of tasks
                        tasks.append((
                            l,                          # Interaction rank
                            q,                          # Interaction projection
                            itype,                      # Interaction type
                            spin1,                      # Right interaction spin 1
                            spin2,                      # Right interaction spin 2
                            tensor,                     # Right interaction tensor
                            top_left,                   # Aux matrix top left
                            sop_T_shared,               # Aux matrix top right
                            bottom_right,               # Aux matrix bottom right
                            t_max,                      # Aux matrix integral limit
                            parameters.zero_aux,        # Aux matrix expm zero
                            parameters.zero_relaxation, # Relaxation zero element
                            sop_Ts,                     # All coupled T
                            interactions                # Left interaction
                        ))

        ### ANISOTROPIC CASE
        elif isinstance(tau_c, (np.ndarray)) and len(tau_c) == 3:
            # Ensure masses and coords are provided
            if masses is None or coords is None:
                raise ValueError("Masses and coordinates must be provided for anisotropic diffusion.")
            
            # Get the rotational diffusion tensor eigenvalues and eigenvectors
            D1, D2, D3 = 1 / (tau_c * l * (l + 1))

            # Compute the eigenvalues and eigenvectors
            diffusion_eigenvalues, diffusion_eigenvectors = \
                ellipsoidal_rot_diff_gen_eigvals_and_eigvecs(l, D1, D2, D3)

            # Calculate the rotational principal axes
            moi_tensor = moment_of_inertia_tensor(masses, coords)
            _, rot_principal_axes = np.linalg.eigh(moi_tensor)

            # Define the integration limit for the auxiliary matrix method
            t_max = np.log(1 / relative_error) * np.max(1 / diffusion_eigenvalues)

            # Iterate over the diffusion tensor eigenvalues/eigenvectors
            for p in range(2 * l + 1):

                # Diagonal matrix of correlation time corresponding to eigenvalue p
                tau_c_diagonal_lp = diffusion_eigenvalues[p] * sp.eye_array(dim, format='csc')

                # Bottom right array of auxiliary matrix
                bottom_right = 1j * H - tau_c_diagonal_lp
                bottom_right, bottom_right_shm = write_shared_sparse(bottom_right)
                shms.extend(bottom_right_shm)

                # Iterate over the projections (negative q values are handled by
                # spherical tensor properties)
                for q in range(0, l + 1):

                    # Iterate over the interactions
                    for interaction in interactions[l]:

                        # Extract the interaction information
                        itype = interaction[0]
                        spin1 = interaction[1]
                        spin2 = interaction[2]
                        tensor = interaction[3]

                        # Show current status
                        if spin2 is None:
                            print(f"l: {l}, q: {q} - {itype} for spin {spin1}")
                        else:
                            print(f"l: {l}, q: {q} - {itype} for spins {spin1}-{spin2}")

                        # Compute the coupled T superoperator
                        sop_T = _get_sop_T(spin_system, l, q, itype, spin1, spin2)

                        # Continue only if T is not empty
                        if sop_T.nnz != 0:

                            # Make a shared version of the coupled T superoperator
                            sop_T_shared, sop_T_shm = write_shared_sparse(sop_T)
                            sop_Ts[(l, q, itype, spin1, spin2)] = sop_T_shared
                            shms.extend(sop_T_shm)

                            # Add to the list of tasks
                            tasks.append((
                                l,                          # Interaction rank
                                q,                          # Interaction projection
                                p,                          # Diffusion eigenvalue/eigenvector index
                                rot_principal_axes,         # Rotational principal axes
                                diffusion_eigenvectors,     # Diffusion eigenvectors
                                itype,                      # Interaction type
                                spin1,                      # Right interaction spin 1
                                spin2,                      # Right interaction spin 2
                                tensor,                     # Right interaction tensor
                                top_left,                   # Aux matrix top left
                                sop_T_shared,               # Aux matrix top right
                                bottom_right,               # Aux matrix bottom right
                                t_max,                      # Aux matrix integral limit
                                parameters.zero_aux,        # Aux matrix expm zero
                                parameters.zero_relaxation, # Relaxation zero element
                                sop_Ts,                     # All coupled T
                                interactions                # Left interaction
                            ))

        else:
            raise ValueError("Invalid tau_c parameter for Redfield relaxation "
                             "theory. Must be either a single value (isotropic "
                             "diffusion) or three values (anisotropic diffusion).")

    # Initialize the relaxation superoperator
    R = sp.csc_array((dim, dim), dtype=complex)

    # SECOND LOOP -- Iterate over the tasks in parallel and build the R
    if dim > parameters.parallel_dim:
        print("Performing the Redfield integrals in parallel...")

        # Create the parallel tasks (depending on isotropic/anisotropic)
        parallel = Parallel(n_jobs=-1, return_as="generator_unordered")
        if isinstance(tau_c, (int, float)):
            output_generator = parallel(
                delayed(sop_R_redfield_term_isotropic)(*task) for task in tasks
            )
        elif isinstance(tau_c, (np.ndarray)) and len(tau_c) == 3:
            output_generator = parallel(
                delayed(sop_R_redfield_term_anisotropic)(*task) for task in tasks
            )
        else:
            raise ValueError("Invalid tau_c parameter for Redfield relaxation "
                             "theory. Must be either a single value (isotropic "
                             "diffusion) or three values (anisotropic diffusion).")

        # Process the results from parallel processing
        for result in output_generator:

            # Parse the result and add term to total relaxation superoperator
            l, q, itype, spin1, spin2, R_term = result
            R += R_term

            # Show current status
            if spin2 is None:
                print(f"l: {l}, q: {q} - {itype} for spin {spin1}")
            else:
                print(f"l: {l}, q: {q} - {itype} for spins {spin1}-{spin2}")

    # SECOND LOOP -- Iterate over the tasks in serial
    else:
        print("Performing the Redfield integrals in serial...")

        # Process the tasks in serial
        for task in tasks:

            # Parse the result and add term to total relaxation superoperator (depending on isotropic/anisotropic)
            if isinstance(tau_c, (int, float)):
                l, q, itype, spin1, spin2, R_term = \
                    sop_R_redfield_term_isotropic(*task)
            elif isinstance(tau_c, (np.ndarray)) and len(tau_c) == 3:
                l, q, itype, spin1, spin2, R_term = \
                    sop_R_redfield_term_anisotropic(*task)
            else:
                raise ValueError("Invalid tau_c parameter for Redfield relaxation "
                                 "theory. Must be either a single value (isotropic "
                                 "diffusion) or three values (anisotropic diffusion).")
            R += R_term

            # Show current status
            if spin2 is None:
                print(f"l: {l}, q: {q} - {itype} for spin {spin1}")
            else:
                print(f"l: {l}, q: {q} - {itype} for spins {spin1}-{spin2}")

    # Clear the shared memories
    for shm in shms:
        shm.close()
        shm.unlink()

    print("Redfield integrals completed.")

    # Return only real values unless dynamic frequency shifts are requested
    if not spin_system.relaxation.dynamic_frequency_shift:
        print("Removing the dynamic frequency shifts...")
        R = R.real
        print("Dynamic frequency shifts removed.")

    # Eliminate small values
    print("Eliminating small values from the relaxation superoperator...")
    eliminate_small(R, parameters.zero_relaxation)
    print("Small values eliminated.")

    print("Redfield relaxation superoperator constructed in "
            f"{time.time() - time_start:.4f} seconds.")
    print()

    return R

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
    print('Constructing the phenomenological relaxation superoperator...')

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

    print("Phenomenological relaxation superoperator constructed in "
          f"{time.time() - time_start:.4f} seconds.")
    print()

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

    print("Processing scalar relaxation of the second kind...")
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

    print(f"SR2K superoperator constructed in {time.time() - time_start:.4f} "
          "seconds.")
    print()
    
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
    print("Applying thermalization to the relaxation superoperator...")
    time_start = time.time()

    # Get the matrix exponential corresponding to the Boltzmann distribution
    with HidePrints():
        P = expm(const.hbar/(const.k*T)*H_left, parameters.zero_thermalization)

    # Calculate the thermalized relaxation superoperator
    R = R @ P

    print(f"Thermalization applied in {time.time() - time_start:.4f} seconds.")
    print()

    return R

def relaxation(spin_system: SpinSystem, 
               masses: np.ndarray = None, 
               coords: np.ndarray = None) -> np.ndarray | sp.csc_array:
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
    masses : ndarray, optional
        A 1-dimensional array specifying the atomic masses of each atom
        in the molecule. Must be given in atomic mass units (u). Required
        only for anisotropic diffusion.
    coords : ndarray, optional
        A 2-dimensional array specifying the cartesian coordinates in
        the XYZ format for each atom in the molecule. Must be given in
        the units of Å. Required only for anisotropic diffusion.

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
        R = _sop_R_redfield(spin_system, masses=masses, coords=coords)
    
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