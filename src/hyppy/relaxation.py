"""
relaxation.py

This module provides functions for calculating the relaxation superoperators.
"""

# For referencing the SpinSystem class
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hyppy.spin_system import SpinSystem

# Imports
import time
import numpy as np
import scipy.constants as const
from scipy.sparse import csc_array, eye_array, lil_array
from scipy.special import eval_legendre
from hyppy import la, operators
from hyppy.basis import idx_to_lq, lq_to_idx, str_to_op_def, state_idx
from hyppy.hamiltonian import hamiltonian_zeeman

def dd_constant(y1: float, y2: float) -> float:
    """
    Calculates the dipole-dipole coupling constant (without the distance).

    Parameters
    ----------
    y1 : float
        Gyromagnetic ratio of the first spin in the units of rad/s/T.
    y2 : float
        Gyromagnetic ratio of the second spin in the units of rad/s/T.

    Returns
    -------
    dd_const : float
        Dipole-dipole coupling constant in rhe units of rad/s * m^3.
    """

    # Calculate the constant
    dd_const = -const.mu_0 / (4 * np.pi) * y1 * y2 * const.hbar

    return dd_const

def Q_constant(S: float, Q_moment: float) -> float:
    """
    Nuclear quadrupolar coupling constant in (rad/s) / (V/m^2).
    
    Parameters
    ----------
    S : float
        Spin quantum number.
    Q_moment : float
        Nuclear quadrupole moment (in units of m2)

    Returns
    -------
    Q_const : float
        Quadrupolar coupling constant.
    """

    # Calculate the quadrupolar coupling constant
    if (S >= 1) and (Q_moment > 0):
        Q_const = -const.e * Q_moment / const.hbar / (2*S * (2*S - 1))
    else:
        Q_const = 0
    
    return Q_const

def G0(tensor1: np.ndarray, tensor2: np.ndarray, l:int) -> float:
    """
    Computes the time correlation function at t = 0, G(0), for two
    Cartesian tensors.

    This is the multiplicative factor in front of the exponential
    decay for the isotropic rotational diffusion model.

    Source: Eq. 70 from Hilla & Vaara: Rel2x: Analytic and automatic NMR relaxation theory
    https://doi.org/10.1016/j.jmr.2024.107828

    Parameters
    ----------
    tensor1 : numpy.ndarray
        Cartesian tensor 1.
    tensor2 : numpy.ndarray
        Cartesian tensor 2.
    l : int
        Common rank of the tensors

    Returns
    -------
    G_0 : float
        Time correlation function evaluated at t = 0.
    """
    # Find the principal axis systems of the tensors
    _, eigvecs1, tensor1_pas = la.principal_axis_system(tensor1)
    _, eigvecs2, tensor2_pas = la.principal_axis_system(tensor2)

    # Find the angle between the principal axes
    angle = la.angle_between_vectors(eigvecs1[0], eigvecs2[0])

    # Write the tensors in the spherical tensor notation
    V1_pas = la.cartesian_tensor_to_spherical_tensor(tensor1_pas)
    V2_pas = la.cartesian_tensor_to_spherical_tensor(tensor2_pas)

    # Compute G0
    G_0 = 1 / (2*l+1) * eval_legendre(2, np.cos(angle)) * sum([V1_pas[l, q] * np.conj(V2_pas[l, q]) for q in range(-l, l+1)])

    return G_0

def tau_c_l(tau_c: float, l: int) -> float:
    """
    Rotational correlation time for a given rank `l`. Applies only for anisotropic
    rotationally modulated interactions (l > 0).

    Source: Eq. 70 from Hilla & Vaara: Rel2x: Analytic and automatic NMR relaxation theory
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
        t_cl = 6*tau_c / (l*(l+1))

    # For isotropic interactions raise an error
    else:
        raise ValueError('Rank l must be different from 0 in tau_c_l.')
    
    return t_cl
    
def dd_coupling_tensors(spin_system:SpinSystem) -> np.ndarray:
    """
    Calculates the dipole-dipole coupling tensor between all nuclei
    in the spin system.

    Parameters
    ----------
    spin_system : SpinSystem

    Returns
    -------
    dd_tensors : numpy.ndarray
        Array of dimensions (N, N, 3, 3) containing the 3x3 tensors
        between all nuclei.
    """

    # Extract the necessary information from the spin system
    xyz = spin_system.xyz
    size = spin_system.size
    gammas = spin_system.gammas

    # Get the connector and distance arrays
    connectors = xyz[:, np.newaxis] - xyz
    distances = np.linalg.norm(connectors, axis=2)

    # Initialize the array of tensors
    dd_tensors = np.zeros((size, size, 3, 3))

    # Go through each spin pair
    for i in range(size):
        for j in range(size):

            # Only the lower triangular part is computed
            if i > j:
                rr = np.outer(connectors[i, j], connectors[i, j])
                dd_tensors[i, j] = 1e30 * dd_constant(gammas[i], gammas[j]) * (3*rr - distances[i, j]**2*np.eye(3)) / distances[i, j]**5

    return dd_tensors

def shielding_intr_tensors(spin_system:SpinSystem, B: float) -> np.ndarray:
    """
    Calculates the shielding interaction tensors for a spin system.

    Parameters
    ----------
    spin_system : SpinSystem
    B : float
        External magnetic field in the units of T.

    Returns
    -------
    shielding_tensors: numpy.ndarray
        Array of shielding tensors.
    """

    # Extract the necessary information from spin system
    shielding = spin_system.shielding
    gammas = spin_system.gammas

    # Convert from ppm to dimensionless
    shielding_tensors = shielding * 1e-6

    # Create Larmor frequencies ("shielding constants" for relaxation)
    w0s = -gammas * B

    # Multiply with the Larmor frequencies
    for i, val in enumerate(w0s):
        shielding_tensors[i] *= val

    return shielding_tensors

def Q_intr_tensors(spin_system:SpinSystem) -> np.ndarray:
    """
    Calculates the quadrupolar interaction tensors for a spin system.

    Parameters
    ----------
    spin_system : SpinSystem
        
    Returns
    -------
    Q_tensors: numpy.ndarray
        Quadrupolar interaction tensors.
    """

    # Extract necessary information from spin system
    efg = spin_system.efg
    quad = spin_system.quad
    spins = spin_system.spins

    # Convert from a.u. to V/m2, and fix Turbomole sign
    Q_tensors = -9.7173624292e21 * efg

    # Create quadrupolar coupling constants
    Q_constants = [Q_constant(S, Q) for S, Q in zip(spins, quad)]

    # Multiply the tensors with the quadrupolar coupling constants
    for i, val in enumerate(Q_constants):
        Q_tensors[i] *= val

    return Q_tensors

def interactions(spin_system:SpinSystem, intrs: dict, zero_intr: float=1e-9) -> dict:
    """
    Processes all the incoherent interactions and sorts them by rank. Small interactions
    are disregarded.

    Parameters
    ----------
    spin_system : SpinSystem
    intrs : dict
        A dictionary where the keys represent the interaction, and the values contain the
        interaction tensors and the ranks.
    zero_intr : float
        Default: 1e-9. If the row-wise 1-norm of the interaction tensor (upper limit for
        its eigenvalues) is smaller than the threshold, the interaction is disregarded.

    Returns
    -------
    interactions : dict
        A dictionary where the interactions are sorted by the rank. The values contain all
        the interactions with meaningful strength. Thee interactions are are tuples and have
        the format ("interaction", spin_1, spin_2).
    """

    # Extract the necessary information from the spin system
    size = spin_system.size

    # Initialize the lists of interaction descriptions for different ranks
    interactions = {
        1 : [],
        2 : []
    }

    # Go through the interactions
    for interaction, properties in intrs.items():

        # Extract the properties
        tensors = properties[0]
        ranks = properties[1]

        # Go through the ranks
        for rank in ranks:

            # Consider single-spin interactions
            if interaction == "CSA" or interaction == "Q":
                for spin_1 in range(size):
                    if la.norm_1(tensors[spin_1], ord='row') > zero_intr:
                        interactions[rank].append((interaction, spin_1, None, tensors[spin_1]))

            # Consider two-spin interactions
            if interaction == "DD":
                for spin_1 in range(size):
                    for spin_2 in range(size):
                        if la.norm_1(tensors[spin_1, spin_2], ord='row') > zero_intr:
                            interactions[rank].append((interaction, spin_1, spin_2, tensors[spin_1, spin_2]))

    return interactions
    
def sop_T(spin_system:SpinSystem, l: int, q: int, interaction_type: str, spin_1: int, spin_2: int=None) -> csc_array:
    """
    Helper function for the relaxation module. Calculates the coupled product
    superoperators for different interaction types.

    Parameters
    ----------
    spin_system : SpinSystem
    l : int
        Operator rank.
    q : int
        Operator projection.
    interaction_type : str
        Describes the interaction. The possible options are "CSA", "Q" and "DD", which stand
        for chemical shift anisotropy, quadrupolar coupling, and dipole-dipole coupling.
    spin_1 : int
        Index of the first spin.
    spin_2 : int
        Index of the second spin. Left empty for linear single-spin interactions (CSA).

    Returns
    -------
    sop : csc_array
        Coupled spherical tensor superoperator of rank `l` and projection `q`.
    """

    # Extract the necessary information from spin system
    size = spin_system.size

    # Single-spin linear interaction
    if interaction_type == "CSA":
        sop = operators.sop_T_coupled(spin_system, l, q, spin_1)

    # Single-spin quadratic interaction
    elif interaction_type == "Q":
        op_def = np.zeros(size, dtype=int)
        op_def[spin_1] = lq_to_idx(l, q)
        op_def = tuple(op_def)
        sop = operators.sop_P(spin_system, op_def, 'comm')

    # Two-spin bilinear interaction
    elif interaction_type == "DD":
        sop = operators.sop_T_coupled(spin_system, l, q, spin_1, spin_2)

    # Otherwise incorrect interaction type
    else:
        raise ValueError(f"Incorrect interaction type: {interaction_type}")

    return sop

def sop_R_redfield(spin_system: SpinSystem,
                    sop_H: csc_array,
                    interactions: dict,
                    tau_c: float,
                    relative_error: float=1e-6,
                    zero_aux: float=1e-18) -> csc_array:
    """
    Calculates the relaxation superoperator using the Redfield relaxation theory.

    Sources:
    
    Eq. 54 from Hilla & Vaara: Rel2x: Analytic and automatic NMR relaxation theory
    https://doi.org/10.1016/j.jmr.2024.107828

    Eq. 24 and 25 from Goodwin & Kuprov: Auxiliary matrix formalism for interaction
    representation transformations, optimal control, and spin relaxation theories
    https://doi.org/10.1063/1.4928978

    Parameters
    ----------
    spin_system : SpinSystem
    sop_H : csc_array
        Coherent part of the Hamiltonian superoperator.
    interactions : dict
        Interactions sorted by the rank. Within each rank, contains a list, where
        the interactions are given in the format ("interaction", spin_1, spin_2, tensor).
    tau_c : float
        Isotropic rotational correlation time in the units of s.
    relative_error : float
        Default: 1e-6. Relative error for the integration in the auxiliary matrix method.
    zero_aux : float
        Default: 1e-18. Values below threshold are considered to be zero in the auxiliary
        matrix method. If extremely short correlation times are used, consider tightening
        this parameter.

    Returns
    -------
    sop_R : csc_array
        Relaxation superoperator.
    """

    # Extract the necessary information from the spin system
    dim = spin_system.basis.dim

    # Initialize relaxation superoperator
    sop_R = csc_array((dim, dim), dtype=complex)

    # Define the integration limit for the auxiliary matrix method
    t_max = np.log(1/relative_error) * tau_c

    # Diagonal matrices of correlation times
    tau_c_diagonal_l1 = 1/tau_c_l(tau_c, 1) * eye_array(sop_H.shape[0], format='csc')
    tau_c_diagonal_l2 = 1/tau_c_l(tau_c, 2) * eye_array(sop_H.shape[0], format='csc')
    
    # Auxiliary matrix arrays
    top_left = 1j*sop_H
    bottom_right_l1 = 1j*sop_H - tau_c_diagonal_l1
    bottom_right_l2 = 1j*sop_H - tau_c_diagonal_l2

    # Loop over the ranks
    for l in [1, 2]:

        # Loop over the projetions (negative q values calculated with spherical tensor properties)
        for q in range(0, l+1):

            print(f"Processing l:{l}, q:{q}")

            # Loop over the RIGHT interactions
            for interaction_right in interactions[l]:
                
                # Extract the interaction information
                type_right = interaction_right[0]
                spin_right1 = interaction_right[1]
                spin_right2 = interaction_right[2]
                tensor_right = interaction_right[3]

                print(f"{type_right}: {spin_right1}-{spin_right2}")

                # Compute the right operator
                sop_T_right = sop_T(spin_system, l, q, type_right, spin_right1, spin_right2)

                # Calculate the Redfield integral using the auxiliary matrix method
                if l == 1:
                    sop_T_right = la.auxiliary_matrix_expm(top_left, sop_T_right, bottom_right_l1, t_max, zero_aux)
                elif l == 2:
                    sop_T_right = la.auxiliary_matrix_expm(top_left, sop_T_right, bottom_right_l2, t_max, zero_aux)

                # Get top left and top right blocks
                top_left = sop_T_right[:dim, :dim]
                top_right = sop_T_right[:dim, dim:]

                # Compute the relaxation superoperator term
                sop_T_right = top_left.conj().T @ top_right

                # Initialize the left operator
                sop_T_left = csc_array((dim, dim), dtype=complex)

                # Loop over the LEFT interactions
                for interaction_left in interactions[l]:

                    # Extract the interaction information
                    type_left = interaction_left[0]
                    spin_left1 = interaction_left[1]
                    spin_left2 = interaction_left[2]
                    tensor_left = interaction_left[3]

                    # Compute G0
                    G_0 = G0(tensor_left, tensor_right, l)

                    # Add to the left operator
                    sop_T_left += G_0 * sop_T(spin_system, l, q, type_left, spin_left1, spin_left2)

                # Add to the total relaxation superoperator term
                if q == 0:
                    sop_R += sop_T_left.conj().T @ sop_T_right
                else:
                    sop_R += sop_T_left.conj().T @ sop_T_right + sop_T_left @ sop_T_right.conj().T

    return sop_R

def relaxation_weighted(spin_system:SpinSystem, R_1: np.ndarray, R_2: np.ndarray) -> csc_array:
    """
    Builds the relaxation superoperator from given R_1 and R_2 values
    for each spin.

    Parameters
    ----------
    spin_system : SpinSystem
    R_1 : numpy.ndarray
        Longitudinal relaxation rates in 1/s for each spin.
        For example: np.array([1.0, 2.0, 2.5])
    R_2 : numpy.ndarray
        Transverse relaxation rates in 1/s for each spin.

    Returns
    -------
    sop_R : csc_array
        Relaxation superoperator.
    """

    # Extract the necessary information from spin system
    dim = spin_system.basis.dim
    basis_dict = spin_system.basis.dict

    # Create an empty array for the relaxation superoperator
    sop_R = lil_array((dim, dim))

    # Loop over the basis set
    for state, idx in basis_dict.items():

        # Initialize relaxation rate for the current state
        R_state = 0
        
        # Loop over the state
        for spin, operator in enumerate(state):

            # Continue only if operator is not unit state
            if operator != 0:

                # Get the projection of the state
                _, q = idx_to_lq(operator)
            
                # Check if current spin has longitudinal state
                if q==0:
                    
                    # Add to the relaxation rate
                    R_state += R_1[spin]

                # If not, then state must be transverse
                else:

                    # Add to the relaxation rate
                    R_state += R_2[spin]

        # Add to relaxation matrix
        sop_R[idx, idx] = R_state

    # Convert to CSC array
    sop_R = sop_R.tocsc()

    return sop_R

def sr2k(spin_system:SpinSystem, sop_R: csc_array, B: float) -> csc_array:
    """
    Calculates the scalar relaxation of the second kind based on the
    Abragam's formula and adds it to the relaxation superoperator.

    Parameters
    ----------
    spin_system : SpinSystem
    sop_R : csc_array
        Relaxation superoperator.
    B : float
        Magnetic field in the units of T.

    Returns
    -------
    sop_R : csc_array
        Relaxation superoperator with SR2K.
    """

    print("Processing scalar relaxation of the second kind.")

    # Extract the necessary information from spin system
    gammas = spin_system.gammas
    chemical_shifts = spin_system.chemical_shifts
    scalar_couplings = spin_system.scalar_couplings
    size = spin_system.size
    spins = spin_system.spins
    isotopes = spin_system.isotopes

    # Assign empty arrays for the relaxation rates
    R_1 = np.zeros(size)
    R_2 = np.zeros(size)

    # Make an empty list for the quadrupolar nuclei
    quadrupolar = []
    
    # Loop over all of the spins
    for i, spin in enumerate(spins):

        # Append list if quadrupolar
        if spin > 0.5:
            quadrupolar.append(i)
    
    # Loop over the quadrupolar nuclei
    for quad in quadrupolar:

        # Find the indices of the longitudinal and transverse state
        op_def_z, _ = str_to_op_def(spin_system, ['I_z'], [quad])
        op_def_p, _ = str_to_op_def(spin_system, ['I_+'], [quad])
        idx_long = state_idx(spin_system, op_def_z[0])
        idx_trans = state_idx(spin_system, op_def_p[0])

        # Find the relaxation times of the quadrupolar nucleus
        T_1 = 1 / sop_R[idx_long, idx_long]
        T_2 = 1 / sop_R[idx_trans, idx_trans]

        # Convert to real
        T_1 = np.real(T_1)
        T_2 = np.real(T_2)

        # Find Larmor frequency of the quadrupolar nucleus
        omega_quad = gammas[quad] * B * (1 + chemical_shifts[quad]*1e-6)

        # Find the spin of the quadrupolar nucleus
        S = spins[quad]

        # Loop over all spins
        for target, isotope in enumerate(isotopes):

            # Proceed only if different isotope
            if not isotopes[quad] == isotope:

                # Find Larmor frequency of the target spin
                omega_target = gammas[target] * B * (1 + chemical_shifts[target]*1e-6)

                # Find scalar coupling between spins in rad/s
                J = 2*np.pi*scalar_couplings[quad][target]

                # Calculate the relaxation rates
                R_1[target] += ((J**2) * S*(S+1))/3 * (2*T_2) / (1 + (omega_target-omega_quad)**2 * T_2**2)
                R_2[target] += ((J**2) * S*(S+1))/3 * (T_1 + (T_2 / (1 + (omega_target-omega_quad)**2 * T_2**2)))
    
    # Get relaxation superoperator corresponding to SR2K
    sop_R_sr2k = relaxation_weighted(spin_system, R_1, R_2)

    # Add to relaxation superoperator
    sop_R += sop_R_sr2k
    
    return sop_R

def relaxation(spin_system: SpinSystem, 
               sop_H: csc_array,
               B: float,
               tau_c: float,
               include_sr2k: bool=False,
               relative_error: float=1e-6,
               real_only: bool=True,
               zero_aux: float=1e-18,
               zero_R: float=1e-12,
               zero_intr: float=1e-9,
               antisymmetric: bool=False) -> csc_array:
    """
    Calculates the relaxation superoperator.

    Parameters
    ----------
    spin_system : SpinSystem
    sop_H : csc_array
        Hamiltonian superoperator (coherent part).
    B : float
        Magnetic field in the units of T.
    tau_c : float
        Isotropic rotational correlation time in the units of s.
    include_sr2k : bool
        Decide whether scalar relaxation of the second kind is taken into account.
        Applies only for spin systems with quadrupolar nuclei.
    relative_error : float
        Default: 1e-6. Relative error for the integration in the auxiliary matrix method.
        Can be increased to make the integration faster.
    real_only : bool
        Default: True. Decide whether to return the imaginary component (dynamic frequency shift).
    zero_aux : float
        Default: 1e-18. Values below zero_aux are considered to be zero in the auxiliary
        matrix method. Significantly affects performance. If extremely short correlation
        times are used, consider tightening this parameter.
    zero_R : float
        Default: 1e-12. Relaxation superoperator is cleaned from values smaller than zero_R
        before it is returned.
    zero_intr : float
        Default: 1e-9. Interaction is disregarded if the inifnity-norm of the interaction tensor
        (upper limit for largest eigenvalue) is smaller than the threshold.
    antisymmetric : bool
        Default: False. Can be used to include the rank-1 components for the CSA.

    Returns
    -------
    sop_R : csc_array
        Relaxation superoperator.
    """

    time_start = time.time()
    print("Starting to construct the relaxation superoperator.")

    # Get the incoherent interactions
    dd_tensors = dd_coupling_tensors(spin_system)
    sh_tensors = shielding_intr_tensors(spin_system, B)
    q_tensors = Q_intr_tensors(spin_system)

    # Choose the interaction ranks
    dd_ranks = [2]
    sh_ranks = [2]
    q_ranks = [2]
    if antisymmetric:
        sh_ranks = [1, 2]

    # Generate default incoherent interactions dictionary
    intrs = {
        'DD':  (dd_tensors, dd_ranks),
        'CSA': (sh_tensors, sh_ranks),
        'Q':   (q_tensors, q_ranks)
    }
    
    # Remove small interactions and re-organize them rank-wise
    intrs = interactions(spin_system, intrs, zero_intr)
    
    # Calculate R using the Redfield theory
    sop_R = sop_R_redfield(spin_system, sop_H, intrs, tau_c, relative_error, zero_aux)

    # Process sr2k if enabled
    if include_sr2k:
        sop_R = sr2k(spin_system, sop_R, B)

    # Return real values only if requested
    if real_only:
        sop_R = sop_R.real

    # Remove small elements from the relaxation superoperator
    la.increase_sparsity(sop_R, zero_R)

    print("Relaxation superoperator constructed.")
    print(f"Elapsed time: {time.time() - time_start} seconds.")

    return sop_R

def thermalize(spin_system:SpinSystem, R: csc_array, B: float, T: float, zero_value: float=1e-18) -> csc_array:
    """
    Applies the Levitt - Di Bari thermalization to the relaxation superoperator.

    Parameters
    ----------
    spin_system : SpinSystem
    R : csc_array
        Relaxation superoperator to be thermalized.
    B : float
        Magnetic field in the units of T.
    T : float
        Temperature in the units of K.
    zero_value : float
        Default: 1e-18. Used to estimate the convergence of matrix exponential.

    Returns
    -------
    R : csc_array
        Thermalized Liouvillian.
    """

    # Build the left Zeeman Hamiltonian
    H = hamiltonian_zeeman(spin_system, B, 'left')

    # Get the matrix exponential corresponding to Boltzmann distribution
    P = la.expm(const.hbar/(const.k*T)*H, zero_value)

    # Calculate the thermalized relaxation superoperator
    R = R @ P

    return R