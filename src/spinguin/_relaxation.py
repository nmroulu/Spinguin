"""
relaxation.py

This module provides functions for calculating relaxation superoperators.

TODO: Random field malli relaksaatioon.
"""

# For referencing the SpinSystem class
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from spinguin._spin_system import SpinSystem

# Imports
import time
import numpy as np
import scipy.constants as const
from scipy.sparse import csc_array, eye_array, lil_array
from scipy.special import eval_legendre
from spinguin import _operators
from spinguin._la import increase_sparsity, principal_axis_system, cartesian_tensor_to_spherical_tensor,\
                         angle_between_vectors, norm_1, auxiliary_matrix_expm, expm
from spinguin._basis import idx_to_lq, lq_to_idx, parse_operator_string, state_idx
from spinguin._hamiltonian import hamiltonian
from spinguin._settings import Settings

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

    Source: Eq. 70 from Hilla & Vaara: Rela2x: Analytic and automatic NMR relaxation theory
    https://doi.org/10.1016/j.jmr.2024.107828

    Parameters
    ----------
    tensor1 : numpy.ndarray
        Cartesian tensor 1.
    tensor2 : numpy.ndarray
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
    G_0 = 1 / (2 * l + 1) * eval_legendre(2, np.cos(angle)) * sum([V1_pas[l, q] * np.conj(V2_pas[l, q]) for q in range(-l, l + 1)])

    return G_0

def tau_c_l(tau_c: float, l: int) -> float:
    """
    Calculates the rotational correlation time for a given rank `l`. 
    Applies only for anisotropic rotationally modulated interactions (l > 0).

    Source: Eq. 70 from Hilla & Vaara: Rela2x: Analytic and automatic NMR relaxation theory
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
    
def dd_coupling_tensors(spin_system: SpinSystem) -> np.ndarray:
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
                dd_tensors[i, j] = 1e30 * dd_constant(gammas[i], gammas[j]) * (3 * rr - distances[i, j]**2 * np.eye(3)) / distances[i, j]**5

    return dd_tensors

def shielding_intr_tensors(spin_system: SpinSystem, B: float) -> np.ndarray:
    """
    Calculates the shielding interaction tensors for a spin system.

    Parameters
    ----------
    spin_system : SpinSystem
    B : float
        External magnetic field in units of T.

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
    # TODO: Check the sign of the Larmor frequency
    w0s = -gammas * B

    # Multiply with the Larmor frequencies
    for i, val in enumerate(w0s):
        shielding_tensors[i] *= val

    return shielding_tensors

# TODO: Check the sign
def Q_intr_tensors(spin_system: SpinSystem) -> np.ndarray:
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

    # Convert from a.u. to V/m^2
    Q_tensors = -9.7173624292e21 * efg

    # Create quadrupolar coupling constants
    Q_constants = [Q_constant(S, Q) for S, Q in zip(spins, quad)]

    # Multiply the tensors with the quadrupolar coupling constants
    for i, val in enumerate(Q_constants):
        Q_tensors[i] *= val

    return Q_tensors

def interactions(spin_system: SpinSystem, intrs: dict) -> dict:
    """
    Processes all incoherent interactions and organizes them by rank. 
    Disregards interactions below a specified threshold.

    Parameters
    ----------
    spin_system : SpinSystem
        The spin system object containing information about the spins.
    intrs : dict
        A dictionary where the keys represent the interaction type, and the values 
        contain the interaction tensors and the ranks.

    Returns
    -------
    interactions : dict
        A dictionary where the interactions are organized by rank. The values contain 
        all interactions with meaningful strength. The interactions are tuples in the 
        format ("interaction", spin_1, spin_2, tensor).
    """

    # Extract the size of the spin system
    size = spin_system.size

    # Initialize the lists of interaction descriptions for different ranks
    interactions = {
        1: [],
        2: []
    }

    # Iterate through the interactions
    for interaction, properties in intrs.items():

        # Extract the properties
        tensors = properties[0]
        ranks = properties[1]

        # Iterate through the ranks
        for rank in ranks:

            # Process single-spin interactions
            if interaction in ["CSA", "Q"]:
                for spin_1 in range(size):
                    if norm_1(tensors[spin_1], ord='row') > Settings.ZERO_INTERACTION:
                        interactions[rank].append((interaction, spin_1, None, tensors[spin_1]))

            # Process two-spin interactions
            if interaction == "DD":
                for spin_1 in range(size):
                    for spin_2 in range(size):
                        if norm_1(tensors[spin_1, spin_2], ord='row') > Settings.ZERO_INTERACTION:
                            interactions[rank].append((interaction, spin_1, spin_2, tensors[spin_1, spin_2]))

    return interactions

def sop_T(spin_system: SpinSystem, l: int, q: int, interaction_type: str, spin_1: int, spin_2: int = None) -> csc_array:
    """
    Helper function for the relaxation module. Calculates the coupled product 
    superoperators for different interaction types.

    Parameters
    ----------
    spin_system : SpinSystem
        The spin system object containing information about the spins.
    l : int
        Operator rank.
    q : int
        Operator projection.
    interaction_type : str
        Describes the interaction type. Possible options are "CSA", "Q", and "DD", 
        which stand for chemical shift anisotropy, quadrupolar coupling, and 
        dipole-dipole coupling, respectively.
    spin_1 : int
        Index of the first spin.
    spin_2 : int, optional
        Index of the second spin. Leave empty for single-spin interactions (e.g., CSA).

    Returns
    -------
    sop : csc_array
        Coupled spherical tensor superoperator of rank `l` and projection `q`.
    """

    # Extract the size of the spin system
    size = spin_system.size

    # Single-spin linear interaction
    if interaction_type == "CSA":
        sop = _operators.sop_T_coupled(spin_system, l, q, spin_1)

    # Single-spin quadratic interaction
    elif interaction_type == "Q":
        op_def = np.zeros(size, dtype=int)
        op_def[spin_1] = lq_to_idx(l, q)
        op_def = tuple(op_def)
        sop = _operators.sop_prod(spin_system, op_def, 'comm')

    # Two-spin bilinear interaction
    elif interaction_type == "DD":
        sop = _operators.sop_T_coupled(spin_system, l, q, spin_1, spin_2)

    # Raise an error for invalid interaction types
    else:
        raise ValueError(f"Invalid interaction type '{interaction_type}' for relaxation superoperator. Possible options are 'CSA', 'Q', and 'DD'.")

    return sop

def sop_R_redfield(spin_system: SpinSystem,
                   sop_H: csc_array,
                   interactions: dict) -> csc_array:
    """
    Calculates the relaxation superoperator using Redfield relaxation theory.

    Sources:
    
    Eq. 54 from Hilla & Vaara: Rela2x: Analytic and automatic NMR relaxation theory
    https://doi.org/10.1016/j.jmr.2024.107828

    Eq. 24 and 25 from Goodwin & Kuprov: Auxiliary matrix formalism for interaction 
    representation transformations, optimal control, and spin relaxation theories
    https://doi.org/10.1063/1.4928978

    Parameters
    ----------
    spin_system : SpinSystem
        The spin system object containing information about the spins.
    sop_H : csc_array
        Coherent part of the Hamiltonian superoperator.
    interactions : dict
        Interactions organized by rank. Within each rank, contains a list of interactions 
        in the format ("interaction", spin_1, spin_2, tensor).

    Returns
    -------
    sop_R : csc_array
        Relaxation superoperator.
    """

    # Extract necessary information from spin system
    tau_c = spin_system.tau_c
    dim = spin_system.basis.dim

    # Initialize the relaxation superoperator
    sop_R = csc_array((dim, dim), dtype=complex)

    # Define the integration limit for the auxiliary matrix method
    t_max = np.log(1 / Settings.RELATIVE_ERROR) * tau_c

    # Diagonal matrices of correlation times
    tau_c_diagonal_l1 = 1 / tau_c_l(tau_c, 1) * eye_array(sop_H.shape[0], format='csc')
    tau_c_diagonal_l2 = 1 / tau_c_l(tau_c, 2) * eye_array(sop_H.shape[0], format='csc')

    # Auxiliary matrix arrays
    top_left = 1j * sop_H
    bottom_right_l1 = 1j * sop_H - tau_c_diagonal_l1
    bottom_right_l2 = 1j * sop_H - tau_c_diagonal_l2

    # Iterate over the ranks
    for l in [1, 2]:

        # Iterate over the projections (negative q values are handled by spherical tensor properties)
        for q in range(0, l + 1):

            print(f"Processing rank l={l}, projection q={q}")

            # Iterate over the RIGHT interactions
            for interaction_right in interactions[l]:

                # Extract the interaction information
                type_right = interaction_right[0]
                spin_right1 = interaction_right[1]
                spin_right2 = interaction_right[2]
                tensor_right = interaction_right[3]

                # Show current status
                if spin_right1 is None:
                    print(f"{type_right} for spin {spin_right2}")
                elif spin_right2 is None:
                    print(f"{type_right} for spin {spin_right1}")
                else:
                    print(f"{type_right} for spins {spin_right1}-{spin_right2}")

                # Compute the right operator
                sop_T_right = sop_T(spin_system, l, q, type_right, spin_right1, spin_right2)

                # Calculate the Redfield integral using the auxiliary matrix method
                if l == 1:
                    sop_T_right = auxiliary_matrix_expm(top_left, sop_T_right, bottom_right_l1, t_max, Settings.ZERO_AUX)
                elif l == 2:
                    sop_T_right = auxiliary_matrix_expm(top_left, sop_T_right, bottom_right_l2, t_max, Settings.ZERO_AUX)

                # Extract top left and top right blocks
                top_left = sop_T_right[:dim, :dim]
                top_right = sop_T_right[:dim, dim:]

                # Compute the relaxation superoperator term
                sop_T_right = top_left.conj().T @ top_right

                # Initialize the left operator
                sop_T_left = csc_array((dim, dim), dtype=complex)

                # Iterate over the LEFT interactions
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

def relaxation_phenomenological(spin_system: SpinSystem, R_1: np.ndarray, R_2: np.ndarray) -> csc_array:
    """
    Constructs the relaxation superoperator from given R_1 and R_2 values
    for each spin.

    Parameters
    ----------
    spin_system : SpinSystem
        The spin system object containing information about the spins.
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

    # Extract the necessary information from the spin system
    dim = spin_system.basis.dim
    basis_dict = spin_system.basis.dict

    # Create an empty array for the relaxation superoperator
    sop_R = lil_array((dim, dim))

    # Loop over the basis set
    for state, idx in basis_dict.items():

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
                    R_state += R_1[spin]

                # Otherwise, the state must be transverse
                else:

                    # Add to the relaxation rate
                    R_state += R_2[spin]

        # Add to the relaxation matrix
        sop_R[idx, idx] = R_state

    # Convert to CSC array
    sop_R = sop_R.tocsc()

    return sop_R

def sr2k(spin_system: SpinSystem, sop_R: csc_array, B: float) -> csc_array:
    """
    Calculates the scalar relaxation of the second kind (SR2K) based on 
    Abragam's formula and adds it to the relaxation superoperator.

    Parameters
    ----------
    spin_system : SpinSystem
        The spin system object containing information about the spins.
    sop_R : csc_array
        Relaxation superoperator.
    B : float
        Magnetic field in units of T.

    Returns
    -------
    sop_R : csc_array
        Relaxation superoperator with SR2K contributions.
    """

    print("Processing scalar relaxation of the second kind.")

    # Extract the necessary information from the spin system
    gammas = spin_system.gammas
    chemical_shifts = spin_system.chemical_shifts
    J_couplings = spin_system.J_couplings
    size = spin_system.size
    spins = spin_system.spins
    isotopes = spin_system.isotopes

    # Initialize arrays for the relaxation rates
    R_1 = np.zeros(size)
    R_2 = np.zeros(size)

    # Create a list for quadrupolar nuclei
    quadrupolar = []
    
    # Loop over all spins
    for i, spin in enumerate(spins):

        # Append to the list if the spin is quadrupolar
        if spin > 0.5:
            quadrupolar.append(i)
    
    # Loop over the quadrupolar nuclei
    for quad in quadrupolar:

        # Find the indices of the longitudinal and transverse states
        op_def_z, _ = parse_operator_string(spin_system, f"I(z, {quad})")
        op_def_p, _ = parse_operator_string(spin_system, f"I(+, {quad})")
        idx_long = state_idx(spin_system, op_def_z[0])
        idx_trans = state_idx(spin_system, op_def_p[0])

        # Find the relaxation times of the quadrupolar nucleus
        T_1 = 1 / sop_R[idx_long, idx_long]
        T_2 = 1 / sop_R[idx_trans, idx_trans]

        # Convert to real values
        T_1 = np.real(T_1)
        T_2 = np.real(T_2)

        # Find the Larmor frequency of the quadrupolar nucleus
        omega_quad = gammas[quad] * B * (1 + chemical_shifts[quad] * 1e-6)

        # Find the spin quantum number of the quadrupolar nucleus
        S = spins[quad]

        # Loop over all spins
        for target, isotope in enumerate(isotopes):

            # Proceed only if the isotope is different
            if not isotopes[quad] == isotope:

                # Find the Larmor frequency of the target spin
                omega_target = gammas[target] * B * (1 + chemical_shifts[target] * 1e-6)

                # Find the scalar coupling between spins in rad/s
                J = 2 * np.pi * J_couplings[quad][target]

                # Calculate the relaxation rates
                R_1[target] += ((J**2) * S * (S + 1)) / 3 * (2 * T_2) / (1 + (omega_target - omega_quad)**2 * T_2**2)
                R_2[target] += ((J**2) * S * (S + 1)) / 3 * (T_1 + (T_2 / (1 + (omega_target - omega_quad)**2 * T_2**2)))
    
    # Get relaxation superoperator corresponding to SR2K
    sop_R_sr2k = relaxation_phenomenological(spin_system, R_1, R_2)

    # Add to the relaxation superoperator
    sop_R += sop_R_sr2k
    
    return sop_R

def relaxation(spin_system: SpinSystem, 
               sop_H: csc_array,
               B: float,
               temperature: float = None,
               include_sr2k: bool = False,
               real_only: bool = True,
               antisymmetric: bool = False) -> csc_array:
    """
    Calculates the relaxation superoperator.

    Parameters
    ----------
    spin_system : SpinSystem
        The spin system object containing information about the spins.
    sop_H : csc_array
        Hamiltonian superoperator (coherent part).
    B : float
        Magnetic field in units of T.
    temperature : float
        Default: None. Temperature of the spin bath in Kelvins. If specified, Levitt-Di Bari
        thermalization of the relaxation superoperator is performed automatically.
    include_sr2k : bool
        Default: False. Whether to include scalar relaxation of the second kind (SR2K).
        Applies only for spin systems with quadrupolar nuclei.
    real_only : bool
        Default: True. Whether to return only the real component (dynamic frequency shift excluded).
    antisymmetric : bool
        Default: False. Whether to include rank-1 components for CSA interactions.

    Returns
    -------
    sop_R : csc_array
        Relaxation superoperator.
    """

    time_start = time.time()
    print('Constructing relaxation superoperator...')

    # Initialize a dictionary for incoherent interactions
    intrs = {}

    # Process dipole-dipole couplings
    if spin_system.xyz is not None:
        dd_tensors = dd_coupling_tensors(spin_system)
        dd_ranks = [2]
        intrs['DD'] = (dd_tensors, dd_ranks)

    # Process nuclear shielding
    if spin_system.shielding is not None:
        sh_tensors = shielding_intr_tensors(spin_system, B)
        if antisymmetric:
            sh_ranks = [1, 2]
        else:
            sh_ranks = [2]
        intrs['CSA'] = (sh_tensors, sh_ranks)

    # Process quadrupolar coupling
    if spin_system.efg is not None:
        q_tensors = Q_intr_tensors(spin_system)
        q_ranks = [2]
        intrs['Q'] = (q_tensors, q_ranks)
    
    # Remove small interactions and reorganize them rank-wise
    intrs = interactions(spin_system, intrs)
    
    # Calculate R using Redfield theory
    sop_R = sop_R_redfield(spin_system, sop_H, intrs)

    # Process SR2K if enabled
    if include_sr2k:
        sop_R = sr2k(spin_system, sop_R, B)

    # Return only real values if requested
    if real_only:
        sop_R = sop_R.real

    # Remove small elements from the relaxation superoperator
    increase_sparsity(sop_R, Settings.ZERO_RELAXATION)

    # Apply thermalization, if temperature is supplied
    if temperature is not None:
        sop_R = ldb_thermalization(spin_system, sop_R, B, temperature)

    print(f'Relaxation superoperator constructed in {time.time() - time_start:.4f} seconds.')
    print()

    return sop_R

def ldb_thermalization(spin_system: SpinSystem, R: csc_array, B: float, T: float) -> csc_array:
    """
    Applies the Levitt-Di Bari thermalization to the relaxation superoperator.

    Parameters
    ----------
    spin_system : SpinSystem
        The spin system object containing information about the spins.
    R : csc_array
        Relaxation superoperator to be thermalized.
    B : float
        Magnetic field in units of T.
    T : float
        Temperature in units of K.

    Returns
    -------
    R : csc_array
        Thermalized relaxation superoperator.
    """

    # Build the left Zeeman Hamiltonian
    H = hamiltonian(spin_system, B, 'left', disable_outputs=True)

    # Get the matrix exponential corresponding to the Boltzmann distribution
    P = expm(const.hbar / (const.k * T) * H, Settings.ZERO_THERMALIZATION, disable_output=True)

    # Calculate the thermalized relaxation superoperator
    R = R @ P

    return R