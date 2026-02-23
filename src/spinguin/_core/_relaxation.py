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
from scipy.special import eval_legendre
from spinguin._core._superoperators import sop_T_coupled, sop_prod
from spinguin._core._la import \
    eliminate_small, principal_axis_system, \
    cartesian_tensor_to_spherical_tensor, angle_between_vectors, norm_1, \
    auxiliary_matrix_expm, expm, decompose_matrix
from spinguin._core._utils import idx_to_lq, lq_to_idx, parse_operator_string
from spinguin._core._hide_prints import HidePrints
from spinguin._core._parameters import parameters
from spinguin._core._hamiltonian import hamiltonian
from typing import Literal

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
        [V1_pas[l, q] * np.conj(V2_pas[l, q]) for q in range(-l, l + 1)]
    ).real

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
    print("Processing interactions for relaxation...")
    print(f"Dropping interactions smaller than {parameters.zero_interaction}:")
    time_start = time.time()

    # Obtain the required parameters
    zv = parameters.zero_interaction

    # Initialize the lists of interaction descriptions for different ranks
    interactions = {
        1: [],
        2: []
    }

    # Process dipole-dipole couplings
    if spin_system.xyz is not None:

        # Get the DD-coupling tensors
        dd_tensors = dd_coupling_tensors(spin_system.xyz, spin_system.gammas)

        # Go through the DD-coupling tensors
        for spin_1 in range(spin_system.nspins):
            for spin_2 in range(spin_1):
                if norm_1(dd_tensors[spin_1, spin_2], ord='row') > zv:
                    interactions[2].append((
                        "DD",
                        spin_1,
                        spin_2, 
                        dd_tensors[spin_1, spin_2]
                    ))
                    print(f"\tKept: DD: {spin_1}-{spin_2}")
                else:
                    print(f"\tDropped: DD: {spin_1}-{spin_2}")

    # Process nuclear shielding
    if spin_system.shielding is not None:

        # Get the shielding tensors
        sh_tensors = shielding_intr_tensors(
            spin_system.shielding,
            spin_system.gammas,
            parameters.magnetic_field
        )
        
        # Go through the shielding tensors
        for spin_1 in range(spin_system.nspins):
            _, antisym, sym = decompose_matrix(sh_tensors[spin_1])

            # Antisymmetric part (only if requested)
            if spin_system.relaxation.antisymmetric:
                if norm_1(antisym, ord='row') > zv:
                    interactions[1].append(
                        ("CSA", spin_1, None, sh_tensors[spin_1])
                    )
                    print(f"\tKept: CSA: {spin_1} (rank 1)")
                else:
                    print(f"\tDropped: CSA: {spin_1} (rank 1)")

            # Symmetric part
            if norm_1(sym, ord='row') > zv:
                interactions[2].append(
                    ("CSA", spin_1, None, sh_tensors[spin_1])
                )
                print(f"\tKept: CSA: {spin_1} (rank 2)")
            else:
                print(f"\tDropped: CSA: {spin_1} (rank 2)")

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
                if norm_1(q_tensors[spin_1], ord='row') > zv:
                    interactions[2].append(
                        ("Q", spin_1, None, q_tensors[spin_1])
                    )
                    print(f"\tKept: Q: {spin_1}")
                else:
                    print(f"\tDropped: Q: {spin_1}")

    print(f"Completed in {time.time() - time_start:.4f} seconds.\n")

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

def _correlation_matrix(interactions: dict) -> dict:
    """
    Calculates the correlation matrix for ranks l = 1 and l = 2. The matrix
    elements µµ' represent the correlation function G0 for the interaction pair
    µµ'.

    Parameters
    ----------
    interactions : dict
        A dictionary that contains two keys: l = 1 and l = 2. The values are
        lists where each interaction is represented by a tuple.

    Returns
    -------
    G_0 : dict
        A dictionary that contains two keys: l = 1 and l = 2. The values are
        NumPy arrays, which contain the correlation matrix.
    """
    print("Calculating the correlation matrix...")
    time_start = time.time()

    # Initialise a dictionary for the correlation matrix
    n1 = len(interactions[1])
    n2 = len(interactions[2])
    G_0 = {
        1 : np.zeros(shape=(n1, n1)),
        2 : np.zeros(shape=(n2, n2))
    }

    # Loop over the ranks
    for l in [1, 2]:

        # Loop over the LEFT interactions
        for i in range(len(interactions[l])):

            # Extract the interaction tensor
            tensor_i = interactions[l][i][3]

            # Loop over the RIGHT interactions (by symmetry µµ' = µ'u)
            for j in range(i+1):

                # Extract the interaction tensor
                tensor_j = interactions[l][j][3]

                # Compute G0 and save to the dictionary
                G_0_curr = G0(tensor_i, tensor_j, l)
                if i == j:
                    G_0[l][i, j] = G_0_curr
                else:
                    G_0[l][i, j] = G_0_curr
                    G_0[l][j, i] = G_0_curr

    print(f"Completed in {time.time() - time_start:.4f} seconds.\n")

    return G_0

def _correlation_matrix_eig(G_0: dict) -> tuple[dict, dict, dict]:
    """
    Calculates the eigendecomposition G_0 = Q * L * Q^T of the correlation
    matrix G_0 for ranks l = 1 and l = 2. Only the three and five largest
    eigenvalues are store for the ranks l = 1 and l = 2, respectively.

    Parameters
    ----------
    G_0 : dict
        A dictionary that has keys l = 1 and l = 2, with the values
        corresponding to the correlation matrices for ranks l = 1 and l = 2.
    
    Returns
    -------
    L : dict
        A dictionary that has keys l = 1 and l = 2, with the values
        corresponding to the eigenvalues L for ranks l = 1 and l = 2.
    Q : dict
        A dictionary that has keys l = 1 and l = 2, with the values
        corresponding to the eigenvectors Q for ranks l = 1 and l = 2.
    """
    print("Performing eigendecomposition of the correlation matrix...")
    time_start = time.time()

    # Initialise dictionaries for the eigendecomposition
    L = {}
    Q = {}

    # Calculate the eigendecomposition for the ranks l = 1 and l = 2
    for l in [1, 2]:
        L_l, Q_l = np.linalg.eigh(G_0[l])

        # Find indices of the eigendecomposition to be kept
        max_idx = min(2*l+1, len(L_l))
        idx = np.flip(np.argsort(L_l))[:max_idx]

        # Write the eigendecomposition to the dictionaries
        L[l] = L_l[idx]
        Q[l] = Q_l[:, idx]

        print(f"Rank {l}:")
        print(f"\tCorrelations before truncation: {len(L_l)}")
        print(f"\tCorrelations after truncation: {len(L[l])}")
        with np.printoptions(precision=4):
            print(f"\tEigenvalues: {L[l]}")

    print(f"Completed in {time.time() - time_start:.4f} seconds.\n")

    return L, Q

def _get_all_sop_T(spin_system: SpinSystem, interactions: dict) -> dict:
    """
    Helper function that builds all the coupled spherical tensor superoperators.

    Parameters
    ----------
    spin_system : SpinSystem
        Spin system for which the superoperators are going to be built.
    interactions : dict
        A dictionary that contains two keys: l = 1 and l = 2. The values are
        lists where each interaction is represented by a tuple.

    Returns
    -------
    sop_Ts : dict
        A dictionary that contains the coupled spherical tensor superoperators.
        The keys are tuples: (l, q, itype, spin1, spin2).
    """
    print("Building the coupled spherical tensor operators...")
    time_start = time.time()

    # Create an empty dictionary for the coupled spherical tensor operators
    sop_Ts = {}

    # Iterate over ranks
    for l in [1, 2]:

        # Iterate over projections
        for q in range(-l, l+1):

            # Iterate over interactions
            for interaction in interactions[l]:

                # Extract the interaction information
                itype = interaction[0]
                spin1 = interaction[1]
                spin2 = interaction[2]

                # Compute the coupled T superoperator
                sop_T = _get_sop_T(spin_system, l, q, itype, spin1, spin2)

                # Store the coupled T superoperator to the dictionary
                sop_Ts[(l, q, itype, spin1, spin2)] = sop_T

    print(f"Completed in {time.time() - time_start:.4f} seconds.\n")
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
    print('Constructing relaxation superoperator using Redfield theory...\n')

    # Extract information from the spin system
    dim = spin_system.basis.dim
    tau_c = spin_system.relaxation.tau_c
    relative_error = spin_system.relaxation.relative_error

    # Initialize the relaxation superoperator
    R = sp.csc_array((dim, dim), dtype=complex)

    # Process the interactions
    interactions = _process_interactions(spin_system)

    # Calculate the correlation matrix
    G_0 = _correlation_matrix(interactions)

    # Calculate the singular value decomposition of the correlation matrix
    L, Q = _correlation_matrix_eig(G_0)

    # Build the coherent Hamiltonian superoperator
    with HidePrints():
        H = hamiltonian(spin_system)

    # Define the integration limit for the auxiliary matrix method
    t_max = np.log(1 / relative_error) * tau_c

    # Build the top left array of the auxiliary matrix (A)
    top_left = 1j * H

    # Build coupled spherical tensor operators
    sop_Ts = _get_all_sop_T(spin_system, interactions)

    # Iterate over the ranks
    print("Calculating the Redfield superoperator terms...")
    time_start = time.time()
    for l in [1, 2]:

        # Diagonal matrix of correlation time
        tau_c_diagonal_l = 1/tau_c_l(tau_c, l) * sp.eye_array(dim, format='csc')

        # Bottom right array of auxiliary matrix (C)
        bottom_right = 1j * H - tau_c_diagonal_l

        # Iterate over the projections
        for q in range(-l, l + 1):

            print(f"l = {l}, q = {q}")

            # Iterate over the eigenvalues
            for j in range(len(L[l])):

                # Calculate the coupled T superoperators
                sop_T_left = sp.csc_array((dim, dim), dtype=complex)
                sop_T_right = sp.csc_array((dim, dim), dtype=complex)
                for u, interaction in enumerate(interactions[l]):

                    # Extract the interaction information
                    itype = interaction[0]
                    spin1 = interaction[1]
                    spin2 = interaction[2]

                    # Acquire the coupled T superoperator
                    sop_T_u = sop_Ts[(l, q, itype, spin1, spin2)]

                    # Add to the sum
                    sop_T_left = sop_T_left + Q[l][u, j] * sop_T_u.conj().T
                    sop_T_right = sop_T_right + Q[l][u, j] * sop_T_u

                # Calculate the Redfield integral
                aux_expm = auxiliary_matrix_expm(
                    A = top_left,
                    B = L[l][j] * sop_T_right,
                    C = bottom_right,
                    t = t_max,
                    zero_value = parameters.zero_aux
                )
                aux_top_l = aux_expm[:dim, :dim]
                aux_top_r = aux_expm[:dim, dim:]
                integral = aux_top_l.conj().T @ aux_top_r

                # Add the current term to the relaxation superoperator
                R = R + sop_T_left @ integral

    print(f"Completed in {time.time() - time_start:.4f} seconds.\n")

    # Return only real values unless dynamic frequency shifts are requested
    if not spin_system.relaxation.dynamic_frequency_shift:
        print("Removing the dynamic frequency shifts...")
        time_start = time.time()
        R = R.real
        print(f"Completed in {time.time() - time_start:.4f} seconds.\n")

    # Eliminate small values
    print("Eliminating small values from the relaxation superoperator...")
    time_start = time.time()
    eliminate_small(R, parameters.zero_relaxation)
    print(f"Completed in {time.time() - time_start:.4f} seconds.\n")
    
    print("Redfield relaxation superoperator constructed in "
          f"{time.time() - time_start_R:.4f} seconds.\n")

    return R