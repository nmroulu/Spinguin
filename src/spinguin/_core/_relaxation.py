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
    auxiliary_matrix_expm, expm, read_shared_sparse, write_shared_sparse
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

def sop_R_redfield_term(
        l: int, q: int,
        type_r: str, spin_r1: int, spin_r2: int, tensor_r: np.ndarray,
        top_l_shared: dict, top_r_shared: dict, bottom_r_shared: dict,
        t_max: float, aux_zero: float, relaxation_zero: float,
        sop_Ts: dict, interactions: dict
) -> tuple[int, int, str, int, int, sp.csc_array]:
    """
    Helper function for the Redfield relaxation theory. This function calculates
    one term of the relaxation superoperator and enables the use of parallel
    computation.

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
    spin_r1 : int
        Index of the first spin in the interaction.
    spin_r2 : int
        Index of the second spin in the interaction. Leave empty for single-spin
        interactions (e.g., CSA).
    tensor_r : np.ndarray
        Interaction tensor for the right-hand interaction.
    top_l_shared : dict
        Dictionary containing the shared top left block of the auxiliary matrix.
    top_r_shared : dict
        Dictionary containing the shared top right block of the auxiliary
        matrix.
    bottom_r_shared : dict
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
    sop_Ts : dict
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
    spin_r1 : int
        Index of the first spin.
    spin_r2 : int
        Index of the second spin.
    sop_R_term : csc_array
        Relaxation superoperator term for the given interaction.
    """
    # Create an empty list for the SharedMemory objects
    shms = []

    # Convert the shared arrays back to CSC arrays
    top_l, top_l_shm = read_shared_sparse(top_l_shared)
    top_r, top_r_shm = read_shared_sparse(top_r_shared)
    bottom_r, bottom_r_shm = read_shared_sparse(bottom_r_shared)
    dim = top_r.shape[0]

    # Store the SharedMemories
    shms.extend(top_l_shm)
    shms.extend(top_r_shm)
    shms.extend(bottom_r_shm)
    
    # Calculate the Redfield integral using the auxiliary matrix method
    aux_expm = auxiliary_matrix_expm(top_l, top_r, bottom_r, t_max, aux_zero)

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
        spin_l1 = interaction_l[1]
        spin_l2 = interaction_l[2]
        tensor_l = interaction_l[3]

        # Continue only if T is found (non-zero)
        if (l, q, type_l, spin_l1, spin_l2) in sop_Ts:

            # Compute G0
            G_0 = G0(tensor_l, tensor_r, l)

            # Get the shared T
            sop_T_shared = sop_Ts[(l, q, type_l, spin_l1, spin_l2)]

            # Add current term to the left operator
            sop_T, sop_T_shm = read_shared_sparse(sop_T_shared)
            sop_T_l += G_0 * sop_T
            shms.extend(sop_T_shm)

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

    return l, q, type_r, spin_r1, spin_r2, sop_R_term

def _sop_R_redfield(
    spin_system: SpinSystem
) -> np.ndarray | sp.csc_array:
    """
    Calculates the relaxation superoperator using Redfield relaxation theory.

    Sources:
    
    Eq. 54 from Hilla & Vaara: Rela2x: Analytic and automatic NMR relaxation
    theory
    https://doi.org/10.1016/j.jmr.2024.107828

    Eq. 24 and 25 from Goodwin & Kuprov: Auxiliary matrix formalism for
    interaction representation transformations, optimal control, and spin
    relaxation theories
    https://doi.org/10.1063/1.4928978

    Parameters
    ----------
    spin_system : SpinSystem
        SpinSystem for which the relaxation superoperator is to be calculated.

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

    # Define the integration limit for the auxiliary matrix method
    t_max = np.log(1 / relative_error) * tau_c

    # Initialize a list to hold all SharedMemories (for parallel processing)
    shms = []

    # Build the top left array of the auxiliary matrix
    top_left = 1j * H
    top_left, top_left_shm = write_shared_sparse(top_left)
    shms.extend(top_left_shm)

    # FIRST LOOP
    # -- PRECOMPUTE THE COUPLED T SUPEROPERATORS
    # -- CREATE THE LIST OF TASKS
    print("Building superoperators...")
    sop_Ts = {}
    tasks = []
    
    # Iterate over the ranks
    for l in [1, 2]:

        # Diagonal matrix of correlation time
        tau_c_diagonal_l = 1/tau_c_l(tau_c, l) * sp.eye_array(dim, format='csc')

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

    # Initialize the relaxation superoperator
    R = sp.csc_array((dim, dim), dtype=complex)

    # SECOND LOOP -- Iterate over the tasks in parallel and build the R
    if dim > parameters.parallel_dim:
        print("Performing the Redfield integrals in parallel...")

        # Create the parallel tasks
        parallel = Parallel(n_jobs=-1, return_as="generator_unordered")
        output_generator = parallel(
            delayed(sop_R_redfield_term)(*task) for task in tasks
        )

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

            # Parse the result and add term to total relaxation superoperator
            l, q, itype, spin1, spin2, R_term = sop_R_redfield_term(*task)
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