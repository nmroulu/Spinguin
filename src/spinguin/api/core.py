"""
core.py

This module provides user-friendly wrapper functions for the core functionality
of the Spinguin package.
"""

# Imports
import numpy as np
import scipy.sparse as sp
from copy import deepcopy
from typing import Literal

from spinguin.api.parameters import parameters
from spinguin.api.spin_system import SpinSystem

from spinguin.core.chem import (
    dissociate as _dissociate,
    associate as _associate,
    permute_spins as _permute_spins
)
from spinguin.core.hamiltonian import sop_H as _sop_H
from spinguin.core.hide_prints import HidePrints
from spinguin.core.liouvillian import sop_L as liouvillian
from spinguin.core.nmr_isotopes import gamma, quadrupole_moment, spin
from spinguin.core.operators import op_from_string as _op_from_string
from spinguin.core.propagation import (
    propagator_to_rotframe as _propagator_to_rotframe,
    sop_propagator as _sop_propagator,
    sop_pulse as _sop_pulse
)
from spinguin.core.relaxation import (
    sop_R_phenomenological as _sop_R_phenomenological,
    sop_R_redfield as _sop_R_redfield,
    sop_R_sr2k as _sop_R_sr2k,
    ldb_thermalization as _ldb_thermalization
)
from spinguin.core.specutils import (
    frequency_to_chemical_shift,
    resonance_frequency as _resonance_frequency,
    spectral_width_to_dwell_time as _spectral_width_to_dwell_time,
    spectrum as _spectrum
)
from spinguin.core.superoperators import sop_from_string as _sop_from_string
from spinguin.core.states import (
    alpha_state as _alpha_state,
    beta_state as _beta_state,
    equilibrium_state as _equilibrium_state,
    measure as _measure,
    singlet_state as _singlet_state,
    state_from_string as _state_from_string,
    state_to_zeeman as _state_to_zeeman,
    triplet_minus_state as _triplet_minus_state,
    triplet_plus_state as _triplet_plus_state,
    triplet_zero_state as _triplet_zero_state,
    unit_state as _unit_state
)

__all__ = [
    "alpha_state",
    "associate",
    "beta_state",
    "dissociate",
    "equilibrium_state",
    "frequency_to_chemical_shift",
    "gamma",
    "hamiltonian",
    "inversion_recovery",
    "liouvillian",
    "measure",
    "operator",
    "permute_spins",
    "propagator",
    "propagator_to_rotframe",
    "pulse",
    "pulse_and_acquire",
    "quadrupole_moment",
    "relaxation",
    "resonance_frequency",
    "singlet_state",
    "spectral_width_to_dwell_time",
    "spectrum",
    "spin",
    "state",
    "state_to_zeeman",
    "superoperator",
    "time_axis",
    "triplet_minus_state",
    "triplet_plus_state",
    "triplet_zero_state",
    "unit_state"
]

def operator(spin_system: SpinSystem,
             operator: str) -> np.ndarray | sp.csc_array:
    """
    Generates an operator for the `spin_system` in Hilbert space from the
    user-specified `operator` string.

    Parameters
    ----------
    spin_system : SpinSystem
        Spin system for which the operator is going to be generated.
    operator : str
        Defines the operator to be generated. The operator string must
        follow the rules below:

        - Cartesian and ladder operators: `I(component,index)` or 
          `I(component)`. Examples:

            - `I(x,4)` --> Creates x-operator for spin at index 4.
            - `I(x)`--> Creates x-operator for all spins.

        - Spherical tensor operators: `T(l,q,index)` or `T(l,q)`. Examples:

            - `T(1,-1,3)` --> \
              Creates operator with `l=1`, `q=-1` for spin at index 3.
            - `T(1, -1)` --> \
              Creates operator with `l=1`, `q=-1` for all spins.
            
        - Product operators have `*` in between the single-spin operators:
          `I(z,0) * I(z,1)`
        - Sums of operators have `+` in between the operators:
          `I(x,0) + I(x,1)`
        - Unit operators are ignored in the input. Interpretation of these
          two is identical: `E * I(z,1)`, `I(z,1)`
        
        Special case: An empty `operator` string is considered as unit
        operator.

        Whitespace will be ignored in the input.

        NOTE: Indexing starts from 0!

    Returns
    -------
    op : ndarray or csc_array
        An array representing the requested operator.
    """
    # Construct the operator
    op = _op_from_string(
        spins = spin_system.spins,
        operator = operator,
        sparse = parameters.sparse_operator
    )
    
    return op

def superoperator(spin_system: SpinSystem,
                  operator: str,
                  side: Literal["comm", "left", "right"] = "comm"
                  ) -> np.ndarray | sp.csc_array:
    """
    Generates a Liouville-space superoperator for the `spin_system` from the
    user-specified `operator` string.

    Parameters
    ----------
    spin_system : SpinSystem
        Spin system for which the superoperator is going to be generated.
    operator : str
        Defines the superoperator to be generated. The operator string must
        follow the rules below:

        - Cartesian and ladder operators: `I(component,index)` or
          `I(component)`. Examples:

            - `I(x,4)` --> Creates x-operator for spin at index 4.
            - `I(x)`--> Creates x-operator for all spins.

        - Spherical tensor operators: `T(l,q,index)` or `T(l,q)`. Examples:

            - `T(1,-1,3)` --> \
              Creates operator with `l=1`, `q=-1` for spin at index 3.
            - `T(1, -1)` --> \
              Creates operator with `l=1`, `q=-1` for all spins.
            
        - Product operators have `*` in between the single-spin operators:
          `I(z,0) * I(z,1)`
        - Sums of operators have `+` in between the operators:
          `I(x,0) + I(x,1)`
        - Unit operators are ignored in the input. Interpretation of these
          two is identical: `E * I(z,1)`, `I(z,1)`
        
        Special case: An empty `operator` string is considered as unit
        operator.

        Whitespace will be ignored in the input.

        NOTE: Indexing starts from 0!
    side : {'comm', 'left', 'right'}
        The type of superoperator:
        - 'comm' -- commutation superoperator (default)
        - 'left' -- left superoperator
        - 'right' -- right superoperator

    Returns
    -------
    sop : ndarray or csc_array
        An array representing the requested superoperator.
    """

    # Check that the basis has been built
    if spin_system.basis.basis is None:
        raise ValueError("Please build the basis before constructing "
                         "superoperators.")
        
    # Construct the superoperator
    sop = _sop_from_string(
        operator = operator,
        basis = spin_system.basis.basis,
        spins = spin_system.spins,
        side = side,
        sparse = parameters.sparse_superoperator
    )
        
    return sop

INTERACTIONTYPE = Literal["zeeman", "chemical_shift", "J_coupling"]
INTERACTIONDEFAULT = ["zeeman", "chemical_shift", "J_coupling"]
def hamiltonian(
        spin_system: SpinSystem,
        interactions: list[INTERACTIONTYPE] = INTERACTIONDEFAULT,
        side: Literal["comm", "left", "right"] = "comm"
) -> np.ndarray | sp.csc_array:
    """
    Creates the requested Hamiltonian superoperator for the spin system.

    Parameters
    ----------
    spin_system : SpinSystem
        Spin system for which the Hamiltonian is going to be generated.
    interactions : list, default=["zeeman", "chemical_shift", "J_coupling"]
        Specifies which interactions are taken into account. The options are:

        - 'zeeman' -- Zeeman interaction
        - 'chemical_shift' -- Isotropic chemical shift
        - 'J_coupling' -- Scalar J-coupling

    side : {'comm', 'left', 'right'}
        The type of superoperator:
        
        - 'comm' -- commutation superoperator (default)
        - 'left' -- left superoperator
        - 'right' -- right superoperator

    Returns
    -------
    H : ndarray or csc_array
        Hamiltonian superoperator.
    """
        
    # Check that the required attributes have been set
    if spin_system.basis.basis is None:
        raise ValueError("Please build basis before constructing " 
                         "the Hamiltonian.")
    if "zeeman" in interactions:
        if parameters.magnetic_field is None:
            raise ValueError("Please set the magnetic field before "
                             "constructing the Zeeman Hamiltonian.")
    if "chemical_shift" in interactions:
        if parameters.magnetic_field is None:
            raise ValueError("Please set the magnetic field before "
                             "constructing the chemical shift Hamiltonian.")
        
    H = _sop_H(
        basis = spin_system.basis.basis,
        spins = spin_system.spins,
        gammas = spin_system.gammas,
        B = parameters.magnetic_field,
        chemical_shifts = spin_system.chemical_shifts,
        J_couplings = spin_system.J_couplings,
        interactions = interactions,
        side = side,
        sparse = parameters.sparse_hamiltonian,
        zero_value = parameters.zero_hamiltonian
    )

    return H

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
            sparse = parameters.sparse_relaxation)

    # Make relaxation superoperator using Redfield theory
    elif spin_system.relaxation.theory == "redfield":
        
        # Build the coherent hamiltonian
        with HidePrints():
            H = hamiltonian(spin_system=spin_system,
                            interactions=INTERACTIONDEFAULT,
                            side="comm")

        # Build the Redfield relaxation superoperator
        R = _sop_R_redfield(
            basis = spin_system.basis.basis,
            sop_H = H,
            tau_c = spin_system.relaxation.tau_c,
            spins = spin_system.spins,
            B = parameters.magnetic_field,
            gammas = spin_system.gammas,
            quad = spin_system.quad,
            xyz = spin_system.xyz,
            shielding = spin_system.shielding,
            efg = spin_system.efg,
            include_antisymmetric = spin_system.relaxation.antisymmetric,
            include_dynamic_frequency_shift = \
                spin_system.relaxation.dynamic_frequency_shift,
            relative_error = spin_system.relaxation.relative_error,
            interaction_zero = parameters.zero_interaction,
            aux_zero = parameters.zero_aux,
            relaxation_zero = parameters.zero_relaxation,
            parallel_dim = parameters.parallel_dim,
            sparse = parameters.sparse_relaxation
        )
    
    # Apply scalar relaxation of the second kind if requested
    if spin_system.relaxation.sr2k:
        R += _sop_R_sr2k(
            basis = spin_system.basis.basis,
            spins = spin_system.spins,
            gammas = spin_system.gammas,
            chemical_shifts = spin_system.chemical_shifts,
            J_couplings = spin_system.J_couplings,
            sop_R = R,
            B = parameters.magnetic_field,
            sparse = parameters.sparse_relaxation
        )
        
    # Apply thermalization if requested
    if spin_system.relaxation.thermalization:
        
        # Build the left Hamiltonian superopertor
        with HidePrints():
            H_left = hamiltonian(
                spin_system = spin_system,
                interactions = INTERACTIONDEFAULT,
                side = "left"
            )
            
        # Perform the thermalization
        R = _ldb_thermalization(
            R = R,
            H_left = H_left,
            T = parameters.temperature,
            zero_value = parameters.zero_thermalization)

    return R

def equilibrium_state(spin_system: SpinSystem) -> np.ndarray | sp.csc_array:
    """
    Creates the thermal equilibrium state for the spin system.

    Parameters
    ----------
    spin_system : SpinSystem
        Spin system for which the equilibrium state is going to be created.

    Returns
    -------
    rho : ndarray or csc_array
        Equilibrium state vector.
    """

    # Check that the required attributes are set
    if spin_system.basis.basis is None:
        raise ValueError("Please build the basis before constructing the "
                         "equilibrium state.")
    if parameters.magnetic_field is None:
        raise ValueError("Please set the magnetic field before "
                         "constructing the equilibrium state.")
    if parameters.temperature is None:
        raise ValueError("Please set the temperature before "
                         "constructing the equilibrium state.")

    # Build the left Hamiltonian superoperator
    with HidePrints():
        H_left = hamiltonian(
            spin_system = spin_system,
            interactions = INTERACTIONDEFAULT,
            side = "left"
        )

    # Build the equilibrium state
    rho = _equilibrium_state(
        basis = spin_system.basis.basis,
        spins = spin_system.spins,
        H_left = H_left,
        T = parameters.temperature,
        sparse = parameters.sparse_state,
        zero_value = parameters.zero_equilibrium
    )

    return rho

def singlet_state(spin_system: SpinSystem,
                  index_1 : int,
                  index_2 : int) -> np.ndarray | sp.csc_array:
    """
    Generates the singlet state between two spin-1/2 nuclei. Unit state is
    assigned to the other spins.

    Parameters
    ----------
    spin_system : SpinSystem
        Spin system for which the singlet state is created.
    index_1 : int
        Index of the first spin in the singlet state.
    index_2 : int
        Index of the second spin in the singlet state.

    Returns
    -------
    rho : ndarray or csc_array
        State vector corresponding to the singlet state.
    """
    # Check that the required attributes are set
    if spin_system.basis.basis is None:
        raise ValueError("Please build the basis before constructing the "
                         "singlet state.")

    # Build the singlet state
    rho = _singlet_state(
        basis = spin_system.basis.basis,
        spins = spin_system.spins,
        index_1 = index_1,
        index_2 = index_2,
        sparse = parameters.sparse_state
    )

    return rho

def pulse(spin_system: SpinSystem,
          operator: str,
          angle: float) -> np.ndarray | sp.csc_array:
    """
    Creates a pulse superoperator that is applied to a state by multiplying
    from the left.

    Parameters
    ----------
    spin_system : SpinSystem
        Spin system for which the pulse superoperator is going to be created.
    operator : str
        Defines the pulse to be generated. The operator string must
        follow the rules below:

        - Cartesian and ladder operators: `I(component,index)` or
          `I(component)`. Examples:

            - `I(x,4)` --> Creates x-operator for spin at index 4.
            - `I(x)`--> Creates x-operator for all spins.

        - Spherical tensor operators: `T(l,q,index)` or `T(l,q)`. Examples:

            - `T(1,-1,3)` --> \
              Creates operator with `l=1`, `q=-1` for spin at index 3.
            - `T(1, -1)` --> \
              Creates operator with `l=1`, `q=-1` for all spins.
            
        - Product operators have `*` in between the single-spin operators:
          `I(z,0) * I(z,1)`
        - Sums of operators have `+` in between the operators:
          `I(x,0) + I(x,1)`
        - Unit operators are ignored in the input. Interpretation of these
          two is identical: `E * I(z,1)`, `I(z,1)`
        
        Special case: An empty `operator` string is considered as unit operator.

        Whitespace will be ignored in the input.

        NOTE: Indexing starts from 0!
    angle : float
        Pulse angle in degrees.

    Returns
    -------
    P : ndarray or csc_array
        Pulse superoperator.
    """

    # Check that the required attributes have been set
    if spin_system.basis.basis is None:
        raise ValueError("Please build the basis before constructing pulse "
                         "superoperators.")

    # Construct the pulse superoperator
    P = _sop_pulse(
        basis = spin_system.basis.basis,
        spins = spin_system.spins,
        operator = operator,
        angle = angle,
        sparse = parameters.sparse_pulse,
        zero_value = parameters.zero_pulse
    )

    return P

def propagator(L: np.ndarray | sp.csc_array,
               t: float) -> np.ndarray | sp.csc_array:
    """
    Constructs the time propagator exp(L*t).

    Parameters
    ----------
    L : csc_array
        Liouvillian superoperator, L = -iH - R + K.
    t : float
        Time step of the simulation in seconds.

    Returns
    -------
    expm_Lt : csc_array or ndarray
        Time propagator exp(L*t).
    """
    # Create the propagator
    P = _sop_propagator(
        L = L,
        t = t,
        zero_value = parameters.zero_propagator,
        density_threshold = parameters.propagator_density
    )
    
    return P

def propagator_to_rotframe(spin_system: SpinSystem,
                           P: np.ndarray | sp.csc_array,
                           t: float,
                           center_frequencies: dict=None
                           ) -> np.ndarray | sp.csc_array:
    """
    Transforms the time propagator to the rotating frame.

    Parameters
    ----------
    spin_system : SpinSystem
        Spin system whose time propagator is going to be transformed.
    P : ndarray or csc_array
        Time propagator in the laboratory frame.
    t : float
        Time step of the simulation in seconds.
    center_frequencies : dict
        Dictionary that describes the center frequencies for each isotope in the
        units of ppm.

    Returns
    -------
    P_rot : ndarray or csc_array
        The time propagator transformed into the rotating frame.
    """
    # Obtain an array of center frequencies for each spin
    center = np.zeros(spin_system.nspins)
    for spin in range(spin_system.nspins):
        if spin_system.isotopes[spin] in center_frequencies:
            center[spin] = center_frequencies[spin_system.isotopes[spin]]

    # Construct Hamiltonian that specifies the interaction frame
    H_frame = _sop_H(
        basis = spin_system.basis.basis,
        spins = spin_system.spins,
        gammas = spin_system.gammas,
        B = parameters.magnetic_field,
        chemical_shifts = center,
        interactions = ["zeeman", "chemical_shift"],
        side = "comm",
        sparse = parameters.sparse_hamiltonian,
        zero_value = parameters.zero_hamiltonian
    )

    # Convert the propagator to rotating frame
    P_rot = _propagator_to_rotframe(
        sop_P = P,
        sop_H0 = H_frame,
        t = t,
        zero_value = parameters.zero_propagator
    )
    
    return P_rot

def measure(spin_system: SpinSystem,
            rho: np.ndarray | sp.csc_array,
            operator: str) -> complex:
    """
    Computes the expectation value of the specified operator for a given state
    vector. Assumes that the state vector `rho` represents a trace-normalized
    density matrix.

    Parameters
    ----------
    spin_system : SpinSystem
        Spin system whose state is going to be measured.
    rho : ndarray or csc_array
        State vector that describes the density matrix.
    operator : str
        Defines the operator to be measured. The operator string must follow the
        rules below:

        - Cartesian and ladder operators: `I(component,index)` or
          `I(component)`. Examples:

            - `I(x,4)` --> Creates x-operator for spin at index 4.
            - `I(x)` --> Creates x-operator for all spins.

        - Spherical tensor operators: `T(l,q,index)` or `T(l,q)`. Examples:

            - `T(1,-1,3)` --> \
              Creates operator with `l=1`, `q=-1` for spin at index 3.
            - `T(1, -1)` --> \
              Creates operator with `l=1`, `q=-1` for all spins.
            
        - Product operators have `*` in between the single-spin operators:
          `I(z,0) * I(z,1)`
        - Sums of operators have `+` in between the operators:
          `I(x,0) + I(x,1)`
        - Unit operators are ignored in the input. Interpretation of these
          two is identical: `E * I(z,1)`, `I(z,1)`
        
        Special case: An empty `operator` string is considered as unit operator.

        Whitespace will be ignored in the input.

        NOTE: Indexing starts from 0!

    Returns
    -------
    ex : complex
        Expectation value.
    """
    # Check that the required attributes are set
    if spin_system.basis.basis is None:
        raise ValueError("Please build the basis before measuring an "
                         "expectation value of an operator.")
    
    # Perform the measurement
    ex = _measure(
        basis = spin_system.basis.basis,
        spins = spin_system.spins,
        rho = rho,
        operator = operator
    )

    return ex

def spectral_width_to_dwell_time(
        spectral_width: float,
        isotope: str
) -> float:
    """
    Calculates the dwell time (in seconds) from the spectral width given in ppm.

    Parameters
    ----------
    spectral_width : float
        Spectral width in ppm.
    isotope: str
        Nucleus symbol (e.g. `'1H'`) used to select the gyromagnetic ratio 
        required for the conversion.

    Returns
    -------
    dwell_time : float
        Dwell time in seconds.

    Notes
    -----
    Requires that the following is set:

    - parameters.magnetic_field
    """
    # Obtain the dwell time
    dwell_time = _spectral_width_to_dwell_time(
        spectral_width = spectral_width,
        isotope = isotope,
        B = parameters.magnetic_field
    )

    return dwell_time

def spectrum(signal: np.ndarray,
             dwell_time: float,
             normalize: bool = True,
             part: Literal["real", "imag"] = "real"
             ) -> tuple[np.ndarray, np.ndarray]:
    """
    A wrapper function for the Fourier transform. Computes the Fourier transform
    and returns the frequency and spectrum (either the real or imaginary part of 
    the Fourier transform).

    Parameters
    ----------
    signal : ndarray
        Input signal in the time domain.
    dwell_time : float
        Time step between consecutive samples in the signal.
    normalize : bool, default=True
        Whether to normalize the Fourier transform.
    part : {'real', 'imag'}
        Specifies which part of the Fourier transform to return. Can be "real" 
        or "imag".

    Returns
    -------
    freqs : ndarray
        Frequencies corresponding to the Fourier-transformed signal.
    spectrum : ndarray
        Specified part (real or imaginary) of the Fourier-transformed signal 
        in the frequency domain.

    Notes
    -----
    Required global parameters:

    - parameters.dwell_time
    """
    # Compute the Fourier transform
    freqs, spectrum = _spectrum(
        signal = signal,
        dt = dwell_time,
        normalize = normalize,
        part = part
    )

    return freqs, spectrum

def resonance_frequency(
        isotope: str,
        offset: float = 0,
        unit: Literal["Hz", "rad/s"] = "Hz"
) -> float:
    """
    Computes the resonance frequency of the given `isotope` at the specified
    magnetic field.

    Parameters
    ----------
    isotope : str
        Nucleus symbol (e.g. `'1H'`) used to select the gyromagnetic ratio 
        required for the conversion.
    offset : float, default=0
        Offset in ppm.
    unit : {'Hz', 'rad/s'}
        Specifies in which units the frequency is returned.
    
    Returns
    -------
    omega : float
        Resonance frequency in the requested units.

    Notes
    -----
    Required global parameters:

    - parameters.magnetic_field
    """
    # Get the resonance frequency
    omega = _resonance_frequency(
        isotope = isotope,
        B = parameters.magnetic_field,
        delta = offset,
        unit = unit
    )

    return omega

def pulse_and_acquire(
        spin_system: SpinSystem,
        isotope: str,
        center_frequency: float,
        npoints: int,
        dwell_time: float,
        angle: float       
) -> np.ndarray:
    """
    Simple pulse-and-acquire experiment.

    This experiment requires the following spin system properties to be defined:

    - spin_system.basis : must be built
    - spin_system.relaxation.theory
    - spin_system.relaxation.thermalization : must be True

    This experiment requires the following parameters to be defined:

    - parameters.magnetic_field : magnetic field of the spectrometer in Tesla
    - parameters.temperature : temperature of the sample in Kelvin

    Parameters
    ----------
    spin_system : SpinSystem
        Spin system to which the pulse-and-acquire experiment is performed.

    Returns
    -------
    fid : ndarray
        Free induction decay signal.
    """
    # Obtain the Liouvillian
    H = hamiltonian(spin_system)
    R = relaxation(spin_system)
    L = liouvillian(H, R)

    # Obtain the equilibrium state
    rho = equilibrium_state(spin_system)

    # Find indices of the isotopes to be measured
    indices = np.where(spin_system.isotopes == isotope)[0]

    # Apply pulse
    op_pulse = "+".join(f"I(y,{i})" for i in indices)
    Px = pulse(spin_system, op_pulse, angle)
    rho = Px @ rho

    # Construct the time propagator
    P = propagator(L=L, t=dwell_time)
    P = propagator_to_rotframe(
        spin_system = spin_system,
        P = P,
        t = dwell_time,
        center_frequencies = {isotope: center_frequency})

    # Initialize an array for storing results
    fid = np.zeros(npoints, dtype=complex)

    # Perform the time evolution
    op_measure = "+".join(f"I(-,{i})" for i in indices)
    for step in range(npoints):
        fid[step] = measure(spin_system, rho, op_measure)
        rho = P @ rho
    
    return fid

def inversion_recovery_fid():
    """
    TODO
    """

def inversion_recovery(
        spin_system: SpinSystem,
        isotope: str,
        npoints: int,
        time_step: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Performs the inversion-recovery experiment. The experiment differs slightly
    from the actual inversion-recovery experiments performed on spectrometers.
    In this experiment, the inversion is performed only once, and the
    magnetization is detected at each step during the recovery (much faster).
    
    If the traditional inversion recovery is desired, use the function
    `inversion_recovery_fid()`.

    This experiment requires the following spin system properties to be defined:

    - spin_system.basis : must be built
    - spin_system.relaxation.theory
    - spin_system.relaxation.thermalization : must be True

    This experiment requires the following parameters to be defined:

    - parameters.magnetic_field : magnetic field of the spectrometer in Tesla
    - parameters.temperature : temperature of the sample in Kelvin

    Parameters
    ----------
    spin_system : SpinSystem
        Spin system to which the inversion-recovery experiment is performed.
    isotope : str
        Specifies the isotope, for example "1H", whose magnetization is inverted
        and detected. This function applies hard pulses.
    npoints : int
        Number of points in the simulation. Defines the total simulation time
        together with `time_step`.
    time_step : float
        Time step in the simulation (in seconds). Should be kept relatively
        short (e.g. 1 ms).

    Returns
    -------
    magnetizations : ndarray
        Two-dimensional array of size (nspins, npoints) containing the
        observed z-magnetizations for each spin at various times.
    """
    # Operate on a copy of the SpinSystem object
    spin_system = deepcopy(spin_system)

    # Check that the required attributes have been set
    if spin_system.basis.basis is None:
        raise ValueError("Please build the basis before using "
                         "inversion recovery.")
    if spin_system.relaxation.theory is None:
        raise ValueError("Please set the relaxation theory before using "
                         "inversion recovery.")
    if spin_system.relaxation.thermalization is False:
        raise ValueError("Please set thermalization to True before using "
                         "inversion recovery.")
    if parameters.magnetic_field is None:
        raise ValueError("Please set the magnetic field before using "
                         "inversion recovery.")
    if parameters.temperature is None:
        raise ValueError("Please set the temperature before using "
                         "inversion recovery.")
    
    # Obtain the Liouvillian
    H = hamiltonian(spin_system)
    R = relaxation(spin_system)
    L = liouvillian(H, R)

    # Obtain the equilibrium state
    rho = equilibrium_state(spin_system)

    # Find indices of the isotopes to be measured
    indices = np.where(spin_system.isotopes == isotope)[0]
    nspins = indices.shape[0]

    # Apply 180-degree pulse
    operator = "+".join(f"I(x,{i})" for i in indices)
    P180 = pulse(spin_system, operator, 180)
    rho = P180 @ rho

    # Change to ZQ-basis to speed up the calculations
    L, rho = spin_system.basis.truncate_by_coherence([0], L, rho)

    # Construct the time propagator
    P = propagator(L, time_step)

    # Initialize an array for storing results
    magnetizations = np.zeros((nspins, npoints), dtype=complex)

    # Perform the time evolution
    for step in range(npoints):
        for i, idx in enumerate(indices):
            magnetizations[i, step] = measure(spin_system, rho, f"I(z,{idx})")
        rho = P @ rho

    return magnetizations

def time_axis(npoints: int, time_step: float):
    """
    Generates a 1D array with `npoints` elements using a constant `time_step`.

    Parameters
    ----------
    npoints : int
        Number of points.
    time_step : float
        Time step (in seconds).
    """
    # Obtain the time array
    start = 0
    stop = npoints * time_step
    num = npoints
    t_axis = np.linspace(start, stop, num, endpoint=False)

    return t_axis

def dissociate(spin_system_A: SpinSystem,
               spin_system_B: SpinSystem,
               spin_system_C: SpinSystem,
               rho_C: np.ndarray | sp.csc_array,
               spin_map_A: list | tuple | np.ndarray,
               spin_map_B: list | tuple | np.ndarray
               ) -> tuple[np.ndarray | sp.csc_array, np.ndarray | sp.csc_array]:
    """
    Dissociates the density vector of composite system C into density vectors of
    two subsystems A and B in a chemical reaction C -> A + B.

    Example. Spin system C has five spins, which are indexed as (0, 1, 2, 3, 4).
    We want to dissociate this into two subsystems A and B. Spins 0 and 2 should
    go to subsystem A and the rest to subsystem B. In this case, we define the
    following spin maps::

        spin_map_A = np.array([0, 2])
        spin_map_B = np.array([1, 3, 4])

    Parameters
    ----------
    spin_system_A : SpinSystem
        Spin system A.
    spin_system_B : SpinSystem
        Spin system B.
    spin_system_C : SpinSystem
        Spin system C.
    rho_C : ndarray or csc_array
        State vector of the composite spin system C.
    spin_map_A : list or tuple or ndarray
        Indices of spin system A within spin system C.
    spin_map_B : list or tuple or ndarray
        Indices of spin system B within spin system C.

    Returns
    -------
    rho_A : ndarray or csc_array
        State vector of spin system A.
    rho_B : ndarray or csc_array
        State vector of spin system B.
    """
    # Perform the dissociation
    rho_A, rho_B = _dissociate(
        basis_A = spin_system_A.basis.basis,
        basis_B = spin_system_B.basis.basis,
        basis_C = spin_system_C.basis.basis,
        spins_A = spin_system_A.spins,
        spins_B = spin_system_B.spins,
        rho_C = rho_C,
        spin_map_A = spin_map_A,
        spin_map_B = spin_map_B
    )

    return rho_A, rho_B

def associate(spin_system_A: SpinSystem,
              spin_system_B: SpinSystem,
              spin_system_C: SpinSystem,
              rho_A: np.ndarray | sp.csc_array,
              rho_B: np.ndarray | sp.csc_array,
              spin_map_A: list | tuple | np.ndarray,
              spin_map_B: list | tuple | np.ndarray
) -> np.ndarray | sp.csc_array:
    """
    Combines two state vectors when spin systems associate in a chemical
    reaction A + B -> C.

    Example. We have spin system A that has two spins and spin system B that has
    three spins. These systems associate to form a composite spin system C that
    has five spins that are indexed (0, 1, 2, 3, 4). Of these, spins (0, 2) are
    from subsystem A and (1, 3, 4) from subsystem B. We have to choose how the
    spin systems A and B will be indexed in spin system C by defining the spin
    maps as follows::

        spin_map_A = np.ndarray([0, 2])
        spin_map_B = np.ndarray([1, 3, 4])

    Parameters
    ----------
    spin_system_A : SpinSystem
        Spin system A.
    spin_system_B : SpinSystem
        Spin system B.
    spin_system_C : SpinSystem
        Spin system C.
    rho_A : ndarray or csc_array
        State vector of spin system A.
    rho_B : ndarray or csc_array
        State vector of spin system B.
    spin_map_A : list or tuple or ndarray
        Indices of spin system A within spin system C.
    spin_map_B : list or tuple or ndarray
        Indices of spin system B within spin system C.

    Returns
    -------
    rho_C : ndarray or csc_array
        State vector of the composite spin system C.
    """
    # Perform the association
    rho_C = _associate(
        basis_A = spin_system_A.basis.basis,
        basis_B = spin_system_B.basis.basis,
        basis_C = spin_system_C.basis.basis,
        rho_A = rho_A,
        rho_B = rho_B,
        spin_map_A = spin_map_A,
        spin_map_B = spin_map_B
    )

    return rho_C

def permute_spins(spin_system: SpinSystem,
                  rho: np.ndarray | sp.csc_array,
                  spin_map: list | tuple | np.ndarray
) -> np.ndarray | sp.csc_array:
    """
    Permutes the state vector of a spin system to correspond to a reordering
    of the spins in the system. 

    Example. Our spin system has three spins, which are indexed (0, 1, 2). We
    want to perform the following permulation:

    - 0 --> 2 (Spin 0 goes to position 2)
    - 1 --> 0 (Spin 1 goes to position 0)
    - 2 --> 1 (Spin 2 goes to position 1)

    In this case, we want to assign the following map::

        spin_map = np.array([2, 0, 1])

    Parameters
    ----------
    spin_system : SpinSystem
        The spin system whose density vector is going to be permuted.
    rho : ndarray or csc_array
        State vector of the spin system.
    spin_map : list or tuple or ndarray
        Indices of the spins in the spin system after permutation.

    Returns
    -------
    rho : ndarray or csc_array
        Permuted state vector of the spin system.
    """
    # Perform the permutation
    rho = _permute_spins(
        basis = spin_system.basis.basis,
        rho = rho,
        spin_map = spin_map
    )

    return rho

def unit_state(spin_system: SpinSystem,
               normalized: bool=True) -> np.ndarray | sp.csc_array:
    """
    Returns a unit state vector, which represents the identity operator. The
    output can be either normalised (trace equal to one) or unnormalised (raw
    identity matrix).

    Parameters
    ----------
    spin_system : SpinSystem
        Spin system to which the unit state is created.
    normalized : bool, default=True
        If set to True, the function will return a state vector that represents
        the trace-normalized density matrix. If False, returns a state vector
        that corresponds to the identity operator.

    Returns
    -------
    rho : ndarray or csc_array
        State vector corresponding to the unit state.
    """
    # Create the unit state
    rho = _unit_state(
        basis = spin_system.basis.basis,
        spins = spin_system.spins,
        sparse = parameters.sparse_state,
        normalized = normalized
    )

    return rho

def state(spin_system: SpinSystem,
          operator: str) -> np.ndarray | sp.csc_array:
    """
    This function returns a column vector representing the density matrix as a
    linear combination of spin operators. Each element of the vector corresponds
    to the coefficient of a specific spin operator in the expansion.
    
    Normalization:
    The output of this function uses a normalised basis built from normalised
    products of single-spin spherical tensor operators. However, the
    coefficients are scaled so that the resulting linear combination represents
    the non-normalised version of the requested operator.

    NOTE: This function is sometimes called often and is cached for high
    performance.

    Parameters
    ----------
    spin_system : SpinSystem
        Spin system for which the state is created.
    operator : str
        Defines the state to be generated. The operator string must follow the
        rules below:

        - Cartesian and ladder operators: `I(component,index)` or
          `I(component)`. Examples:

            - `I(x,4)` --> Creates x-operator for spin at index 4.
            - `I(x)`--> Creates x-operator for all spins.

        - Spherical tensor operators: `T(l,q,index)` or `T(l,q)`. Examples:

            - `T(1,-1,3)` --> \
              Creates operator with `l=1`, `q=-1` for spin at index 3.
            - `T(1, -1)` --> \
              Creates operator with `l=1`, `q=-1` for all spins.
            
        - Product operators have `*` in between the single-spin operators:
          `I(z,0) * I(z,1)`
        - Sums of operators have `+` in between the operators:
          `I(x,0) + I(x,1)`
        - Unit operators are ignored in the input. Interpretation of these
          two is identical: `E * I(z,1)`, `I(z,1)`
        
        Special case: An empty `operator` string is considered as unit operator.

        Whitespace will be ignored in the input.

        NOTE: Indexing starts from 0!

    Returns
    -------
    rho : ndarray or csc_array
        State vector corresponding to the requested state.
    """
    # Check that the required attributes are set
    if spin_system.basis.basis is None:
        raise ValueError("Please build the basis before constructing a state.")
    
    # Build the state
    rho = _state_from_string(
        basis = spin_system.basis.basis,
        spins = spin_system.spins,
        operator = operator,
        sparse = parameters.sparse_state
    )

    return rho

def state_to_zeeman(
        spin_system: SpinSystem,
        rho: np.ndarray | sp.csc_array
        ) -> np.ndarray | sp.csc_array:
    """
    Takes the state vector defined in the normalized spherical tensor basis
    and converts it into the Zeeman eigenbasis. Useful for error checking.

    Parameters
    ----------
    spin_system : SpinSystem
        Spin system whose state vector is going to be converted into a density
        matrix.
    rho : ndarray or csc_array
        State vector defined in the normalized spherical tensor basis.

    Returns
    -------
    rho_zeeman : ndarray or csc_array
        Spin density matrix defined in the Zeeman eigenbasis.
    """
    # Check that the required attributes are set
    if spin_system.basis.basis is None:
        raise ValueError("Please build the basis before converting the "
                         "state vector into density matrix.")
    
    # Convert the state vector into density matrix
    rho_zeeman = _state_to_zeeman(
        basis = spin_system.basis.basis,
        spins = spin_system.spins,
        rho = rho,
        sparse = parameters.sparse_state
    )
    
    return rho_zeeman

def alpha_state(spin_system: SpinSystem,
                index: int) -> np.ndarray | sp.csc_array:
    """
    Generates the alpha state for a given spin-1/2 nucleus. Unit state is
    assigned to the other spins.

    Parameters
    ----------
    spin_system: SpinSystem
        Spin system for which the alpha state is created.
    index : int
        Index of the spin that has the alpha state.

    Returns
    -------
    rho : ndarray or csc_array
        State vector corresponding to the alpha state of the given spin index.
    """
    # Check that the required attributes are set
    if spin_system.basis.basis is None:
        raise ValueError("Please build the basis before constructing the "
                         "alpha state.")
    
    # Create the alpha state
    rho = _alpha_state(
        basis = spin_system.basis.basis,
        spins = spin_system.spins,
        index = index,
        sparse = parameters.sparse_state
    )

    return rho

def beta_state(spin_system: SpinSystem,
               index: int) -> np.ndarray | sp.csc_array:
    """
    Generates the beta state for a given spin-1/2 nucleus. Unit state is
    assigned to the other spins.

    Parameters
    ----------
    spin_system: SpinSystem
        Spin system for which the beta state is created.
    index : int
        Index of the spin that has the beta state.

    Returns
    -------
    rho : ndarray or csc_array
        State vector corresponding to the beta state of the given spin index.
    """
    # Check that the required attributes are set
    if spin_system.basis.basis is None:
        raise ValueError("Please build the basis before constructing the "
                         "beta state.")
    
    # Create the beta state
    rho = _beta_state(
        basis = spin_system.basis.basis,
        spins = spin_system.spins,
        index = index,
        sparse = parameters.sparse_state
    )

    return rho

def triplet_zero_state(spin_system: SpinSystem,
                       index_1: int,
                       index_2: int) -> np.ndarray | sp.csc_array:
    """
    Generates the triplet zero state between two spin-1/2 nuclei. Unit state is
    assigned to the other spins.

    Parameters
    ----------
    spin_system: SpinSystem
        Spin system for which the triplet zero state is created.
    index_1 : int
        Index of the first spin in the triplet zero state.
    index_2 : int
        Index of the second spin in the triplet zero state.

    Returns
    -------
    rho : ndarray or csc_array
        State vector corresponding to the triplet zero state.
    """
    # Check that the required attributes are set
    if spin_system.basis.basis is None:
        raise ValueError("Please build the basis before constructing the "
                         "triplet zero state.")
    
    # Make the triplet zero state
    rho = _triplet_zero_state(
        basis = spin_system.basis.basis,
        spins = spin_system.spins,
        index_1 = index_1,
        index_2 = index_2,
        sparse = parameters.sparse_state
    )

    return rho

def triplet_plus_state(spin_system: SpinSystem,
                       index_1: int,
                       index_2: int) -> np.ndarray | sp.csc_array:
    """
    Generates the triplet plus state between two spin-1/2 nuclei. Unit state is
    assigned to the other spins.

    Parameters
    ----------
    spin_system : SpinSystem
        Spin system for which the triplet plus state is created.
    index_1 : int
        Index of the first spin in the triplet plus state.
    index_2 : int
        Index of the second spin in the triplet plus state.

    Returns
    -------
    rho : ndarray or csc_array
        State vector corresponding to the triplet plus state.
    """
    # Check that the required attributes are set
    if spin_system.basis.basis is None:
        raise ValueError("Please build the basis before constructing the "
                         "triplet plus state.")
    
    # Create the triplet plus state
    rho = _triplet_plus_state(
        basis = spin_system.basis.basis,
        spins = spin_system.spins,
        index_1 = index_1,
        index_2 = index_2,
        sparse = parameters.sparse_state
    )

    return rho

def triplet_minus_state(spin_system: SpinSystem,
                        index_1: int,
                        index_2: int) -> np.ndarray | sp.csc_array:
    """
    Generates the triplet minus state between two spin-1/2 nuclei. Unit state is
    assigned to the other spins.

    Parameters
    ----------
    spin_system : SpinSystem
        Spin system for which the triplet minus state is created.
    index_1 : int
        Index of the first spin in the triplet minus state.
    index_2 : int
        Index of the second spin in the triplet minus state.

    Returns
    -------
    rho : ndarray or csc_array
        State vector corresponding to the triplet minus state.
    """
    # Check that the required attributes are set
    if spin_system.basis.basis is None:
        raise ValueError("Please build the basis before constructing the "
                         "triplet minus state.")
    
    # Create the triplet minus state
    rho = _triplet_minus_state(
        basis = spin_system.basis.basis,
        spins = spin_system.spins,
        index_1 = index_1,
        index_2 = index_2,
        sparse = parameters.sparse_state
    )

    return rho