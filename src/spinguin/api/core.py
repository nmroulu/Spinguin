"""
core.py

This module provides user-friendly wrapper functions for the core functionality
of the Spinguin package.
"""

# Imports
import numpy as np
import scipy.sparse as sp
from typing import Literal
from spinguin.core.operators import op_from_string
from spinguin.core.superoperators import sop_from_string
from spinguin.core.hamiltonian import sop_H as csop_H
from spinguin.core.relaxation import sop_R_phenomenological, sop_R_redfield, \
    sop_R_sr2k, ldb_thermalization
from spinguin.core import states, propagation, specutils
from spinguin.core.liouvillian import sop_L
from spinguin.api.parameters import parameters
from spinguin.api.config import config
from spinguin.api.spin_system import SpinSystem
from spinguin.core.hide_prints import HidePrints

def spin_system(isotopes: list | tuple | np.ndarray | str) -> SpinSystem:
    """
    Initializes a new `SpinSystem` object.

    Parameters
    ----------
    isotopes : list or tuple or ndarray or str
        Specifies the isotopes that constitute the spin system and determine
        other properties, such as spin quantum numbers and gyromagnetic ratios.
        Two input types are supported:

        - If `ArrayLike`: A 1D array of size N containing isotope names as
          strings. Example:

        ```python
        np.array(['1H', '15N', '19F'])
        ```

        - If `str`: Path to the file containing the isotopes.

        The input will be stored as a NumPy array.

    Returns
    -------
    spin_system : SpinSystem
        The presently initialized `SpinSystem` instance.
    """
    # Create a new SpinSystem instance
    spin_system = SpinSystem(isotopes=isotopes)

    return spin_system

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
    op = op_from_string(spins = spin_system.spins,
                        operator = operator,
                        sparse = config.sparse_operator)
    
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
    sop = sop_from_string(operator = operator,
                          basis = spin_system.basis.basis,
                          spins = spin_system.spins,
                          side = side,
                          sparse = config.sparse_superoperator)
        
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
        
    H = csop_H(
        basis = spin_system.basis.basis,
        spins = spin_system.spins,
        gammas = spin_system.gammas,
        B = parameters.magnetic_field,
        chemical_shifts = spin_system.chemical_shifts,
        J_couplings = spin_system.J_couplings,
        interactions = interactions,
        side = side,
        sparse = config.sparse_hamiltonian,
        zero_value = config.zero_hamiltonian
    )

    return H

def relaxation(spin_system: SpinSystem) -> np.ndarray | sp.csc_array:
    """
    Creates the relaxation superoperator using the requested relaxation theory.

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
        R = sop_R_phenomenological(
            basis = spin_system.basis.basis,
            R1 = spin_system.relaxation.R1,
            R2 = spin_system.relaxation.R2,
            sparse = config.sparse_relaxation)

    # Make relaxation superoperator using Redfield theory
    elif spin_system.relaxation.theory == "redfield":
        
        # Build the coherent hamiltonian
        with HidePrints():
            H = hamiltonian(spin_system=spin_system,
                            interactions=INTERACTIONDEFAULT,
                            side="comm")

        # Build the Redfield relaxation superoperator
        R = sop_R_redfield(
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
            interaction_zero = config.zero_interaction,
            aux_zero = config.zero_aux,
            relaxation_zero = config.zero_relaxation,
            sparse = config.sparse_relaxation
        )
    
    # Apply scalar relaxation of the second kind if requested
    if spin_system.relaxation.sr2k:
        R += sop_R_sr2k(
            basis = spin_system.basis.basis,
            spins = spin_system.spins,
            gammas = spin_system.gammas,
            chemical_shifts = spin_system.chemical_shifts,
            J_couplings = spin_system.J_couplings,
            sop_R = R,
            B = parameters.magnetic_field,
            sparse = config.sparse_relaxation
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
        R = ldb_thermalization(
            R = R,
            H_left = H_left,
            T = parameters.temperature,
            zero_value = config.zero_thermalization)

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
    rho = states.equilibrium_state(
        basis = spin_system.basis.basis,
        spins = spin_system.spins,
        H_left = H_left,
        T = parameters.temperature,
        sparse = config.sparse_state,
        zero_value = config.zero_equilibrium
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
    P = propagation.sop_pulse(
        basis = spin_system.basis.basis,
        spins = spin_system.spins,
        operator = operator,
        angle = angle,
        sparse = config.sparse_pulse,
        zero_value = config.zero_pulse
    )

    return P

def liouvillian(H: np.ndarray | sp.csc_array = None,
                R: np.ndarray | sp.csc_array = None,
                K: np.ndarray | sp.csc_array = None
                ) -> np.ndarray | sp.csc_array:
    """
    Constructs the Liouvillian superoperator from the Hamiltonian, relaxation
    superoperator, and exchange superoperator.

    Parameters
    ----------
    H : ndarray or csc_array
        Hamiltonian superoperator.
    R : ndarray or csc_array
        Relaxation superoperator
    K : ndarray or csc_array
        Exchange superoperator.

    Returns
    -------
    L : ndarray or csc_array
        Liouvillian superoperator.
    """

    # Construct the Liouvillian
    L = sop_L(H, R, K)

    return L

def propagator(L: np.ndarray | sp.csc_array,
               t: float) -> np.ndarray | sp.csc_array:
    """
    TODO
    """
    # Create the propagator
    P = propagation.sop_propagator(
        L = L,
        t = t,
        custom_dot = config.custom_dot,
        zero_value = config.zero_propagator,
        density_threshold = config.propagator_density
    )
    
    return P

def propagator_to_rotframe(spin_system: SpinSystem,
                           P: np.ndarray | sp.csc_array,
                           t: float,
                           center_frequencies: dict=None
                           ) -> np.ndarray | sp.csc_array:
    """
    TODO
    """
    # Obtain an array of center frequencies for each spin
    center = np.zeros(spin_system.nspins)
    for spin in range(spin_system.nspins):
        if spin_system.isotopes[spin] in center_frequencies:
            center[spin] = center_frequencies[spin_system.isotopes[spin]]

    # Construct Hamiltonian that specifies the interaction frame
    H_frame = csop_H(
        basis = spin_system.basis.basis,
        spins = spin_system.spins,
        gammas = spin_system.gammas,
        B = parameters.magnetic_field,
        chemical_shifts = center,
        interactions = ["zeeman", "chemical_shift"],
        side = "comm",
        sparse = config.sparse_hamiltonian,
        zero_value = config.zero_hamiltonian
    )

    # Convert the propagator to rotating frame
    P = propagation.propagator_to_rotframe(
        sop_P = P,
        sop_H0 = H_frame,
        t = t,
        zero_value = config.zero_propagator,
        custom_dot = config.custom_dot
    )
    
    return P

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
    ex : complex
        Expectation value.
    """
    # Perform the measurement
    ex = states.measure(
        basis = spin_system.basis.basis,
        spins = spin_system.spins,
        rho = rho,
        operator = operator
    )

    return ex

def spectral_width_to_dwell_time(spectral_width: float,
                                 isotope: str) -> float:
    """
    Calculates the dwell time (in seconds) from the spectral width given in ppm.

    Parameters
    ----------
    spectral_width : float
        Spectral width in ppm.
    isotope : str
        Nucleus symbol (e.g. `'1H'`) used to select the gyromagnetic ratio 
        required for the conversion.

    Returns
    -------
    dwell_time : float
        Dwell time in seconds.
    """
    # Obtain the dwell time
    dwell_time = specutils.spectral_width_to_dwell_time(
        spectral_width = spectral_width,
        isotope = isotope,
        B = parameters.magnetic_field
    )

    return dwell_time

def spectrum(signal: np.ndarray,
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
    dt : float
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
    freqs, spectrum = specutils.spectrum(
        signal = signal,
        dt = parameters.dwell_time[-1],
        normalize = normalize,
        part = part
    )

    return freqs, spectrum

def resonance_frequency(isotope: str,
                        delta: float = 0,
                        unit: Literal["Hz", "rad/s"] = "Hz") -> float:
    """
    Computes the resonance frequency of a nucleus at specified magnetic field
    and chemical shift.

    Parameters
    ----------
    isotope : str
        Nucleus symbol (e.g. `'1H'`) used to select the gyromagnetic ratio.
    delta : float, default=0
        Chemical shift in ppm.
    units :{'Hz', 'rad/s'}
        Specifies in which units the frequency is returned.

    Returns
    -------
    omega : float
        Resonance frequency of the given nucleus.
    """
    # Calculate the resonance frequency
    omega = specutils.resonance_frequency(
        isotope = isotope,
        B = parameters.magnetic_field,
        delta = delta,
        unit = unit
    )

    return omega

def frequency_to_chemical_shift(
        frequency: float | np.ndarray) -> float | np.ndarray:
    """
    Converts a frequency (or an array of frequencies, e.g., a frequency axis) to
    a chemical shift value.

    Parameters
    ----------
    frequency : float or ndarray
        Frequency (or array of frequencies) to convert [in Hz].

    Returns
    -------
    chemical_shift : float or ndarray
        Converted chemical shift value (or array of values).

    Notes
    -----
    Required global parameters:
    - parameters.isotope
    - parameters.magnetic_field
    - parameters.center_frequency
    """
    # Obtain the rotating frame frequency
    freq_rotframe = spectrometer_frequency("Hz", True)

    # Add back the rotating frame frequency
    frequency = frequency + freq_rotframe

    # Obtain the base spectrometer frequency (reference frequency)
    freq_spectrometer = spectrometer_frequency("Hz", False)

    # Obtain the chemical shift
    chemical_shift = specutils.frequency_to_chemical_shift(
        frequency = frequency,
        reference_frequency = freq_spectrometer,
        spectrometer_frequency = freq_spectrometer
    )
    return chemical_shift

def pulse_and_acquire(
        spin_system: SpinSystem
) -> np.ndarray:
    """
    Simple pulse-and-acquire experiment.

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
    indices = np.where(spin_system.isotopes == parameters.isotope[0])[0]

    # Apply 180-degree pulse
    op_pulse = "+".join(f"I(y,{i})" for i in indices)
    Px = pulse(spin_system, op_pulse, parameters.angle[0])
    rho = Px @ rho

    # Construct the time propagator
    dt = parameters.dwell_time[0]
    P = propagator(L, dt)
    P = propagator_to_rotframe(
        spin_system = spin_system,
        P = P,
        t = parameters.dwell_time[0],
        center_frequencies = parameters.center_frequency)

    # Initialize an array for storing results
    npoints = parameters.npoints[0]
    fid = np.zeros(npoints, dtype=complex)

    # Perform the time evolution
    op_measure = "+".join(f"I(-,{i})" for i in indices)
    for step in range(npoints):
        fid[step] = measure(spin_system, rho, op_measure)
        rho = P @ rho
    
    return fid

def inversion_recovery(
        spin_system: SpinSystem) -> tuple[np.ndarray, np.ndarray]:
    """
    Performs the inversion-recovery experiment. The experiment differs slightly
    from the actual inversion-recovery experiments performed on spectrometers.
    In this experiment, the inversion is performed only once, and the
    magnetization is detected at each step during the recovery.

    This experiment requires the following spin system properties to be defined:
    - basis : must be built
    - relaxation.theory
    - relaxation.thermalization : must be True

    This experiment requires the following parameters to be defined:
    - magnetic_field : magnetic field of the spectrometer in Tesla
    - temperature : temperature of the sample in Kelvin
    - isotope : nucleus to-be-measured
    - dwell_time : time step in the simulation in seconds
    - npoints : number of time steps

    Parameters
    ----------
    spin_system : SpinSystem
        Spin system to which the inversion-recovery experiment is performed.

    Returns
    -------
    magnetizations : ndarray
        Two-dimensional array of size (nspins, npoints) containing the
        observed z-magnetizations for each spin at various times.
    """
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
    if parameters.isotope is None:
        raise ValueError("Please define isotope when using inversion recovery.")
    if parameters.dwell_time is None:
        raise ValueError("Please define dwell time when using inversion "
                         "recovery.")
    if parameters.npoints is None:
        raise ValueError("Please define number of points when using "
                         "inversion recovery.")
    
    # Obtain the Liouvillian
    H = hamiltonian(spin_system)
    R = relaxation(spin_system)
    L = liouvillian(H, R)

    # Obtain the equilibrium state
    rho = equilibrium_state(spin_system)

    # Find indices of the isotopes to be measured
    indices = np.where(spin_system.isotopes == parameters.isotope[0])[0]
    nspins = indices.shape[0]

    # Apply 180-degree pulse
    operator = "+".join(f"I(x,{i})" for i in indices)
    P180 = pulse(spin_system, operator, 180)
    rho = P180 @ rho

    # Change to ZQ-basis to speed up the calculations
    L, rho = spin_system.basis.truncate_by_coherence([0], L, rho)

    # Construct the time propagator
    dt = parameters.dwell_time[0]
    P = propagator(L, dt)

    # Initialize an array for storing results
    npoints = parameters.npoints[0]
    magnetizations = np.zeros((nspins, npoints), dtype=complex)

    # Perform the time evolution
    for step in range(npoints):
        for i, idx in enumerate(indices):
            magnetizations[i, step] = measure(spin_system, rho, f"I(z,{idx})")
        rho = P @ rho

    return magnetizations

def time_axis():
    """
    Generates the time axis for an FID signal.

    Notes
    -----
    Required global parameters:
    - parameters.npoints
    - parameters.dwell_time
    """
    # Obtain the time array
    start = 0
    stop = parameters.npoints[-1] * parameters.dwell_time[-1]
    num = parameters.npoints[-1]
    t_axis = np.linspace(start, stop, num, endpoint=False)

    return t_axis