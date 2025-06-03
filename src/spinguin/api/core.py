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
from spinguin.core.hamiltonian import sop_H_coherent, sop_H_CS, sop_H_J, sop_H_Z
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

INTERACTIONS = ["zeeman", "chemical_shift", "J_coupling"]
def hamiltonian(spin_system: SpinSystem,
                interactions: list[Literal[
                    "zeeman", "chemical_shift", "J_coupling"]] = INTERACTIONS,
                side: Literal["comm", "left", "right"] = "comm"
                ) -> np.ndarray | sp.csc_array:
    """
    Creates the requested Hamiltonian superoperator for the spin system.

    Parameters
    ----------
    spin_system : SpinSystem
        Spin system for which the Hamiltonian is going to be generated.
    interactions : list, default=["zeeman", "chemical_shift", "J_coupling"]
        Specifies which interactions are taken into account.
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
        
    # Check that basis has been set
    if spin_system.basis.basis is None:
        raise ValueError("Please build basis before constructing " 
                         "the Hamiltonian.")
    
    # Initialize the Hamiltonian
    dim = spin_system.basis.dim
    if config.sparse_hamiltonian:
        H = sp.csc_array((dim, dim), dtype=complex)
    else:
        H = np.zeros((dim, dim), dtype=complex)

    # Go through the requested interactions
    for interaction in interactions:

        if interaction == "zeeman":

            # Check that the required attributes have been set
            if parameters.magnetic_field is None:
                raise ValueError("Please set the magnetic field before "
                                 "constructing the Zeeman Hamiltonian.")

            # Construct the Hamiltonian
            H += sop_H_Z(basis = spin_system.basis.basis,
                         gammas = spin_system.gammas,
                         spins = spin_system.spins,
                         B = parameters.magnetic_field,
                         side = side,
                         sparse = config.sparse_hamiltonian)
            
        elif interaction == "chemical_shift":

            # Check that the required attributes have been set
            if parameters.magnetic_field is None:
                raise ValueError("Please set the magnetic field before "
                                 "constructing the chemical shift "
                                 "Hamiltonian.")
            
            # Construct the Hamiltonian
            H += sop_H_CS(basis = spin_system.basis.basis,
                          gammas = spin_system.gammas,
                          spins = spin_system.spins,
                          chemical_shifts = spin_system.chemical_shifts,
                          B = parameters.magnetic_field,
                          side = side,
                          sparse = config.sparse_hamiltonian)
            
        elif interaction == "J_coupling":   

            # Construct the Hamiltonian
            H += sop_H_J(basis = spin_system.basis.basis,
                         spins = spin_system.spins,
                         J_couplings = spin_system.J_couplings,
                         side = side,
                         sparse = config.sparse_hamiltonian)
            
        else:
            raise ValueError(f"Unknown interaction type: {interaction}. " 
                             f"The available options are: {INTERACTIONS}.")

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
        raise ValueError("Please set the relaxation theory before "
                         "constructing the relaxation superoperator.")
    if spin_system.basis.basis is None:
        raise ValueError("Please build basis before constructing the "
                         "relaxation superoperator.")

    # Make phenomenological relaxation superoperator
    if spin_system.relaxation.theory == "phenomenological":

        if spin_system.relaxation.T1 is None:
            raise ValueError("Please set T1 times before constructing the "
                             "relaxation superoperator.")
        if spin_system.relaxation.T2 is None:
            raise ValueError("Please set T2 times before constructing the "
                             "relaxation superoperator.")

        # Build the phenomenological relaxation superoperator
        R = sop_R_phenomenological(
            basis = spin_system.basis.basis,
            R1 = spin_system.relaxation.R1,
            R2 = spin_system.relaxation.R2,
            sparse = config.sparse_relaxation)

    # Make relaxation superoperator using Redfield theory
    elif spin_system.relaxation.theory == "redfield":
        
        if spin_system.relaxation.tau_c is None:
            raise ValueError("Please set the correlation time before "
                             "constructing the Redfield relaxation "
                             "superoperator.")
        if parameters.magnetic_field is None:
            raise ValueError("Please set the magnetic field before "
                             "constructing the Redfield relaxation "
                             "superoperator.")
        
        # Build the coherent hamiltonian
        with HidePrints():
            H = hamiltonian(spin_system=spin_system,
                            interactions=INTERACTIONS,
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
        R = R + sop_R_sr2k(basis = spin_system.basis.basis,
                            spins = spin_system.spins,
                            gammas = spin_system.gammas,
                            chemical_shifts = spin_system.chemical_shifts,
                            J_couplings = spin_system.J_couplings,
                            sop_R = R,
                            B = parameters.magnetic_field,
                            sparse = config.sparse_relaxation)
        
    # Apply thermalization if requested
    if spin_system.relaxation.thermalization:

        # Check that the required attributes have been set
        if parameters.magnetic_field is None:
            raise ValueError("Please set the magnetic field when applying "
                             "thermalization.")
        
        # Build the left Hamiltonian superopertor
        with HidePrints():
            H_left = sop_H_coherent(
                basis = spin_system.basis.basis,
                gammas = spin_system.gammas,
                spins = spin_system.spins,
                chemical_shifts = spin_system.chemical_shifts,
                J_couplings = spin_system.J_couplings,
                B = parameters.magnetic_field,
                side = "left",
                sparse = config.sparse_hamiltonian,
                zero_value = config.zero_hamiltonian)
            
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
        H_left = sop_H_coherent(
            basis = spin_system.basis.basis,
            gammas = spin_system.gammas,
            spins = spin_system.spins,
            chemical_shifts = spin_system.chemical_shifts,
            J_couplings = spin_system.J_couplings,
            B = parameters.magnetic_field,
            side = "left",
            sparse = config.sparse_hamiltonian,
            zero_value = config.zero_hamiltonian)

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
    # Construct Hamiltonian that specifies the interaction frame
    Hz = sop_H_Z(basis = spin_system.basis.basis,
                 gammas = spin_system.gammas,
                 spins = spin_system.spins,
                 B = parameters.magnetic_field,
                 side = "comm",
                 sparse = config.sparse_hamiltonian)
    
    center = np.zeros(spin_system.nspins)
    for spin in range(spin_system.nspins):
        if spin_system.isotopes[spin] in center_frequencies:
            center[spin] = center_frequencies[spin_system.isotopes[spin]]

    Hcenter = sop_H_CS(basis = spin_system.basis.basis,
                       gammas = spin_system.gammas,
                       spins = spin_system.spins,
                       chemical_shifts = center,
                       B = parameters.magnetic_field,
                       side = "comm",
                       sparse = config.sparse_hamiltonian)
    
    H_frame = Hz + Hcenter

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
             dt: float,
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
    """
    # Compute the Fourier transform
    freqs, spectrum = specutils.spectrum(
        signal = signal,
        dt = dt,
        normalize = normalize,
        part = part
    )

    return freqs, spectrum

def resonance_frequency(isotope: str,
                        delta: float = 0,
                        units: Literal["Hz", "rad/s"] = "Hz") -> float:
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
        units = units
    )

    return omega

def frequency_to_chemical_shift(frequency: float | np.ndarray, 
                                reference_frequency: float,
                                spectrometer_frequency: float
                                ) -> float | np.ndarray:
    """
    Converts a frequency (or an array of frequencies, e.g., a frequency axis) to
    a chemical shift value based on the reference frequency and the spectrometer
    frequency.

    Parameters
    ----------
    frequency : float or ndarray
        Frequency (or array of frequencies) to convert [in Hz].
    reference_frequency : float
        Reference frequency for the conversion [in Hz].
    spectrometer_frequency : float
        Spectrometer frequency for the conversion [in Hz].

    Returns
    -------
    chemical_shift : float or ndarray
        Converted chemical shift value (or array of values).
    """
    # Obtain the chemical shift
    chemical_shift = specutils.frequency_to_chemical_shift(
        frequency = frequency,
        reference_frequency = reference_frequency,
        spectrometer_frequency = spectrometer_frequency
    )
    return chemical_shift
