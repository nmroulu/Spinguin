"""
core.py

This module provides user-friendly wrapper functions for the core functionality
of the Spinguin package.
"""

# Imports
import numpy as np
import scipy.sparse as sp
from typing import Literal

from spinguin._core._config import config
from spinguin._core._parameters import parameters
from spinguin._core._spin_system import SpinSystem
from spinguin._core._hamiltonian import hamiltonian
from spinguin._core._hamiltonian import _sop_H
from spinguin.utils import HidePrints
from spinguin._core._liouvillian import (
    sop_L as liouvillian,
)
from spinguin._core._nmr_isotopes import gamma, quadrupole_moment, spin
from spinguin._core._propagation import (
    propagator_to_rotframe as _propagator_to_rotframe,
    sop_pulse as _sop_pulse,
    propagator
)
from spinguin._core._specutils import (
    frequency_to_chemical_shift,
    resonance_frequency as _resonance_frequency,
    spectral_width_to_dwell_time as _spectral_width_to_dwell_time,
    spectrum as _spectrum
)
from spinguin._core._superoperators import superoperator

__all__ = [
    "associate",
    "dissociate",
    "frequency_to_chemical_shift",
    "gamma",
    "hamiltonian",
    "inversion_recovery",
    "liouvillian",
    "measure",
    "permute_spins",
    "propagator",
    "propagator_to_rotframe",
    "pulse",
    "pulse_and_acquire",
    "quadrupole_moment",
    "resonance_frequency",
    "rotating_frame"
    "spectral_width_to_dwell_time",
    "spectrum",
    "spin",
    "superoperator",
    "time_axis",
]

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
        sparse = config.sparse_pulse,
        zero_value = config.zero_pulse
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
        sparse = config.sparse_hamiltonian,
        zero_value = config.zero_hamiltonian
    )

    # Convert the propagator to rotating frame
    P_rot = _propagator_to_rotframe(
        sop_P = P,
        sop_H0 = H_frame,
        t = t,
        zero_value = config.zero_propagator
    )
    
    return P_rot

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