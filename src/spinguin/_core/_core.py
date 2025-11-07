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

from spinguin._core._parameters import parameters
from spinguin._core._spin_system import SpinSystem
from spinguin._core._hamiltonian import hamiltonian
from spinguin._core._liouvillian import sop_L as liouvillian
from spinguin._core._nmr_isotopes import gamma, quadrupole_moment, spin
from spinguin._core._operators import op_from_string as _op_from_string
from spinguin._core._propagation import (
    propagator_to_rotframe,
    propagator,
    pulse
)
from spinguin._core._relaxation import relaxation
from spinguin._core._specutils import (
    frequency_to_chemical_shift,
    resonance_frequency as _resonance_frequency,
    spectral_width_to_dwell_time as _spectral_width_to_dwell_time,
    spectrum as _spectrum
)
from spinguin._core._states import (
    equilibrium_state,
    measure
)

__all__ = [
    "frequency_to_chemical_shift",
    "gamma",
    "inversion_recovery",
    "liouvillian",
    "pulse_and_acquire",
    "quadrupole_moment",
    "resonance_frequency",
    "spectral_width_to_dwell_time",
    "spectrum",
    "spin",
    "time_axis",
]

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