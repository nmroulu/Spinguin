"""
This module provides core functions for spectral data analysis, including
Fourier transforms, spectrum generation, and unit conversions commonly used in
NMR and signal processing.
"""

# Imports
import numpy as np
from typing import Literal
from spinguin._core._nmr_isotopes import gamma
from spinguin._core._parameters import parameters

def resonance_frequency(
    isotope: str,
    B: float = None,
    delta: float = 0,
    unit: Literal["Hz", "rad/s"] = "Hz"
) -> float:
    """
    Computes the resonance frequency of a nucleus at specified magnetic field
    and chemical shift.

    Parameters
    ----------
    isotope : str
        Nucleus symbol (e.g. `'1H'`) used to select the gyromagnetic ratio.
    B : float, default=None
        Magnetic field strength in the units of T. If not supplied, the function 
        uses the magnetic field determined in parameters.magnetic_field.
    delta : float, default=0
        Chemical shift in ppm.
    unit :{'Hz', 'rad/s'}
        Specifies in which units the frequency is returned.

    Returns
    -------
    omega : float
        Resonance frequency of the given nucleus.
    """
    # Retrieve the magnetic field
    if B is None:

        # Check that the magnetic field has been set in parameters
        if parameters.magnetic_field is None:
            raise ValueError("'magnetic_field' has not been set in parameters.")
        
        # Set the magnetic field
        B = parameters.magnetic_field

    # Calculate the resonance frequency
    omega = - gamma(isotope, unit) * B * (1 + delta*1e-6)

    return omega

def fourier_transform(signal: np.ndarray,
                      dt: float,
                      normalize: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes the Fourier transform of a given time-domain signal and returns 
    the corresponding frequency-domain representation. The Fourier transform 
    can be normalized to ensure consistent peak intensities regardless of the
    time step.

    Parameters
    ----------
    signal : ndarray
        Input signal in the time domain.
    dt : float
        Time step between consecutive samples in the signal.
    normalize : bool, default=True
        Whether to normalize the Fourier transform.

    Returns
    -------
    freqs : ndarray
        Frequencies corresponding to the Fourier-transformed signal.
    fft_signal : ndarray
        Fourier-transformed signal in the frequency domain (normalized if
        specified).
    """
    # Compute the frequencies
    freqs = np.fft.fftfreq(len(signal), dt)

    # Compute the Fourier transform
    fft_signal = np.fft.fft(signal)

    # Normalize the Fourier transform if specified
    if normalize:
        fft_signal = fft_signal * dt

    # Apply frequency shifting
    freqs = np.fft.fftshift(freqs)
    fft_signal = np.fft.fftshift(fft_signal)

    return freqs, fft_signal

def spectrum(
    signal: np.ndarray,
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
    freqs, fft_signal = fourier_transform(signal, dt, normalize=normalize)

    # Get the specified part of the Fourier transform
    if part == "real":
        spectrum = np.real(fft_signal)
    elif part == "imag":
        spectrum = np.imag(fft_signal)
    else:
        raise ValueError("Invalid value for 'part'. Must be 'real' or 'imag'.")

    return freqs, spectrum

def frequency_to_chemical_shift(
        frequency: float | np.ndarray, 
        reference_frequency: float,
        spectrometer_frequency: float) -> float | np.ndarray:
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
    return (frequency - reference_frequency) / spectrometer_frequency * 1e6

def spectral_width_to_dwell_time(
    spectral_width: float,
    isotope: str,
    B: float=None
) -> float:
    """
    Calculates the dwell time (in seconds) from the spectral width given in ppm.

    Parameters
    ----------
    spectral_width : float
        Spectral width in ppm.
    isotope : str
        Nucleus symbol (e.g. `'1H'`) used to select the gyromagnetic ratio 
        required for the conversion.
    B : float, default=None
        Magnetic field of the spectrometer in T. If not supplied, the magnetic
        field is obtained from parameters.magnetic_field.

    Returns
    -------
    dwell_time : float
        Dwell time in seconds.
    """
    # Retrieve the magnetic field
    if B is None:

        # Check that the magnetic field has been set in parameters
        if parameters.magnetic_field is None:
            raise ValueError("'magnetic_field' has not been set in parameters.")
        
        # Set the magnetic field
        B = parameters.magnetic_field

    # Calculate the spectral width in Hz
    spectral_width = spectral_width * 1e-6 * gamma(isotope, "Hz") * B

    # Obtain the dwell time
    dwell_time = 1/spectral_width

    return dwell_time

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