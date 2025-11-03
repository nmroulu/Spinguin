"""
This module provides core functions for spectral data analysis, including
Fourier transforms, spectrum generation, and unit conversions commonly used in
NMR and signal processing.
"""

# Imports
import numpy as np
from typing import Literal
from spinguin._core import resonance_frequency, parameters

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
    return (frequency - reference_frequency) / spectrometer_frequency * 1e6

def spectral_width_to_dwell_time(spectral_width: float, isotope: str) -> float:
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
    # Calculate the spectral width in Hz
    spectral_width = spectral_width * resonance_frequency(isotope, 0, "Hz")

    # Obtain the dwell time
    dwell_time = 1/spectral_width

    return dwell_time