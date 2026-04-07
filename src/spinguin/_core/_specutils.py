"""
Spectral utilities for common NMR post-processing tasks.

This module provides helper functions for resonance-frequency evaluation,
Fourier transformation, chemical-shift conversion, and construction of
uniformly sampled time axes.
"""

from typing import Literal

import numpy as np

from spinguin._core._nmr_isotopes import gamma
from spinguin._core._parameters import parameters

def _resolve_magnetic_field(B: float | None) -> float:
    """
    Return the magnetic field to be used in a spectral calculation.

    Usage: ``_resolve_magnetic_field(B)``.

    Parameters
    ----------
    B : float or None
        Magnetic field in T. If ``None``, the value is taken from
        ``parameters.magnetic_field``.

    Returns
    -------
    float
        Magnetic field in T.

    Raises
    ------
    ValueError
        Raised if ``B`` is ``None`` and the global magnetic field has not been
        configured.
    """

    # Use the explicitly supplied magnetic field whenever available.
    if B is not None:
        return B

    # Ensure that a default magnetic field has been configured.
    if parameters.magnetic_field is None:
        raise ValueError("'magnetic_field' has not been set in parameters.")

    # Return the shared magnetic-field setting.
    return parameters.magnetic_field


def resonance_frequency(
    isotope: str,
    B: float | None=None,
    delta: float=0,
    unit: Literal["Hz", "rad/s"]="Hz",
) -> float:
    """
    Compute the resonance frequency at a given magnetic field and shift.

    Usage: ``resonance_frequency(isotope, B=None, delta=0, unit='Hz')``.

    Parameters
    ----------
    isotope : str
        Nucleus symbol, for example ``'1H'``, used to select the gyromagnetic
        ratio.
    B : float, default=None
        Magnetic field strength in T. If not supplied, the value stored in
        ``parameters.magnetic_field`` is used.
    delta : float, default=0
        Chemical shift in ppm.
    unit : {'Hz', 'rad/s'}
        Unit in which the resonance frequency is returned.

    Returns
    -------
    omega : float
        Resonance frequency of the selected nucleus.

    Raises
    ------
    ValueError
        Raised if no magnetic field is supplied and no default value has been
        configured.
    """

    # Resolve the magnetic field used for the frequency calculation.
    magnetic_field = _resolve_magnetic_field(B)

    # Evaluate the resonance frequency including the chemical-shift offset.
    omega = (
        -gamma(isotope, unit)
        * magnetic_field
        * (1 + delta * 1e-6)
    )

    return omega


def fourier_transform(
    signal: np.ndarray,
    dt: float,
    normalize: bool=True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the discrete Fourier transform of a time-domain signal.

    Usage: ``fourier_transform(signal, dt, normalize=True)``.

    The returned frequency axis and transformed signal are both shifted so that
    zero frequency is centred in the output arrays.

    Parameters
    ----------
    signal : ndarray
        Input signal in the time domain.
    dt : float
        Time step between consecutive samples in the signal.
    normalize : bool, default=True
        Whether to normalise the Fourier transform.

    Returns
    -------
    freqs : ndarray
        Frequency axis corresponding to the Fourier-transformed signal.
    fft_signal : ndarray
        Fourier-transformed signal in the frequency domain. If ``normalize``
        is ``True``, the result is scaled by the sampling interval.
    """

    # Construct the unshifted frequency axis.
    freqs = np.fft.fftfreq(len(signal), dt)

    # Evaluate the discrete Fourier transform of the signal.
    fft_signal = np.fft.fft(signal)

    # Apply the sampling-interval normalisation when requested.
    if normalize:
        fft_signal = fft_signal * dt

    # Shift zero frequency to the centre of the arrays.
    freqs = np.fft.fftshift(freqs)
    fft_signal = np.fft.fftshift(fft_signal)

    return freqs, fft_signal


def spectrum(
    signal: np.ndarray,
    dt: float,
    normalize: bool=True,
    part: Literal["real", "imag"]="real",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return one component of the Fourier-domain spectrum.

    Usage: ``spectrum(signal, dt, normalize=True, part='real')``.

    Parameters
    ----------
    signal : ndarray
        Input signal in the time domain.
    dt : float
        Time step between consecutive samples in the signal.
    normalize : bool, default=True
        Whether to normalise the Fourier transform.
    part : {'real', 'imag'}
        Selects which component of the Fourier transform to return.

    Returns
    -------
    freqs : ndarray
        Frequency axis corresponding to the Fourier-transformed signal.
    spectrum_part : ndarray
        Selected real or imaginary component of the Fourier-domain signal.

    Raises
    ------
    ValueError
        Raised if ``part`` is not ``'real'`` or ``'imag'``.
    """

    # Compute the shifted Fourier transform of the input signal.
    freqs, fft_signal = fourier_transform(signal, dt, normalize=normalize)

    # Extract the requested component of the transformed signal.
    if part == "real":
        spectrum_part = np.real(fft_signal)
    elif part == "imag":
        spectrum_part = np.imag(fft_signal)
    else:
        raise ValueError("Invalid value for 'part'. Must be 'real' or 'imag'.")

    return freqs, spectrum_part


def frequency_to_chemical_shift(
    frequency: float | np.ndarray,
    reference_frequency: float,
    spectrometer_frequency: float,
) -> float | np.ndarray:
    """
    Convert frequencies in Hz to chemical shifts in ppm.

    Usage: ``frequency_to_chemical_shift(frequency, reference_frequency,
    spectrometer_frequency)``.

    Parameters
    ----------
    frequency : float or ndarray
        Frequency, or an array of frequencies, in Hz.
    reference_frequency : float
        Reference frequency in Hz.
    spectrometer_frequency : float
        Spectrometer frequency in Hz.

    Returns
    -------
    chemical_shift : float or ndarray
        Chemical shift, or an array of chemical shifts, in ppm.
    """

    # Convert the frequency offset to the chemical-shift scale.
    return (
        (frequency - reference_frequency)
        / spectrometer_frequency
        * 1e6
    )


def spectral_width_to_dwell_time(
    spectral_width: float,
    isotope: str,
    B: float | None=None,
) -> float:
    """
    Convert a spectral width in ppm to a dwell time in seconds.

    Usage: ``spectral_width_to_dwell_time(spectral_width, isotope, B=None)``.

    Parameters
    ----------
    spectral_width : float
        Spectral width in ppm.
    isotope : str
        Nucleus symbol, for example ``'1H'``, used to select the gyromagnetic
        ratio required for the conversion.
    B : float, default=None
        Magnetic field of the spectrometer in T. If not supplied, the magnetic
        field is obtained from ``parameters.magnetic_field``.

    Returns
    -------
    dwell_time : float
        Dwell time in seconds.

    Raises
    ------
    ValueError
        Raised if no magnetic field is supplied and no default value has been
        configured.
    """

    # Resolve the magnetic field used for the conversion.
    magnetic_field = _resolve_magnetic_field(B)

    # Convert the spectral width from ppm to Hz.
    spectral_width_hz = (
        spectral_width
        * 1e-6
        * gamma(isotope, "Hz")
        * magnetic_field
    )

    # Invert the spectral width to obtain the dwell time.
    dwell_time = 1 / spectral_width_hz

    return dwell_time


def time_axis(
    npoints: int,
    time_step: float,
) -> np.ndarray:
    """
    Generate a uniformly spaced time axis.

    Usage: ``time_axis(npoints, time_step)``.

    Parameters
    ----------
    npoints : int
        Number of points.
    time_step : float
        Time step in seconds.

    Returns
    -------
    ndarray
        One-dimensional time axis with ``npoints`` samples.
    """

    # Construct the uniformly sampled time axis.
    t_axis = np.linspace(0, npoints * time_step, npoints, endpoint=False)

    return t_axis