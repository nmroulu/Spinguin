"""
This module provides user friendly wrapper functions of the Spinguin's core 
functionality by making use of the `SpinSystem` class.
"""

# Expose only the necessary functionality from the API
from spinguin._api.core import (
    frequency_to_chemical_shift,
    gamma,
    liouvillian,
    propagator,
    propagator_to_rotframe,
    pulse,
    quadrupole_moment,
    resonance_frequency,
    spin,
    spectral_width_to_dwell_time,
    spectrum,
    time_axis,
)

__all__ = [
    "frequency_to_chemical_shift",
    "gamma",
    "liouvillian",
    "propagator",
    "propagator_to_rotframe",
    "pulse",
    "quadrupole_moment",
    "resonance_frequency",
    "spectral_width_to_dwell_time",
    "spectrum",
    "spin",
    "time_axis",
]