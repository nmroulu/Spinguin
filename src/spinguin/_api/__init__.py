"""
This module provides user friendly wrapper functions of the Spinguin's core 
functionality by making use of the `SpinSystem` class.
"""

# Expose only the necessary functionality from the API
from spinguin._api.core import (
    alpha_state,
    beta_state,
    equilibrium_state,
    frequency_to_chemical_shift,
    gamma,
    inversion_recovery,
    liouvillian,
    measure,
    propagator,
    propagator_to_rotframe,
    pulse,
    pulse_and_acquire,
    quadrupole_moment,
    relaxation,
    resonance_frequency,
    rotating_frame,
    singlet_state,
    spin,
    spectral_width_to_dwell_time,
    spectrum,
    state,
    state_to_zeeman,
    time_axis,
    triplet_minus_state,
    triplet_plus_state,
    triplet_zero_state,
    unit_state
)

__all__ = [
    "alpha_state",
    "beta_state",
    "equilibrium_state",
    "frequency_to_chemical_shift",
    "gamma",
    "inversion_recovery",
    "liouvillian",
    "measure",
    "propagator",
    "propagator_to_rotframe",
    "pulse",
    "pulse_and_acquire",
    "quadrupole_moment",
    "relaxation",
    "resonance_frequency",
    "rotating_frame",
    "singlet_state",
    "spectral_width_to_dwell_time",
    "spectrum",
    "spin",
    "state",
    "state_to_zeeman",
    "time_axis",
    "triplet_minus_state",
    "triplet_plus_state",
    "triplet_zero_state",
    "unit_state"
]