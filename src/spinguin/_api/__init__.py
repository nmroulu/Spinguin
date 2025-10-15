"""
This module provides user friendly wrapper functions of the Spinguin's core 
functionality by making use of the `SpinSystem` class.
"""

# Expose only the necessary functionality from the API
from spinguin._api.core import (
    alpha_state,
    beta_state,
    associate,
    dissociate,
    equilibrium_state,
    frequency_to_chemical_shift,
    gamma,
    hamiltonian,
    inversion_recovery,
    liouvillian,
    measure,
    permute_spins,
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
    superoperator,
    time_axis,
    triplet_minus_state,
    triplet_plus_state,
    triplet_zero_state,
    unit_state
)
from spinguin._api._config import config
from spinguin._api._parameters import parameters
from spinguin._api._spin_system import SpinSystem

__all__ = [
    # spinguin._api._config
    "config",

    # spinguin._api._parameters
    "parameters",

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
    "permute_spins",
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
    "SpinSystem",
    "state",
    "state_to_zeeman",
    "superoperator",
    "time_axis",
    "triplet_minus_state",
    "triplet_plus_state",
    "triplet_zero_state",
    "unit_state"
]