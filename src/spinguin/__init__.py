"""
Public package namespace for Spinguin.

Spinguin is intended to be imported as ``import spinguin as sg`` so that the
main simulation functionality is available directly under the package
namespace. For example, a one-spin system may be created as follows::

    import spinguin as sg
    spin_system = sg.SpinSystem(["1H"])

The package also re-exports the ready-to-use pulse-sequence namespace as
``sg.sequences``.
"""

# Re-export the public core functionality under the package namespace.
from spinguin._core import (
    # Cache
    clear_cache,

    # Chemistry
    associate,
    dissociate,
    permute_spins,

    # Hamiltonian
    hamiltonian,

    # Liouvillian
    liouvillian,

    # NMR isotopes
    atomic_mass,
    gamma,
    natural_abundance,
    quadrupole_moment,
    spin,

    # Operators
    op_E,
    op_Sm,
    op_Sp,
    op_Sx,
    op_Sy,
    op_Sz,
    op_T,
    op_T_coupled,
    operator,

    # Molecule
    Molecule,

    # Parameters
    parameters,

    # Propagation
    propagator,
    propagator_to_rotframe,
    pulse,

    # Relaxation
    relaxation,

    # Rotating frame
    rotating_frame,

    # Spectral utilities
    fourier_transform,
    frequency_to_chemical_shift,
    resonance_frequency,
    spectral_width_to_dwell_time,
    spectrum,
    time_axis,

    # Spin system
    SpinSystem,

    # States
    alpha_state,
    beta_state,
    equilibrium_state,
    measure,
    singlet_state,
    state,
    state_to_zeeman,
    triplet_minus_state,
    triplet_plus_state,
    triplet_zero_state,
    unit_state,

    # Superoperators
    sop_T_coupled,
    superoperator,

    # Utilities
    coherence_order,
    idx_to_lq,
    lq_to_idx,
)

# Re-export the ready-to-use pulse-sequence namespace.
from . import sequences

# Define the public package interface for star imports and documentation.
__all__ = [
    # Core: cache
    "clear_cache",

    # Core: chemistry
    "associate",
    "dissociate",
    "permute_spins",

    # Core: hamiltonian
    "hamiltonian",

    # Core: liouvillian
    "liouvillian",

    # Core: molecule
    "Molecule",

    # Core: NMR isotopes
    "atomic_mass",
    "gamma",
    "natural_abundance",
    "quadrupole_moment",
    "spin",

    # Core: operators
    "op_E",
    "op_Sm",
    "op_Sp",
    "op_Sx",
    "op_Sy",
    "op_Sz",
    "op_T",
    "op_T_coupled",
    "operator",

    # Core: parameters
    "parameters",

    # Core: propagation
    "propagator",
    "propagator_to_rotframe",
    "pulse",

    # Core: relaxation
    "relaxation",

    # Core: rotating frame
    "rotating_frame",

    # Core: spectral utilities
    "fourier_transform",
    "frequency_to_chemical_shift",
    "resonance_frequency",
    "spectral_width_to_dwell_time",
    "spectrum",
    "time_axis",

    # Core: spin system
    "SpinSystem",

    # Core: states
    "alpha_state",
    "beta_state",
    "equilibrium_state",
    "measure",
    "singlet_state",
    "state",
    "state_to_zeeman",
    "triplet_minus_state",
    "triplet_plus_state",
    "triplet_zero_state",
    "unit_state",

    # Core: superoperators
    "sop_T_coupled",
    "superoperator",

    # Core: utilities
    "coherence_order",
    "idx_to_lq",
    "lq_to_idx",

    # sequences
    "sequences",
]