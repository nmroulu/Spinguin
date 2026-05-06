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
    # cache
    clear_cache,

    # chem
    associate,
    dissociate,
    permute_spins,

    # hamiltonian
    hamiltonian,

    # interactions
    dd_coupling_tensors,
    dd_constants,
    Q_intr_tensors,
    shielding_intr_tensors,

    # liouvillian
    liouvillian,

    # nmr_isotopes
    add_isotope,
    atomic_mass,
    gamma,
    natural_abundance,
    quadrupole_moment,
    spin,

    # operators
    op_E,
    op_Sm,
    op_Sp,
    op_Sx,
    op_Sy,
    op_Sz,
    op_T,
    op_T_coupled,
    operator,

    # molecule
    Molecule,

    # parameters
    parameters,

    # propagation
    propagator,
    propagator_to_rotframe,
    pulse,

    # relaxation
    relaxation,

    # rotframe
    rotating_frame,

    # specutils
    fourier_transform,
    frequency_to_chemical_shift,
    resonance_frequency,
    spectral_width_to_dwell_time,
    spectrum,
    time_axis,

    # spin_system
    SpinSystem,

    # states
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

    # superoperators
    sop_T_coupled,
    superoperator,

    # utils
    coherence_order,
    idx_to_lq,
    lq_to_idx,
)

# Re-export the ready-to-use pulse-sequence namespace.
from . import sequences

# Define the public package interface for star imports and documentation.
__all__ = [
    # core: cache
    "clear_cache",

    # core: chem
    "associate",
    "dissociate",
    "permute_spins",

    # core: hamiltonian
    "hamiltonian",

    # core: interactions
    "dd_coupling_tensors",
    "dd_constants",
    "Q_intr_tensors",
    "shielding_intr_tensors",

    # core: liouvillian
    "liouvillian",

    # core: molecule
    "Molecule",

    # core: nmr_isotopes
    "add_isotope",
    "atomic_mass",
    "gamma",
    "natural_abundance",
    "quadrupole_moment",
    "spin",

    # core: operators
    "op_E",
    "op_Sm",
    "op_Sp",
    "op_Sx",
    "op_Sy",
    "op_Sz",
    "op_T",
    "op_T_coupled",
    "operator",

    # core: parameters 
    "parameters",

    # core: propagation
    "propagator",
    "propagator_to_rotframe",
    "pulse",

    # core: relaxation
    "relaxation",

    # core: rotframe
    "rotating_frame",

    # core: specutils
    "fourier_transform",
    "frequency_to_chemical_shift",
    "resonance_frequency",
    "spectral_width_to_dwell_time",
    "spectrum",
    "time_axis",

    # core: spin_system
    "SpinSystem",

    # core: states
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
    
    # core: superoperators
    "sop_T_coupled",
    "superoperator",

    # core: utils
    "coherence_order",
    "idx_to_lq",
    "lq_to_idx",

    # sequences
    "sequences",
]