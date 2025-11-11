"""
Spinguin package is desined to be imported as `import spinguin as sg`, which
reveals the main functionality of the package to the user in a user-friendly
manner. For example, to create a SpinSystem object with one 1H nucleus, the user
should::

    import spinguin as sg
    spin_system = sg.SpinSystem(["1H"])
"""

# Make the core functionality directly available under the spinguin namespace
from spinguin._core import (
    # cache
    clear_cache,

    # chem
    associate,
    dissociate,
    permute_spins,

    # hamiltonian
    hamiltonian,

    # liouvillian
    liouvillian,

    # nmr_isotopes
    gamma,
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

    # parameters
    parameters,

    # propagation
    propagator,
    propagator_to_rotframe,
    pulse,

    # relaxation
    relaxation,

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
from spinguin import sequences

__all__ = [
    # core: cache
    "clear_cache",

    # core: chem
    "associate",
    "dissociate",
    "permute_spins",

    # core: hamiltonian
    "hamiltonian",

    # core: liouvillian
    "liouvillian",

    # core: nmr_isotopes
    "gamma",
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