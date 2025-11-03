"""
Spinguin package is desined to be imported as `import spinguin as sg`, which
reveals the user-friendly functionality to the user. This is documented under
the Spinguin (Basic). For more in-depth documentation of the package, including
the documentation of the re-usable, core functionality, see Spinguin (Advanced).
"""

# Make functionality from the API accessible directly under the spinguin
# namespace
from spinguin._core._chem import (
    associate,
    dissociate,
    permute_spins
)
from spinguin._core import (
    # config
    config,

    # hamiltonian
    hamiltonian,

    # nmr_isotopes
    gamma,
    quadrupole_moment,
    resonance_frequency,
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
    pulse,

    # relaxation
    relaxation,

    # spin_system
    SpinSystem,

    # states
    alpha_state,
    beta_state,
    equilibrium_state,
    measure,
    singlet_state,
    state,
    state_to_truncated_basis,
    state_to_zeeman,
    triplet_minus_state,
    triplet_plus_state,
    triplet_zero_state,
    unit_state,

    # superoperator
    superoperator
)
from spinguin import la
from spinguin import processing
from spinguin import sequences

__all__ = [
    # _chem
    "associate",
    "dissociate",
    "permute_spins",
    
    # _core: config
    config,

    # _core: hamiltonian
    "hamiltonian",

    # _core: nmr_isotopes
    "gamma",
    "quadrupole_moment",
    "resonance_frequency",
    "spin",

    # _core: operators
    "op_E",
    "op_Sm",
    "op_Sp",
    "op_Sx",
    "op_Sy",
    "op_Sz",
    "op_T",
    "op_T_coupled",
    "operator",

    # _core: parameters
    "parameters",

    # _core: propagation
    "propagator",
    "pulse",

    # _core: relaxation
    "relaxation",

    # _core: spin_system
    "SpinSystem",

    # _core: states
    "alpha_state",
    "beta_state",
    "equilibrium_state",
    "measure",
    "singlet_state",
    "state",
    "state_to_truncated_basis",
    "state_to_zeeman",
    "triplet_minus_state",
    "triplet_plus_state",
    "triplet_zero_state",
    "unit_state",

    # _core: superoperator
    "superoperator",

    # la
    "la",

    # processing
    "processing",

    # sequences
    "sequences",
]