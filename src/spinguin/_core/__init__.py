"""
This module provides the core functionality of Spinguin. The module is not meant
to be imported. The preferred way to use Spinguin is to use the functionality
directly under `spinguin` namespace, using::

    import spinguin as sg

If you still wish to import the _core module, continue with precaution!
"""
from ._cache import clear_cache
from ._chem import (
    associate,
    dissociate,
    permute_spins
)
from ._hamiltonian import hamiltonian
from ._liouvillian import liouvillian
from ._nmr_isotopes import (
    gamma,
    quadrupole_moment,
    spin
)
from ._operators import (
    op_E,
    op_Sm,
    op_Sp,
    op_Sx,
    op_Sy,
    op_Sz,
    op_T,
    op_T_coupled,
    operator
)
from ._parameters import parameters
from ._propagation import (
    propagator,
    propagator_to_rotframe,
    pulse
)
from ._relaxation import relaxation
from ._specutils import (
    fourier_transform,
    frequency_to_chemical_shift,
    resonance_frequency,
    spectral_width_to_dwell_time,
    spectrum,
    time_axis
)
from ._spin_system import SpinSystem
from ._states import (
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
)
from ._superoperators import (
    sop_T_coupled,
    superoperator
)
from ._utils import (
    coherence_order,
    idx_to_lq,
    lq_to_idx
)

__all__ = [
    #cache
    "clear_cache",

    #chem
    "associate",
    "dissociate",
    "permute_spins",

    #hamiltonian
    "hamiltonian",

    #liouvillian
    "liouvillian",

    #nmr_isotopes
    "gamma",
    "quadrupole_moment",
    "spin",

    #operators
    "op_E",
    "op_Sm",
    "op_Sp",
    "op_Sx",
    "op_Sy",
    "op_Sz",
    "op_T",
    "op_T_coupled",
    "operator",

    #parameters 
    "parameters",

    #propagation
    "propagator",
    "propagator_to_rotframe",
    "pulse",

    #relaxation
    "relaxation",

    #specutils
    "fourier_transform",
    "frequency_to_chemical_shift",
    "resonance_frequency",
    "spectral_width_to_dwell_time",
    "spectrum",
    "time_axis",

    #spin_system
    "SpinSystem",

    #states
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
    
    #superoperators
    "sop_T_coupled",
    "superoperator",

    #utils
    "coherence_order",
    "idx_to_lq",
    "lq_to_idx",
]