"""
Internal core namespace for Spinguin.

This module re-exports the internal core functionality of Spinguin. In normal
use, the public package namespace should be preferred::

	import spinguin as sg

Direct imports from :mod:`spinguin._core` are possible but intended primarily
for internal use and advanced development workflows.
"""
from ._cache import clear_cache
from ._chem import (
    associate,
    dissociate,
    permute_spins
)
from ._hamiltonian import hamiltonian
from ._liouvillian import liouvillian
from ._molecule import Molecule
from ._nmr_isotopes import (
    atomic_mass,
    gamma,
    natural_abundance,
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
from ._rotframe import rotating_frame
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
    # Cache
    "clear_cache",

    # Chemistry
    "associate",
    "dissociate",
    "permute_spins",

    # Hamiltonian
    "hamiltonian",

    # Liouvillian
    "liouvillian",

    # Molecule
    "Molecule",

    # NMR isotopes
    "atomic_mass",
    "gamma",
    "natural_abundance",
    "quadrupole_moment",
    "spin",

    # Operators
    "op_E",
    "op_Sm",
    "op_Sp",
    "op_Sx",
    "op_Sy",
    "op_Sz",
    "op_T",
    "op_T_coupled",
    "operator",

    # Parameters
    "parameters",

    # Propagation
    "propagator",
    "propagator_to_rotframe",
    "pulse",

    # Relaxation
    "relaxation",

    # Rotating frame
    "rotating_frame",

    # Spectral utilities
    "fourier_transform",
    "frequency_to_chemical_shift",
    "resonance_frequency",
    "spectral_width_to_dwell_time",
    "spectrum",
    "time_axis",

    # Spin system
    "SpinSystem",

    # States
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

    # Superoperators
    "sop_T_coupled",
    "superoperator",

    # Utilities
    "coherence_order",
    "idx_to_lq",
    "lq_to_idx",
]