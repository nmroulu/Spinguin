"""
This module provides the core functionality of Spinguin which is not dependent
on any inbuilt class. Therefore, these functions can be reused in any context
elsewhere. These can be accessed using::

    # Replace module and function with something
    from spinguin.core.module.function

The preferred way to use Spinguin is to use the functionality under `spinguin`
namespace, described in Spinguin (Basic), using::

    import spinguin as sg
"""

# Imports
from spinguin._core._config import config
from spinguin._core._hamiltonian import hamiltonian
from spinguin._core._nmr_isotopes import (
    gamma,
    quadrupole_moment,
    resonance_frequency,
    spin
)
from spinguin._core._operators import (
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
from spinguin._core._parameters import parameters
from spinguin._core._propagation import propagator, pulse
from spinguin._core._relaxation._relaxation import relaxation
from spinguin._core._spin_system import SpinSystem
from spinguin._core._states import (
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
    unit_state
)
from spinguin._core._superoperators import superoperator

__all__ = [
    # spinguin._core._config
    "config",

    # spinguin._core._hamiltonian
    "hamiltonian",

    # spinguin._core._nmr_isotopes
    "gamma",
    "quadrupole_moment",
    "resonance_frequency",
    "spin",

    # spinguin._core._operators
    "op_E",
    "op_Sm",
    "op_Sp",
    "op_Sx",
    "op_Sy",
    "op_Sz",
    "op_T",
    "op_T_coupled",
    "operator",

    # spinguin._core._parameters
    "parameters",

    # spinguin._core._propagation
    "propagator",
    "pulse",

    # spinguin._core._relaxation
    "relaxation",

    # spinguin._core._spin_system
    "SpinSystem",

    # spinguin._core._states
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

    # spinguin._core._superoperators
    "superoperator",
]