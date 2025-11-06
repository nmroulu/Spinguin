"""
This module provides the core functionality of Spinguin. The module is not meant
to be imported. The preferred way to use Spinguin is to use the functionality
directly under `spinguin` namespace, using::

    import spinguin as sg

If you still wish to import the _core module, continue with precaution!
"""
from ._core import *
from ._parameters import parameters
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