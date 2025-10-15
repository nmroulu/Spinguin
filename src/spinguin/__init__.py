"""
Spinguin package is desined to be imported as `import spinguin as sg`, which
reveals the user-friendly functionality to the user. This is documented under
the Spinguin (Basic). For more in-depth documentation of the package, including
the documentation of the re-usable, core functionality, see Spinguin (Advanced).
"""

# Make functionality from the API accessible directly under the spinguin
# namespace
from spinguin._api import *
from spinguin._chem import (
    associate,
    dissociate,
    permute_spins
)
from spinguin._config import config
from spinguin._operators import (
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
from spinguin._parameters import parameters
from spinguin._spin_system import SpinSystem
from spinguin._superoperators import superoperator

__all__ = [
    # _chem
    "associate",
    "dissociate",
    "permute_spins",

    # _config
    "config",

    # _operators
    "op_E",
    "op_Sm",
    "op_Sp",
    "op_Sx",
    "op_Sy",
    "op_Sz",
    "op_T",
    "op_T_coupled",
    "operator",

    # _parameters
    "parameters",

    # _spin_system
    "SpinSystem",

    # _superoperators
    "superoperator"
]