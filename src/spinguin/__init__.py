"""
Spinguin package is desined to be imported as `import spinguin as sg`, which
reveals the user-friendly functionality to the user. This is documented under
the Spinguin (Basic). For more in-depth documentation of the package, including
the documentation of the re-usable, core functionality, see Spinguin (Advanced).
"""

# Make functionality from the API accessible directly under the spinguin
# namespace
from spinguin._api import *
from spinguin._core._chem import (
    associate,
    dissociate,
    permute_spins
)
from spinguin._core import (
    config,
    hamiltonian,
    op_E,
    op_Sm,
    op_Sp,
    op_Sx,
    op_Sy,
    op_Sz,
    op_T,
    op_T_coupled,
    operator,
    parameters,
    relaxation,
    SpinSystem,
    superoperator
)
from spinguin import la
from spinguin import sequences

__all__ = [
    # _chem
    "associate",
    "dissociate",
    "permute_spins",
    
    # _core
    "config",
    "hamiltonian",
    "op_E",
    "op_Sm",
    "op_Sp",
    "op_Sx",
    "op_Sy",
    "op_Sz",
    "op_T",
    "op_T_coupled",
    "operator",
    "parameters",
    "relaxation",
    "SpinSystem",
    "superoperator",

    # la
    "la",

    # sequences
    "sequences",
]