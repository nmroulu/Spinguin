"""
This module provides functionality to clear the caches of the functions.
"""
# Imports
from spinguin._core._chem import (
    clear_cache_associate_index_map,
    clear_cache_dissociate_index_map,
    clear_cache_permutation_matrix
)
from spinguin._core._la import clear_cache_CG_coeff
from spinguin._core._operators import clear_cache_op_T
from spinguin._core._states import (
    clear_cache_state_from_op_def,
    clear_cache_state_from_string
)
from spinguin._core._superoperators import (
    clear_cache_sop_prod,
    clear_cache_sop_T_coupled,
    clear_cache_structure_coefficients
)

def clear_cache():
    """
    Clears the cache.
    """
    # Clear caches from various modules
    clear_cache_associate_index_map()
    clear_cache_dissociate_index_map()
    clear_cache_permutation_matrix()
    clear_cache_CG_coeff()
    clear_cache_op_T()
    clear_cache_state_from_op_def()
    clear_cache_state_from_string()
    clear_cache_sop_prod()
    clear_cache_sop_T_coupled()
    clear_cache_structure_coefficients()
