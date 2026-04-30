"""
Cache-management utilities for selected core Spinguin routines.

This module collects the cache-clearing helpers of several internal
submodules and exposes a single convenience function for resetting them.
"""

# Import cache-clearing helpers from the core submodules.
from spinguin._core._chem import (
    clear_cache_associate_index_map,
    clear_cache_dissociate_index_map,
    clear_cache_permutation_matrix,
)
from spinguin._core._la import clear_cache_CG_coeff
from spinguin._core._operators import clear_cache_op_T
from spinguin._core._superoperators import (
    clear_cache_sop_prod,
    clear_cache_sop_T_coupled,
    clear_cache_structure_coefficients,
)


def clear_cache() -> None:
    """
    Clear the internal caches used by selected core routines.

    This function resets cached results from the chemistry, linear algebra,
    operator, and superoperator helper functions that expose explicit
    cache-clearing entry points.

    Returns
    -------
    None
    """

    # Clear cache entries related to chemical index mappings.
    clear_cache_associate_index_map()
    clear_cache_dissociate_index_map()
    clear_cache_permutation_matrix()

    # Clear cached Clebsch-Gordan coefficients.
    clear_cache_CG_coeff()

    # Clear cached single-spin tensor operators.
    clear_cache_op_T()

    # Clear cached superoperator products and structure coefficients.
    clear_cache_sop_prod()
    clear_cache_sop_T_coupled()
    clear_cache_structure_coefficients()
