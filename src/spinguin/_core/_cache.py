"""
cache.py

Provides utility functions for clearing the internal caches used by selected
core Spinguin routines.
"""

# Import cache-clearing helpers from the chemistry module.
from spinguin._core._chem import (
    clear_cache_associate_index_map,
    clear_cache_dissociate_index_map,
    clear_cache_permutation_matrix,
)

# Import the cache-clearing helper for Clebsch-Gordan coefficients.
from spinguin._core._la import clear_cache_CG_coeff

# Import the cache-clearing helper for tensor operators.
from spinguin._core._operators import clear_cache_op_T

# Import cache-clearing helpers from the superoperator module.
from spinguin._core._superoperators import (
    clear_cache_sop_prod,
    clear_cache_sop_T_coupled,
    clear_cache_structure_coefficients,
)


def clear_cache() -> None:
    """
    Clear the internal caches used by selected core routines.

    This function resets cached results from chemistry, linear algebra,
    operator, and superoperator helper functions.

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
