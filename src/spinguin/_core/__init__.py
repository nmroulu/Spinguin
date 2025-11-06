"""
This module provides the core functionality of Spinguin, which is accessed
directly under the Spinguin namespace. For example, to create an instance of
SpinSystem, the user should::

    import spinguin as sg
    spin_system = sg.SpinSystem
"""

# Imports
from ._basis_indexing import (
    coherence_order,
    idx_to_lq,
    lq_to_idx,
    spin_order
)

from ._chem import (
    associate,
    dissociate,
    permutation_matrix,
    permute_spins
)

from ._config import config

from ._hamiltonian import hamiltonian

from ._hide_prints import HidePrints

from ._la import (
    angle_between_vectors,
    cartesian_tensor_to_spherical_tensor,
    CG_coeff,
    comm,
    custom_dot,
    decompose_matrix,
    eliminate_small,
    expm,
    expm_vec,
    find_common_rows,
    isvector,
    norm_1,
    principal_axis_system,
    vector_to_spherical_tensor
)

from ._liouvillian import liouvillian

from ._nmr_isotopes import (
    dd_constant,
    gamma,
    Q_constant,
    quadrupole_moment,
    resonance_frequency,
    spin
)

from ._operators import (
    op_E,
    op_from_op_def,
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
    pulse
)

from ._relaxation import (
    G0,
    relaxation,
    tau_c_l
)

from ._rotframe import rotating_frame

from ._spin_system import SpinSystem

from ._states import (
    alpha_state,
    beta_state,
    equilibrium_state,
    measure,
    singlet_state,
    state,
    state_from_op_def,
    state_vector_to_density_matrix,
    triplet_minus_state,
    triplet_plus_state,
    triplet_zero_state,
    unit_state
)

from ._superoperators import (
    superoperator,
    superoperator_from_op_def,
    superoperator_T_coupled
)

from ._type_conversions import (
    arraylike_to_array,
    arraylike_to_tuple,
    bytes_to_sparse,
    sparse_to_bytes
)

__all__ = [
    # basis_indexing
    "coherence_order",
    "idx_to_lq",
    "lq_to_idx",
    "spin_order",

    # chem
    "associate",
    "dissociate",
    "permutation_matrix",
    "permute_spins",
    
    # config
    "config",

    # hamiltonian
    "hamiltonian",

    # hide_prints
    "HidePrints",

    # la
    "angle_between_vectors",
    "cartesian_tensor_to_spherical_tensor",
    "CG_coeff",
    "comm",
    "custom_dot",
    "decompose_matrix",
    "eliminate_small",
    "expm",
    "expm_vec",
    "find_common_rows",
    "isvector",
    "norm_1",
    "principal_axis_system",
    "vector_to_spherical_tensor",

    # liouvillian
    "liouvillian",

    # nmr_isotopes
    "dd_constant",
    "gamma",
    "Q_constant",
    "quadrupole_moment",
    "resonance_frequency",
    "spin",

    # operators
    "op_E",
    "op_from_op_def",
    "op_Sm",
    "op_Sp",
    "op_Sx",
    "op_Sy",
    "op_Sz",
    "op_T",
    "op_T_coupled",
    "operator",

    # parameters
    "parameters",

    # propagation
    "propagator",
    "pulse",

    # relaxation
    "G0",
    "relaxation",
    "tau_c_l",

    # rotframe
    "rotating_frame",

    # spin_system
    "SpinSystem",

    # states
    "alpha_state",
    "beta_state",
    "equilibrium_state",
    "measure",
    "singlet_state",
    "state",
    "state_from_op_def",
    "state_vector_to_density_matrix",
    "triplet_minus_state",
    "triplet_plus_state",
    "triplet_zero_state",
    "unit_state",

    # superoperators
    "superoperator",
    "superoperator_from_op_def",
    "superoperator_T_coupled",

    # type_conversions
    "arraylike_to_array",
    "arraylike_to_tuple",
    "bytes_to_sparse",
    "sparse_to_bytes",
]