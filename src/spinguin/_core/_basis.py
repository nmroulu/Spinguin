"""
Basis-set machinery for `SpinSystem`.

The `Basis` class constructs Liouville-space basis states and provides
multiple truncation strategies for reducing the basis-set dimension
before simulation.

The basis functionality is intended to be accessed through the owning
`SpinSystem` instance. Example::

    import spinguin as sg

    spin_system = sg.SpinSystem(["1H"])
    spin_system.basis.max_spin_order = 1
    spin_system.basis.build()
"""

# Imports
from __future__ import annotations

import time
import warnings
from itertools import combinations, product
from typing import TYPE_CHECKING, Iterator

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from spinguin._core._hide_prints import HidePrints
from spinguin._core._la import (
    arraylike_to_tuple,
    eliminate_small,
    expm_vec,
    isvector,
    norm_1,
)
from spinguin._core._parameters import parameters
from spinguin._core._status import status
from spinguin._core._utils import coherence_order
from spinguin._core._relaxation import dd_constant

if TYPE_CHECKING:
    from spinguin._core._spin_system import SpinSystem


class Basis:
    """
    Manages the basis set associated with a spin system.

    The basis stores the product-operator definitions used in Liouville space
    and provides methods for constructing and truncating that basis.

    The class is designed to be instantiated and accessed through
    `SpinSystem`. Example::

        import spinguin as sg

        spin_system = sg.SpinSystem(["1H"])
        spin_system.basis.max_spin_order = 1
        spin_system.basis.build()
    """

    _basis: np.ndarray | None = None
    _basis_dict: dict[tuple[int, ...], int] | None = None
    _max_spin_order: int | None = None
    _spin_system: SpinSystem | None = None

    def __init__(self, spin_system: SpinSystem):
        """
        Initialise the basis set for a spin system.

        Parameters
        ----------
        spin_system : SpinSystem
            Spin system that owns the basis object.
        """

        # Store the parent spin system for later basis operations
        self._spin_system = spin_system

    @property
    def dim(self) -> int:
        """
        Return the current basis dimension.

        Returns
        -------
        int
            Number of basis states currently stored.
        """
        return self.basis.shape[0]

    @property
    def max_spin_order(self) -> int | None:
        """
        Maximum number of active spins (spin order)
        allowed in basis operators.

        Returns
        -------
        int or None
            Largest spin order included when the basis is built. A value of
            `None` indicates that the limit has not yet been set.
        """
        return self._max_spin_order

    @property
    def basis(self) -> np.ndarray | None:
        """
        Return the current basis-state definitions.

        The basis is stored as an array of shape `(N, M)`, where `N` is the
        number of basis states and `M` is the number of spins. Each row encodes
        a Kronecker product of irreducible spherical tensor operators. The
        integer labelling follows increasing rank `l` and decreasing projection
        `q`:

        - 0 --> T(0, 0)
        - 1 --> T(1, 1)
        - 2 --> T(1, 0)
        - 3 --> T(1, -1), and so on.

        Returns
        -------
        ndarray or None
            Array containing the basis-state definitions, or `None` if the
            basis has not yet been built.
        """
        return self._basis

    @basis.setter
    def basis(self, basis: np.ndarray):
        """
        Store a validated basis array and build the corresponding index map.

        Parameters
        ----------
        basis : ndarray
            Basis array of shape `(N, M)`, where `M` matches the number of
            spins in the parent spin system.
        """

        # Validate the type and shape of the supplied basis array
        if not isinstance(basis, np.ndarray) or basis.ndim != 2:
            raise ValueError("Basis set must be a two-dimensional NumPy array "
                             "of shape (N, M), where N is the number of "
                             "basis states and M is the number of spins.")

        if basis.shape[1] != self._spin_system.nspins:
            raise ValueError("Mismatch between basis set and number of spins.")

        # Freeze the basis array to avoid accidental external modification
        basis.flags.writeable = False

        # Store the basis array and build a state-to-index lookup table
        self._basis = basis
        self._basis_dict = {tuple(state): i for i, state in enumerate(basis)}

        status(f"Basis set has been set with dimension: {self.dim}.\n")

    @max_spin_order.setter
    def max_spin_order(self, max_spin_order: int) -> None:
        """
        Set the maximum spin order used when building the basis.

        Parameters
        ----------
        max_spin_order : int
            Largest number of active spins allowed in any product operator.
        """

        # Validate the supplied maximum spin order against the system size
        if max_spin_order < 1:
            raise ValueError("Maximum spin order must be at least 1.")
        if max_spin_order > self._spin_system.nspins:
            raise ValueError("Maximum spin order cannot be larger than "
                             "the number of spins in the system.")
        self._max_spin_order = max_spin_order
        status(f"Maximum spin order set to {self.max_spin_order}.\n")

    def build(self) -> None:
        """
        Build the basis set for the parent spin system (see `SpinSystem`).

        If `max_spin_order` has not been specified explicitly, it is set to the
        total number of spins in the system.
        """

        # Default to the full spin order if no truncation limit was specified
        if self.max_spin_order is None:
            warnings.warn("Maximum spin order not specified. "
                          "Using the number of spins in the system.")
            self.max_spin_order = self._spin_system.nspins

        # Construct and store the basis for the current spin system
        self.basis = _make_basis(
            spins=self._spin_system.spins,
            max_spin_order=self.max_spin_order,
        )

    def indexof(self, op_def: np.ndarray | list | tuple) -> int:
        """
        Find the index of a basis state.

        Parameters
        ----------
        op_def : ndarray or list or tuple
            A one-dimensional array, list or tuple that defines a basis state
            using the integer indexing scheme. The indices are given by
            `N = l^2 + l - q`, where `l` is the operator rank and `q` is the
            projection. For example::

                op_def = [0, 2, 0]

        Returns
        -------
        idx : int
            Index of the requested state in the current basis.
        """

        # Ensure that a basis exists before attempting the lookup
        if self.basis is None:
            raise ValueError("Basis must be built before obtaining indices. "
                             "Call 'build()' on the basis object first.")

        # Convert the operator definition to the internal tuple format
        op_def = arraylike_to_tuple(op_def)

        # Look up the requested basis state in the cached dictionary
        if op_def in self._basis_dict:
            idx = self._basis_dict[op_def]
        else:
            raise ValueError(f"Could not find {op_def} in the basis set.")

        return idx

    def truncate_by_coherence(
        self,
        coherence_orders: list[int],
        *objs: np.ndarray | sp.csc_array,
    ) -> (
        None
        | np.ndarray
        | sp.csc_array
        | tuple[np.ndarray | sp.csc_array, ...]
    ):
        """
        Truncate the basis set by coherence order. Coherence order
        of a basis state is defined as the sum of the projections `q` of its
        constituent spherical tensor operators.

        Only basis states whose coherence order belongs to
        `coherence_orders` are retained. Optionally supplied state vectors and
        superoperators are converted into the truncated basis.

        Parameters
        ----------
        coherence_orders : list
            List of coherence orders to be retained in the basis.
        *objs : tuple of {ndarray, csc_array}
            Superoperators or state vectors defined in the original basis.

        Returns
        -------
        objs_transformed : ndarray or csc_array or tuple or None
            Transformed objects in the truncated basis. If no objects are
            supplied, `None` is returned.
        """

        status("Truncating the basis set by coherence order...")
        status(f"Retaining coherence orders: {coherence_orders}")
        status(f"Original dimension: {self.dim}")
        time_start = time.time()

        # Initialise the retained basis states and the old-to-new index map
        truncated_basis = []
        index_map = []

        # Retain only basis states with the requested coherence order
        for idx, state in enumerate(self.basis):
            if coherence_order(state) in coherence_orders:
                truncated_basis.append(state)
                index_map.append(idx)

        # Pack the retained basis states into a NumPy array
        truncated_basis = np.array(truncated_basis)

        # Replace the basis without printing duplicate status information
        with HidePrints():
            self.basis = truncated_basis

        status(f"Dimension after truncation: {self.dim}")
        status(f"Completed in {time.time() - time_start:.4f} seconds.\n")

        # Transform any supplied objects into the truncated basis
        if objs:
            objs_transformed = _sop_or_state_to_truncated_basis(objs, index_map)
            return objs_transformed

    def truncate_by_coupling(
        self,
        threshold_J: float=0.01,
        threshold_DD: float=500,
        *objs: np.ndarray | sp.csc_array,
    ) -> (
        None
        | np.ndarray
        | sp.csc_array
        | tuple[np.ndarray | sp.csc_array, ...]
    ):
        """
        Truncate the basis based on the J- and dipole-dipole
        couplings between the spins. 

        A basis state is retained if its participating spins form a connected
        coupling network once weak J- and dipole-dipole couplings have
        been discarded.

        Optionally supplied state vectors and superoperators are converted into
        the truncated basis.

        Parameters
        ----------
        threshold_J : float
            J-couplings less than this value (in Hz) are considered to be zero.
        threshold_DD : float
            Dipole-dipole couplings less than this value (in Hz) are considered
            to be zero.
        *objs : tuple of {ndarray, csc_array}
            Superoperators or state vectors defined in the original basis. These
            will be converted into the truncated basis.

        Returns
        -------
        objs_transformed : ndarray or csc_array or tuple or None
            Transformed objects in the truncated basis. If no objects are
            supplied, `None` is returned.
        """

        status("Truncating the basis set based on spin-spin couplings...")
        status(f"Neglecting J-couplings below {threshold_J} Hz.")
        status(f"Neglecting dipole-dipole couplings below {threshold_DD} Hz.")
        status(f"Original dimension: {self.dim}")
        time_start = time.time()

        # Initialise the retained basis states and the old-to-new index map
        truncated_basis = []
        index_map = []

        # Initialise the dipole-dipole coupling matrix in Hz
        nspins = self._spin_system.nspins
        dd_couplings = np.zeros(shape=(nspins, nspins))

        # Estimate dipole-dipole couplings if Cartesian coordinates are known
        if self._spin_system.xyz is not None:

            # Convert the molecular coordinates from Å to metres
            xyz = self._spin_system.xyz * 1e-10

            # Build the inter-spin vectors and distances
            connectors = xyz[:, np.newaxis] - xyz
            r = np.linalg.norm(connectors, axis=2)

            # Evaluate the pairwise dipole-dipole couplings
            y = self._spin_system.gammas
            for i in range(nspins):
                for j in range(i):
                    dd_couplings[i, j] = (
                        dd_constant(y[i], y[j]) / r[i, j]**3 
                        / (2 * np.pi) # In Hz
                    )

        # Build the binary connectivity matrix from scalar couplings
        j_connectivity = self._spin_system._J_couplings.copy()
        eliminate_small(j_connectivity, zero_value=threshold_J)
        j_connectivity[j_connectivity != 0] = 1

        # Build the binary connectivity matrix from dipole-dipole couplings
        dd_connectivity = dd_couplings.copy()
        eliminate_small(dd_connectivity, zero_value=threshold_DD)
        dd_connectivity[dd_connectivity != 0] = 1

        # Combine both connectivity criteria into a single graph
        connectivity = j_connectivity + dd_connectivity
        connectivity[connectivity != 0] = 1

        # Cache connected-component counts for repeated spin combinations
        conn = {}

        # Retain basis states whose active spins form a connected graph
        for idx, state in enumerate(self.basis):

            # Identify the spins that participate in the current basis state
            idx_spins = np.nonzero(state)[0]

            # Always retain the unit operator
            if len(idx_spins) == 0:
                truncated_basis.append(state)
                index_map.append(idx)
                continue

            # Analyse each active-spin pattern only once
            if tuple(idx_spins) not in conn:

                # Extract the connectivity subgraph for the active spins
                connectivity_curr = connectivity[np.ix_(idx_spins, idx_spins)]

                # Count how many connected components are present
                n_components = connected_components(
                    csgraph = connectivity_curr,
                    directed = False,
                    return_labels = False
                )

                # Store the connectivity result for later reuse
                conn[tuple(idx_spins)] = n_components

            # Retain only states whose active spins form one component
            if conn[tuple(idx_spins)] == 1:
                truncated_basis.append(state)
                index_map.append(idx)

        # Pack the retained basis states into a NumPy array
        truncated_basis = np.array(truncated_basis)

        status(f"Dimension after truncation: {len(truncated_basis)}")
        status(f"Completed in {time.time() - time_start:.4f} seconds.\n")

        # Replace the basis without printing duplicate status information
        with HidePrints():
            self.basis = truncated_basis

        # Transform any supplied objects into the truncated basis
        if objs:
            objs_transformed = _sop_or_state_to_truncated_basis(objs, index_map)
            return objs_transformed

    def truncate_by_zte(
        self,
        L: np.ndarray | sp.csc_array,
        rho: np.ndarray | sp.csc_array,
        *objs: np.ndarray | sp.csc_array,
    ) -> tuple[np.ndarray | sp.csc_array, ...]:
        """
        Truncate the basis using Zero-Track Elimination (ZTE).

        The method follows:

        I. Kuprov (2008):
        https://doi.org/10.1016/j.jmr.2008.08.008

        Parameters
        ----------
        L : ndarray or csc_array
            Liouvillian superoperator
        rho : ndarray or csc_array
            Initial spin density vector.
        *objs : tuple of {ndarray, csc_array}
            Superoperators or state vectors defined in the original basis. These
            will be converted into the truncated basis.

        Returns
        -------
        objs_transformed : tuple of {ndarray, csc_array}
            Liouvillian, density vector, and any additional supplied objects in
            the truncated basis. The transformed `L` and `rho` are always
            included in the returned tuple.
        """

        status("Truncating the basis set using zero-track elimination (ZTE)...")
        status(f"Original dimension: {self.dim}")
        time_start = time.time()

        # Copy the initial state vector for the ZTE propagation loop
        rho_zte = rho.copy()

        # Track the largest absolute value reached by each basis state
        rho_max = abs(np.array(rho_zte))

        # Rescale the ZTE threshold for the current state-vector amplitude
        zero_zte = parameters.zero_zte / max(rho_max[1:, 0])
        status(f"Normalised zero value of ZTE: {zero_zte}")

        # Set the propagation step from the Liouvillian column norm
        time_step = 1 / norm_1(L, ord='col')
        status(f"Time-step of ZTE set to: {time_step:.4e} s")

        # Estimate the number of basis states required initially
        dim = np.sum(rho_max > zero_zte)
        status(f"Basis dimension before ZTE: {self.dim}")

        # Propagate until the required basis dimension stops changing
        for i in range(parameters.nsteps_zte):

            # Propagate the state and update the recorded maxima
            with HidePrints():
                rho_zte = expm_vec(L*time_step, rho_zte, zero_zte)
                rho_max = np.maximum(rho_max, abs(rho_zte))

            # Re-estimate the required basis dimension after propagation
            dim_curr = np.sum(rho_max > zero_zte)
            status(f"ZTE step {i+1} of {parameters.nsteps_zte}. "
                   f"Current basis dimension: {dim_curr}")

            # Stop once the required dimension has converged
            if dim == dim_curr:
                status("Basis dimension converged. Finishing ZTE.")
                break
            else:
                dim = dim_curr

        # Extract the indices of the basis states that survive ZTE
        index_map = list(np.where(rho_max > zero_zte)[0])

        # Replace the basis with the ZTE-truncated basis
        with HidePrints():
            self.basis = self.basis[index_map]

        status(f"Dimension after ZTE: {self.dim}")
        status(f"Completed in {time.time() - time_start:.4f} seconds.\n")

        # Transform the Liouvillian, the state, and any extra objects
        all_objs = (L, rho, *objs)
        objs_transformed = _sop_or_state_to_truncated_basis(all_objs, index_map)
        return objs_transformed

    def truncate_by_indices(
        self,
        indices: list[int] | np.ndarray,
        *objs: np.ndarray | sp.csc_array,
    ) -> (
        None
        | np.ndarray
        | sp.csc_array
        | tuple[np.ndarray | sp.csc_array, ...]
    ):
        """
        Truncate the basis to a user-specified set of indices.

        Parameters
        ----------
        indices : list or ndarray
            List of indices that specify which basis states to retain.
        *objs : tuple of {ndarray, csc_array}
            Superoperators or state vectors defined in the original basis. These
            will be converted into the truncated basis.

        Returns
        -------
        objs_transformed : ndarray or csc_array or tuple or None
            Transformed objects in the truncated basis. If no objects are
            supplied, `None` is returned.
        """

        status("Truncating the basis set based on supplied indices...")
        status(f"Original dimension: {self.dim}")
        time_start = time.time()

        # Sort the retained indices to preserve the original basis order
        indices = np.sort(indices)

        # Replace the basis with the selected basis states
        with HidePrints():
            self.basis = self.basis[indices]

        status(f"Dimension after truncation: {self.dim}")
        status(f"Completed in {time.time() - time_start:.4f} seconds.\n")

        # Transform any supplied objects into the truncated basis
        if objs:
            objs_transformed = _sop_or_state_to_truncated_basis(objs, indices)
            return objs_transformed


def _make_basis(spins: np.ndarray, max_spin_order: int) -> np.ndarray:
    """
    Construct a Liouville-space basis set.

    The basis is spanned by Kronecker products of irreducible spherical tensor
    operators up to the specified maximum spin order. The products themselves
    are not formed explicitly. Instead, each product operator (basis state) 
    is represented as a tuple of integer indices.

    Each integer corresponds to a spherical tensor operator of rank `l` and
    projection `q` through the relation 
    
    `N = l^2 + l - q`. 
    
    The indexing scheme follows:

    Hogben, H. J., Hore, P. J., & Kuprov, I. (2010):
    https://doi.org/10.1063/1.3398146

    Parameters
    ----------
    spins : ndarray
        A one-dimensional array that specifies the spin quantum numbers of the
        spin system.
    max_spin_order : int
        Defines the maximum spin order that is considered in the basis set.

    Returns
    -------
    basis : ndarray
        Basis array whose rows store the integer-encoded product operators.
    """

    # Determine the total number of spins in the system
    nspins = spins.shape[0]

    # Reject invalid maximum spin-order values
    if not isinstance(max_spin_order, int):
        raise TypeError("'max_spin_order' must be an integer.")
    if max_spin_order < 1:
        raise ValueError("'max_spin_order' must be at least 1.")
    if max_spin_order > nspins:
        raise ValueError(
            "'max_spin_order' must not be larger than number of "
            "spins in the system."
        )

    # Enumerate all subsystems with the requested spin order.
    subsystems = combinations(range(nspins), max_spin_order)

    # Collect unique basis states in insertion order
    basis = {}

    # Build the full basis by visiting each subsystem in turn
    state_index = 0
    for subsystem in subsystems:

        # Generate the basis states available within the current subsystem
        sub_basis = _make_subsystem_basis(spins, subsystem)

        # Insert each new state once into the global basis dictionary
        for state in sub_basis:

            # Skip duplicate states that have already been encountered
            if state not in basis:
                basis[state] = state_index
                state_index += 1

    # Convert the collected basis states to a NumPy array
    basis = np.array(list(basis.keys()))

    # Sort the basis so that the first spin index changes slowest.
    sorted_indices = np.lexsort(
        tuple(basis[:, i] for i in reversed(range(basis.shape[1]))))
    basis = basis[sorted_indices]

    return basis


def _make_subsystem_basis(
    spins: np.ndarray,
    subsystem: tuple,
) -> Iterator[tuple[int, ...]]:
    """
    Generate the basis set for a given subsystem.

    Parameters
    ----------
    spins : ndarray
        A one-dimensional array that specifies the spin quantum numbers of the
        spin system.
    subsystem : tuple
        Indices of the spins involved in the subsystem.

    Returns
    -------
    basis : Iterator of tuple of int
        Iterator over the basis set for the given subsystem, represented as
        tuples.

        For example, identity operator and z-operator for the 3rd spin:

        `[(0, 0, 0), (0, 0, 2), ...]`

        where the number 2 comes from the indexing scheme `N = l^2 + l - q` 
        for the T_{1, 0} operator of the 3rd spin.
    """

    # Determine the system size and spin multiplicities
    nspins = spins.shape[0]
    mults = (2*spins + 1).astype(int)

    # Assemble the available operator labels for each spin position
    operators = []

    # Visit each spin position in the full spin system
    for spin in range(nspins):

        # Add all local operators for spins that belong to the subsystem
        if spin in subsystem:

            # Include the full single-spin operator basis at this position
            operators.append(list(range(mults[spin] ** 2)))

        # Otherwise keep only the identity operator at this position
        else:
            operators.append([0])

    # Form all tensor-product combinations for the subsystem basis
    basis = product(*operators)

    return basis


def _sop_or_state_to_truncated_basis(
    objs: tuple[np.ndarray | sp.csc_array, ...],
    index_map: list[int],
) -> (
    np.ndarray
    | sp.csc_array
    | tuple[np.ndarray | sp.csc_array, ...]
):
    """
    Convert superoperators or state vectors into a truncated basis.

    Parameters
    ----------
    objs : tuple
        Tuple of superoperators and state vectors to be converted to the
        truncated basis.
    index_map : list
        An index map between the original basis and the truncated basis.

    Returns
    -------
    objs_transformed : ndarray or csc_array or tuple
        Transformed object, or a tuple of transformed objects if multiple input
        objects are supplied.
    """

    status(
        "Converting superoperators and/or state vectors to the truncated "
        "basis..."
    )
    time_start = time.time()

    # Transform each supplied object according to whether it is a state or SOP
    objs_transformed = []
    for obj in objs:

        # Retain only the selected coefficients for state vectors.
        if isvector(obj):
            objs_transformed.append(obj[index_map])

        # Retain only the selected rows and columns for superoperators.
        else:
            objs_transformed.append(obj[np.ix_(index_map, index_map)])

    # Match the return shape to the number of supplied input objects
    if len(objs_transformed) == 1:
        objs_transformed = objs_transformed[0]
    else:
        objs_transformed = tuple(objs_transformed)

    status(f"Completed in {time.time() - time_start:.4f} seconds.\n")

    return objs_transformed