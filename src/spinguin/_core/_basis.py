"""
This module provides the Basis class which is assigned as a part of `SpinSystem`
object upon its instantiation. It provides functionality for constructing and
truncating the basis set. The basis set functionality is designed to be accessed
through the `SpinSystem` object. Example::

    import spinguin as sg                   # Import the package
    spin_system = sg.SpinSystem(["1H"])     # Create an example spin system
    spin_system.basis.max_spin_order = 1    # Set the maximum spin order
    spin_system.basis.build()               # Build the basis set
"""
# Referencing SpinSystem class
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from spinguin._core._spin_system import SpinSystem

# Imports
import numpy as np
import scipy.sparse as sp
import time
import warnings
from itertools import product, combinations
from typing import Iterator
from scipy.sparse.csgraph import connected_components
from scipy.constants import mu_0, hbar
from spinguin._core._la import (
    eliminate_small,
    expm_vec,
    arraylike_to_tuple,
    norm_1
)
from spinguin._core._hide_prints import HidePrints
from spinguin._core._utils import coherence_order
from spinguin._core._states import state_to_truncated_basis
from spinguin._core._superoperators import sop_to_truncated_basis
from spinguin._core._la import isvector
from spinguin._core._parameters import parameters
from spinguin._core._status import status

class Basis:
    """
    Basis class manages the basis set of a spin system. Most importantly, the
    basis set contains the information on the truncation of the basis set and is
    responsible for building and making changes to the basis set.

    The `Basis` instance is designed to be created and accessed through the
    `SpinSystem` object. For example::

        import spinguin as sg                   # Import the package
        spin_system = sg.SpinSystem(["1H"])     # Create an example spin system
        spin_system.basis.max_spin_order = 1    # Set the maximum spin order
        spin_system.basis.build()               # Build the basis set
    """

    # Basis set properties
    _basis: np.ndarray = None
    _basis_dict: dict = None
    _max_spin_order: int = None
    _spin_system: SpinSystem = None

    def __init__(self, spin_system: SpinSystem):

        # Store a reference to the SpinSystem
        self._spin_system = spin_system

    @property
    def dim(self) -> int:
        """Dimension of the basis set."""
        return self.basis.shape[0]

    @property
    def max_spin_order(self) -> int:
        """
        Specifies the maximum number of active spins that are included in the
        product operators that constitute the basis set. Must be at least 1 and
        not larger than the number of spins in the system.
        """
        return self._max_spin_order
    
    @property
    def basis(self) -> np.ndarray:
        """
        Contains the actual basis set as an array of dimensions (N, M) where
        N is the number of states in the basis and M is the number of spins in
        the system. The basis set is constructed from Kronecker products of
        irreducible spherical tensor operators, which are indexed using integers
        starting from 0 with increasing rank `l` and decreasing projection `q`:

        - 0 --> T(0, 0)
        - 1 --> T(1, 1)
        - 2 --> T(1, 0)
        - 3 --> T(1, -1) and so on...

        """
        return self._basis
    
    @basis.setter
    def basis(self, basis: np.ndarray):
        # Check the input
        if not isinstance(basis, np.ndarray):
            raise ValueError("Basis set must be a NumPy array.")
        if basis.ndim != 2:
            raise ValueError("Basis set must be a two-dimensional array.")
        if basis.shape[1] != self._spin_system.nspins:
            raise ValueError("Mismatch between basis set and number of spins.")

        # Change the basis set to immutable
        basis.flags.writeable = False

        # Set the basis set and the basis dictionary
        self._basis = basis
        self._basis_dict = {tuple(state): i for i, state in enumerate(basis)}

        status(f"Basis set has been set. Dimension: {self.dim}.\n")
    
    @max_spin_order.setter
    def max_spin_order(self, max_spin_order):
        if max_spin_order < 1:
            raise ValueError("Maximum spin order must be at least 1.")
        if max_spin_order > self._spin_system.nspins:
            raise ValueError("Maximum spin order must not be larger than "
                             "the number of spins in the system.")
        self._max_spin_order = max_spin_order
        status(f"Maximum spin order set to: {self.max_spin_order}\n")

    def build(self):
        """
        Builds the basis set for the spin system. Prior to building the basis,
        the maximum spin order should be defined. If it is not defined, it is
        set equal to the number of spins in the system (may be very slow)!
        """
        # If maximum spin order is not specified, raise a warning and set it
        # equal to the number of spins
        if self.max_spin_order is None:
            warnings.warn("Maximum spin order not specified. "
                          "Defaulting to the number of spins.")
            self.max_spin_order = self._spin_system.nspins

        # Build the basis
        self.basis = _make_basis(spins = self._spin_system.spins,
                                 max_spin_order = self.max_spin_order)
        
    def indexof(self, op_def: np.ndarray | list | tuple) -> int:
        """
        Finds the index of the basis state defined by `op_def`.

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
            Index of the state in the basis set.
        """
        # Raise a warning if basis has not been built
        if self.basis is None:
            raise ValueError("Basis must be built before obtaining indices.")
        
        # Convert the input as tuple
        op_def = arraylike_to_tuple(op_def)

        # Obtain the index
        if op_def in self._basis_dict:
            idx = self._basis_dict[op_def]
        else:
            raise ValueError(f"Could not find {op_def} in the basis set.")

        return idx
        
    def truncate_by_coherence(
        self,
        coherence_orders: list,
        *objs: np.ndarray | sp.csc_array
    ) -> None | np.ndarray | sp.csc_array | tuple[np.ndarray | sp.csc_array]:
        """
        Truncates the basis set by retaining only the product operators that
        correspond to coherence orders specified in the `coherence_orders` list.

        Optionally, superoperators or state vectors can be given as input. These
        will be converted to the truncated basis.

        Parameters
        ----------
        coherence_orders : list
            List of coherence orders to be retained in the basis.

        Returns
        -------
        objs_transformed : ndarray or csc_array or tuple
            Superoperators and state vectors transformed into the truncated
            basis.
        """
        status("Truncating the basis set...")
        status(f"\tRetaining coherence orders {coherence_orders}")
        status(f"\tOriginal dimension: {self.dim}")
        time_start = time.time()

        # Create an empty list for the new basis
        truncated_basis = []

        # Create an empty list for the mapping from old to new basis
        index_map = []

        # Iterate over the basis
        for idx, state in enumerate(self.basis):

            # Check if coherence order is in the list
            if coherence_order(state) in coherence_orders:

                # Assign state to the truncated basis and increment index
                truncated_basis.append(state)

                # Assign index to the index map
                index_map.append(idx)

        # Convert basis to NumPy array
        truncated_basis = np.array(truncated_basis)

        # Update the basis
        with HidePrints():
            self.basis = truncated_basis
        
        status(f"\tTruncated dimension: {len(truncated_basis)}")
        status(f"Completed in {time.time() - time_start:.4f} seconds.\n")

        # Optionally, convert the superoperators and state vectors to the
        # truncated basis
        if objs:
            objs_transformed = _sop_or_state_to_truncated_basis(objs, index_map)
            return objs_transformed
        
    def truncate_by_coupling(
        self,
        threshold_J: float=0.01,
        threshold_DD: float=500,
        *objs: np.ndarray | sp.csc_array
    ) -> None | np.ndarray | sp.csc_array | tuple[np.ndarray | sp.csc_array]:
        """
        Removes basis states based on the scalar J-couplings and DD-couplings.
        Whenever there exists a coupling network of sufficient strength between
        spins that constitute a product state, the particular state is kept in
        the basis. Otherwise, the state is removed.

        Optionally, superoperators or state vectors can be given as input. These
        will be converted to the truncated basis.

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
        objs_transformed : tuple of {ndarray, csc_array}
            Superoperators and state vectors transformed into the truncated
            basis.
        """
        status("Truncating the basis set based on couplings...")
        status(f"\tJ-couplings <{threshold_J} Hz are considered zero.")
        status(f"\tDD-couplings <{threshold_DD} Hz are considered zero.")
        status(f"\tOriginal dimension: {self.dim}")
        time_start = time.time()

        # Create an empty list for the new basis
        truncated_basis = []

        # Create an empty list for the mapping from old to new basis
        index_map = []

        # Initialise an array for the DD-couplings
        nspins = self._spin_system.nspins
        DD_couplings = np.zeros(shape=(nspins, nspins))

        # Calculate DD-couplings only if XYZ is assigned
        if self._spin_system.xyz is not None:
        
            # Convert the molecular coordinates to SI units
            xyz = self._spin_system.xyz * 1e-10

            # Get the connector and distance arrays
            connectors = xyz[:, np.newaxis] - xyz
            r = np.linalg.norm(connectors, axis=2)

            # Get the DD-couplings (in Hz)
            y = self._spin_system.gammas
            for i in range(nspins):
                for j in range(i):
                    DD_couplings[i, j] = \
                        - mu_0 * y[i] * y[j] * hbar / (8*np.pi**2 * r[i,j]**3)

        # Create the connectivity matrix from the J-couplings
        J_connectivity = self._spin_system._J_couplings.copy()
        eliminate_small(J_connectivity, zero_value=threshold_J)
        J_connectivity[J_connectivity!=0] = 1

        # Create the connectivity matrix from the DD-couplings
        DD_connectivity = DD_couplings.copy()
        eliminate_small(DD_connectivity, zero_value=threshold_DD)
        DD_connectivity[DD_connectivity!=0] = 1

        # Get the union of the connectivity matrices
        connectivity = J_connectivity + DD_connectivity
        connectivity[connectivity!=0] = 1

        # Cache the connectivity of spins
        conn = dict()

        # Iterate over the basis
        for idx, state in enumerate(self.basis):

            # Obtain the indices of the participating spins
            idx_spins = np.nonzero(state)[0]

            # Special case: always include the unit state
            if len(idx_spins) == 0:
                truncated_basis.append(state)
                index_map.append(idx)
                continue

            # Analyse the connectivity if not already
            if not tuple(idx_spins) in conn:

                # Obtain the current connectivity graphs
                connectivity_curr = connectivity[np.ix_(idx_spins, idx_spins)]

                # Calculate the number of connected components
                n_components = connected_components(
                    csgraph = connectivity_curr,
                    directed = False,
                    return_labels = False
                )

                # Store the connectivity of the state
                conn[tuple(idx_spins)] = n_components

            # If the state is connected, keep it
            if conn[tuple(idx_spins)] == 1:
                truncated_basis.append(state)
                index_map.append(idx)

        # Convert basis to NumPy array
        truncated_basis = np.array(truncated_basis)

        status(f"\tTruncated dimension: {len(truncated_basis)}")
        status(f"Completed in {time.time() - time_start:.4f} seconds.\n")

        # Update the basis
        with HidePrints():
            self.basis = truncated_basis

        # Optionally, convert the superoperators and state vectors to the
        # truncated basis
        if objs:
            objs_transformed = _sop_or_state_to_truncated_basis(objs, index_map)
            return objs_transformed
        
    def truncate_by_zte(
        self,
        L: np.ndarray | sp.csc_array,
        rho: np.ndarray | sp.csc_array,
        *objs: np.ndarray | sp.csc_array
    ) ->  tuple[np.ndarray | sp.csc_array]:
        """
        Removes basis states using the Zero-Track Elimination (ZTE) described
        in:

        Kuprov, I. (2008):
        https://doi.org/10.1016/j.jmr.2008.08.008

        Parameters
        ----------
        L : ndarray or csc_array
            Liouvillian superoperator, L = -iH - R + K.
        rho : ndarray or csc_array
            Initial spin density vector.
        *objs : tuple of {ndarray, csc_array}
            Superoperators or state vectors defined in the original basis. These
            will be converted into the truncated basis.

        Returns
        -------
        objs_transformed : tuple of {ndarray, csc_array}
            Superoperators and state vectors transformed into the truncated
            basis. The Liouvillian and the density vector, which are given as
            the input, are always returned.
        """
        status("Truncating the basis set using zero-track elimination...")
        status(f"\tOriginal dimension: {self.dim}")
        time_start = time.time()

        # Take a copy of the state vector for ZTE
        rho_zte = rho.copy()

        # Create a vector to store the maximum values of rho
        rho_max = abs(np.array(rho_zte))

        # Scale the zero value of the ZTE to take into account different norms
        # and the possibility of having a hyperpolarised state
        zero_zte = parameters.zero_zte / max(rho_max[1:, 0])
        status(f"\tNormalised zero value of ZTE: {zero_zte}")

        # Obtain the time step
        time_step = 1 / norm_1(L, ord='col')
        status(f"\tTime-step of ZTE set to: {time_step:.4e} s")

        # Obtain the dimension required to describe density vector
        dim = np.sum(rho_max > zero_zte)
        status(
            "\tBefore ZTE. "
            f"Current required basis dimension: {dim}"
        )

        # Perform the ZTE steps
        for i in range(parameters.nsteps_zte):

            # Propagate the state forward and update the maximum values
            with HidePrints():
                rho_zte = expm_vec(L*time_step, rho_zte, zero_zte)
                rho_max = np.maximum(rho_max, abs(rho_zte))

            # Obtain the dimension required to describe density vector
            dim_curr = np.sum(rho_max > zero_zte)
            status(
                f"\tZTE step {i+1} of {parameters.nsteps_zte}. "
                f"Current required basis dimension: {dim_curr}"
            )

            # Terminate ZTE if the required dimension remains the same
            if dim == dim_curr:
                status("\tDimension remained the same. Finishing ZTE.")
                break
            else:
                dim = dim_curr

        # Obtain indices of states that should remain in the basis
        index_map = list(np.where(rho_max > zero_zte)[0])

        # Update the basis
        with HidePrints():
            self.basis = self.basis[index_map]

        status(f"\tTruncated dimension: {self.dim}")
        status(f"Completed in: {time.time() - time_start:.4f} seconds.\n")

        # Convert the L and rho to the truncated basis. In addition, convert
        # other objects given as input
        all_objs = (L, rho, *objs)
        objs_transformed = _sop_or_state_to_truncated_basis(all_objs, index_map)
        return objs_transformed
        
    def truncate_by_indices(
        self,
        indices: list | np.ndarray,
        *objs: np.ndarray | sp.csc_array
    ) -> np.ndarray:
        """
        Truncate the basis set to include only the basis states specified by the
        `indices` supplied by the user.

        Parameters
        ----------
        indices : list or ndarray
            List of indices that specify which basis states to retain.
        *objs : tuple of {ndarray, csc_array}
            Superoperators or state vectors defined in the original basis. These
            will be converted into the truncated basis.

        Returns
        -------
        objs_transformed : tuple of {ndarray, csc_array}
            Superoperators and state vectors transformed into the truncated
            basis.
        """
        status("Truncating the basis set based on supplied indices...")
        status(f"\tOriginal dimension: {self.dim}")
        time_start = time.time()

        # Sort the indices
        indices = np.sort(indices)

        # Update the basis
        with HidePrints():
            self.basis = self.basis[indices]

        status(f"\tTruncated dimension: {self.dim}")
        status(f"Completed in {time.time() - time_start:.4f} seconds.\n")

        # Optionally, convert the superoperators and state vectors to the
        # truncated basis
        if objs:
            objs_transformed = _sop_or_state_to_truncated_basis(objs, indices)
            return objs_transformed
        
def _make_basis(spins: np.ndarray, max_spin_order: int):
    """
    Constructs a Liouville-space basis set, where the basis is spanned by all
    possible Kronecker products of irreducible spherical tensor operators, up
    to the defined maximum spin order.

    The Kronecker products themselves are not calculated. Instead, the operators
    are expressed as sequences of integers, where each integer represents a
    spherical tensor operator of rank `l` and projection `q` using the following
    relation: `N = l^2 + l - q`. The indexing scheme has been adapted from:

    Hogben, H. J., Hore, P. J., & Kuprov, I. (2010):
    https://doi.org/10.1063/1.3398146

    Parameters
    ----------
    spins : ndarray
        A one-dimensional array that specifies the spin quantum numbers of the
        spin system.
    max_spin_order : int
        Defines the maximum spin entanglement that is considered in the basis
        set.
    """

    # Find the number of spins in the system
    nspins = spins.shape[0]

    # Catch out-of-range maximum spin orders
    if max_spin_order < 1:
        raise ValueError("'max_spin_order' must be at least 1.")
    if max_spin_order > nspins:
        raise ValueError("'max_spin_order' must not be larger than number of"
                         "spins in the system.")

    # Get all possible subsystems of the specified maximum spin order
    indices = [i for i in range(nspins)]
    subsystems = combinations(indices, max_spin_order)

    # Create an empty dictionary for the basis set
    basis = {}

    # Iterate through all subsystems
    state_index = 0
    for subsystem in subsystems:

        # Get the basis for the subsystem
        sub_basis = _make_subsystem_basis(spins, subsystem)

        # Iterate through the states in the subsystem basis
        for state in sub_basis:

            # Add state to the basis set if not already added
            if state not in basis:
                basis[state] = state_index
                state_index += 1

    # Convert dictionary to NumPy array
    basis = np.array(list(basis.keys()))

    # Sort the basis (index of the first spin changes the slowest)
    sorted_indices = np.lexsort(
        tuple(basis[:, i] for i in reversed(range(basis.shape[1]))))
    basis = basis[sorted_indices]
    
    return basis

def _make_subsystem_basis(spins: np.ndarray, subsystem: tuple) -> Iterator:
    """
    Generates the basis set for a given subsystem.

    Parameters
    ----------
    spins : ndarray
        A one-dimensional array that specifies the spin quantum numbers of the
        spin system.
    subsystem : tuple
        Indices of the spins involved in the subsystem.

    Returns
    -------
    basis : Iterator
        An iterator over the basis set for the given subsystem, represented as
        tuples.

        For example, identity operator and z-operator for the 3rd spin:
        `[(0, 0, 0), (0, 0, 2), ...]`
    """

    # Extract the necessary information from the spin system
    nspins = spins.shape[0]
    mults = (2*spins + 1).astype(int)

    # Define all possible spin operators for each spin
    operators = []

    # Loop through every spin in the full system
    for spin in range(nspins):

        # Add spin if it exists in the subsystem
        if spin in subsystem:

            # Add all possible states of the given spin
            operators.append(list(range(mults[spin] ** 2)))

        # Add identity state if not
        else:
            operators.append([0])

    # Get all possible product operator states in the subsystem
    basis = product(*operators)

    return basis

def _sop_or_state_to_truncated_basis(objs: tuple, index_map: list):
    """
    Internal helper function to convert the superoperators or state vectors
    into the truncated basis set.

    Parameters
    ----------
    objs : tuple
        Tuple of superoperators and state vectors to be converted to the
        truncated basis.
    index_map : list
        An index map between the original basis and the truncated basis.

    Returns
    -------
    objs_transformed : list
        List of superoperators and state vectors transformed into the truncated
        basis.
    """
    status("Converting superoperators or state vectors to the "
           "truncated basis...")
    time_start = time.time()
    objs_transformed = []
    for obj in objs:

        # Consider state vectors
        if isvector(obj):
            objs_transformed.append(state_to_truncated_basis(
                index_map=index_map,
                rho=obj))
            
        # Consider superoperators
        else:
            objs_transformed.append(sop_to_truncated_basis(
                index_map=index_map,
                sop=obj
            ))

    # Convert to tuple or just single value
    if len(objs_transformed) == 1:
        objs_transformed = objs_transformed[0]
    else:
        objs_transformed = tuple(objs_transformed)

    status(f"Completed in {time.time() - time_start:.4f} seconds.\n")

    return objs_transformed