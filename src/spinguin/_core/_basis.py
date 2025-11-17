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
import math
import warnings
from itertools import product, combinations
from typing import Iterator, Literal
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree
from spinguin._core._la import eliminate_small, expm_vec
from spinguin._core._hide_prints import HidePrints
from spinguin._core._utils import coherence_order, state_idx
from spinguin._core._states import state_to_truncated_basis
from spinguin._core._superoperators import sop_to_truncated_basis
from spinguin._core._la import isvector
from spinguin._core._parameters import parameters

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
    _max_spin_order: int = None
    _spin_system: SpinSystem = None

    def __init__(self, spin_system: SpinSystem):
        print("Basis set has been initialized with the following defaults:")
        print(f"max_spin_order: {self.max_spin_order}\n")

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
    
    @max_spin_order.setter
    def max_spin_order(self, max_spin_order):
        if max_spin_order < 1:
            raise ValueError("Maximum spin order must be at least 1.")
        if max_spin_order > self._spin_system.nspins:
            raise ValueError("Maximum spin order must not be larger than "
                             "the number of spins in the system.")
        self._max_spin_order = max_spin_order
        print(f"Maximum spin order set to: {self.max_spin_order}\n")

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
        self._basis = make_basis(spins = self._spin_system.spins,
                                 max_spin_order = self.max_spin_order)
        
    def indexof(self, op_def: np.ndarray | list | tuple):
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
        
        # Convert the input as array if not already
        op_def = np.asarray(op_def)

        # Obtain the index
        idx = state_idx(self.basis, op_def)

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
        # Truncate the basis and obtain the index map
        truncated_basis, index_map = truncate_basis_by_coherence(
            basis = self.basis,
            coherence_orders = coherence_orders
        )

        # Update the basis
        self._basis = truncated_basis

        # Optionally, convert the superoperators and state vectors to the
        # truncated basis
        if objs:
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

            return objs_transformed
        
    def truncate_by_coupling(
        self,
        threshold: float,
        method: Literal["weakest_link", "network_strength"] = "weakest_link",
        *objs: np.ndarray | sp.csc_array
    ) -> None | np.ndarray | sp.csc_array | tuple[np.ndarray | sp.csc_array]:
        """
        Removes basis states based on the scalar J-couplings. Whenever there
        exists a J-coupling network of sufficient strength between spins that
        constitute a product state, the particular state is kept in the basis.
        Otherwise, the state is removed. The coupling strength is evaluated
        either by the weakest link or by the overall network strength.

        Optionally, superoperators or state vectors can be given as input. These
        will be converted to the truncated basis.

        Parameters
        ----------
        threshold : float
            Coupling strength must be above this value in order for the product
            state to be considered in the basis set.
        method : {"weakest_link", "network_strength"}
            Decides how the importance of a product state is evaluated. Weakest
            link method considers a J-coupling network invalid based on the
            smallest J-coupling within that network. Network strength method
            calculates the effective coupling as a geometric mean scaled by the
            factorial of the number of couplings within the network.
        *objs : tuple of {ndarray, csc_array}
            Superoperators or state vectors defined in the original basis. These
            will be converted into the truncated basis.

        Returns
        -------
        objs_transformed : tuple of {ndarray, csc_array}
            Superoperators and state vectors transformed into the truncated
            basis.
        """
        # Truncate the basis and obtain the index map
        truncated_basis, index_map = truncate_basis_by_coupling(
            basis = self.basis,
            J_couplings = self._spin_system.J_couplings,
            threshold = threshold,
            method = method
        )

        # Update the basis
        self._basis = truncated_basis

        # Optionally, convert the superoperators and state vectors to the
        # truncated basis
        if objs:
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

            return objs_transformed
        
    def truncate_by_zte(
        self,
        L: np.ndarray | sp.csc_array,
        rho: np.ndarray | sp.csc_array,
        time_step: float,
        nsteps: int,
        *objs: np.ndarray | sp.csc_array
    ) -> None | np.ndarray | sp.csc_array | tuple[np.ndarray | sp.csc_array]:
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
        time_step : float
            Time step of the propagation within the ZTE.
        nsteps : int
            Number of steps to take in the ZTE.
        *objs : tuple of {ndarray, csc_array}
            Superoperators or state vectors defined in the original basis. These
            will be converted into the truncated basis.

        Returns
        -------
        objs_transformed : tuple of {ndarray, csc_array}
            Superoperators and state vectors transformed into the truncated
            basis.
        """
        # Truncate the basis and obtain the index map
        truncated_basis, index_map = truncate_basis_by_zte(
            basis = self.basis,
            L = L,
            rho = rho,
            time_step = time_step,
            nsteps = nsteps,
            zero_zte = parameters.zero_zte,
            zero_expm_vec = parameters.zero_time_step
        )

        # Update the basis
        self._basis = truncated_basis

        # Optionally, convert the superoperators and state vectors to the
        # truncated basis
        if objs:
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
        # Obtain the truncated basis
        truncated_basis = truncate_basis_by_indices(
            basis = self.basis,
            indices = indices
        )

        # Update the basis
        self._basis = truncated_basis

        # Optionally, convert the superoperators and state vectors to the
        # truncated basis
        if objs:
            objs_transformed = []
            for obj in objs:

                # Consider state vectors
                if isvector(obj):
                    objs_transformed.append(state_to_truncated_basis(
                        index_map=indices,
                        rho=obj))
                    
                # Consider superoperators
                else:
                    objs_transformed.append(sop_to_truncated_basis(
                        index_map=indices,
                        sop=obj
                    ))

            # Convert to tuple or just single value
            if len(objs_transformed) == 1:
                objs_transformed = objs_transformed[0]
            else:
                objs_transformed = tuple(objs_transformed)

            return objs_transformed
        
def make_basis(spins: np.ndarray, max_spin_order: int):
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
        sub_basis = make_subsystem_basis(spins, subsystem)

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

def make_subsystem_basis(spins: np.ndarray, subsystem: tuple) -> Iterator:
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
    
def truncate_basis_by_coherence(
    basis: np.ndarray,coherence_orders: list
) -> tuple[np.ndarray, list]:
    """
    Truncates the basis set by retaining only the product operators that
    correspond to coherence orders specified in the `coherence_orders` list.

    The function generates an index map from the original basis to the truncated
    basis.
    This map can be used to transform superoperators or state vectors to the new
    basis.

    Parameters
    ----------
    basis : ndarray
        A two-dimensional array where each row contains integers that represent
        a Kronecker product of single-spin irreducible spherical tensors.
    coherence_orders : list
        List of coherence orders to be retained in the basis.

    Returns
    -------
    truncated_basis : ndarray
        A two-dimensional array containing the basis set with only the specified
        coherence orders retained.
    index_map : list
        List that contains an index map from the original basis to the truncated
        basis.
    """

    print("Truncating the basis set. The following coherence orders are "
          f"retained: {coherence_orders}")
    time_start = time.time()

    # Create an empty list for the new basis
    truncated_basis = []

    # Create an empty list for the mapping from old to new basis
    index_map = []

    # Iterate over the basis
    for idx, state in enumerate(basis):

        # Check if coherence order is in the list
        if coherence_order(state) in coherence_orders:

            # Assign state to the truncated basis and increment index
            truncated_basis.append(state)

            # Assign index to the index map
            index_map.append(idx)

    # Convert basis to NumPy array
    truncated_basis = np.array(truncated_basis)

    print("Truncated basis created.")
    print(f"Original dimension: {len(basis)}")
    print(f"Truncated dimension: {len(truncated_basis)}")
    print(f"Elapsed time: {time.time() - time_start:.4f} seconds.")
    print()

    return truncated_basis, index_map

def truncate_basis_by_coupling_weakest_link(
    basis: np.ndarray,
    J_couplings: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, list]:
    """
    Removes basis states based on the scalar J-couplings. Whenever there exists
    a coupling network between the spins that constitute the product state, in
    which the couplings surpass the given threshold, the basis state is kept.
    Otherwise, the basis state is dropped.

    Parameters
    ----------
    basis : ndarray
        A two-dimensional array where each row contains integers that represent
        a Kronecker product of single-spin irreducible spherical tensors.
    J_couplings : ndarray
        A two-dimensional array that contains the scalar J-couplings between
        the spins in Hz.
    threshold : float
        J-coupling between two spins must be above this value in order for the
        algorithm to consider them connected.

    Returns
    -------
    truncated_basis : ndarray
        A two-dimensional array containing the truncated basis set.
    index_map : list
        List that contains an index map from the original basis to the truncated
        basis.
    """
    print("Truncating the basis set based on J-couplings.")
    time_start = time.time()

    # Create an empty list for the new basis
    truncated_basis = []

    # Create an empty list for the mapping from old to new basis
    index_map = []

    # Create the connectivity matrix from the J-couplings
    J_connectivity = J_couplings.copy()
    eliminate_small(J_connectivity, zero_value=threshold)
    J_connectivity[J_connectivity!=0] = 1

    # Cache the connectivity of spins
    connectivity = dict()

    # Iterate over the basis
    for idx, state in enumerate(basis):

        # Obtain the indices of the participating spins
        idx_spins = np.nonzero(state)[0]

        # Special case always include the unit state:
        if len(idx_spins) == 0:
            truncated_basis.append(state)
            index_map.append(idx)
            continue

        # Analyse the connectivity if not already
        if not tuple(idx_spins) in connectivity:

            # Obtain the current connectivity graph
            J_connectivity_curr = J_connectivity[np.ix_(idx_spins, idx_spins)]

            # Calculate the number of connected components
            n_components = connected_components(
                csgraph = J_connectivity_curr,
                directed = False,
                return_labels = False
            )

            connectivity[tuple(idx_spins)] = n_components

        # If the state is connected, keep it
        if connectivity[tuple(idx_spins)] == 1:
            truncated_basis.append(state)
            index_map.append(idx)

    # Convert basis to NumPy array
    truncated_basis = np.array(truncated_basis)

    print("Truncated basis created.")
    print(f"Original dimension: {len(basis)}")
    print(f"Truncated dimension: {len(truncated_basis)}")
    print(f"Elapsed time: {time.time() - time_start:.4f} seconds.")
    print()

    return truncated_basis, index_map

def truncate_basis_by_coupling_network_strength(
    basis: np.ndarray,
    J_couplings: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, list]:
    """
    Removes basis states based on the scalar J-couplings. The coupling network
    within a product state is evaluated based on the maximum overall coupling
    strength defined as the geometric mean of the J-couplings divided by the
    factorial of the number of the couplings.
    
    TODO: More rigorous way to estimate the network strength?

    Parameters
    ----------
    basis : ndarray
        A two-dimensional array where each row contains integers that represent
        a Kronecker product of single-spin irreducible spherical tensors.
    J_couplings : ndarray
        A two-dimensional array that contains the scalar J-couplings between
        the spins in Hz.
    threshold : float
        Calculated effective J-coupling network strength must be above this
        value in order for the algorithm to consider them connected.

    Returns
    -------
    truncated_basis : ndarray
        A two-dimensional array containing the truncated basis set.
    index_map : list
        List that contains an index map from the original basis to the truncated
        basis.
    """
    print("Truncating the basis set based on J-couplings.")
    time_start = time.time()

    # Create an empty list for the new basis
    truncated_basis = []

    # Create an empty list for the mapping from old to new basis
    index_map = []

    # Prepare the coupling matrix for the Kruskal's algorithm
    J_couplings = -np.abs(J_couplings)

    # Iterate over the basis
    for idx, state in enumerate(basis):

        # Obtain the indices of the participating spins
        idx_spins = np.nonzero(state)[0]

        # Special case: always include the unit state and one-spin states:
        if len(idx_spins) in {0, 1}:
            truncated_basis.append(state)
            index_map.append(idx)
            continue

        # Obtain the current J-coupling network
        J_couplings_curr = J_couplings[np.ix_(idx_spins, idx_spins)]

        # Obtain the maximum spanning tree
        maxtree = abs(minimum_spanning_tree(J_couplings_curr))

        # Continue only if the state is connected in the first place
        connections = len(idx_spins) - 1
        if maxtree.nnz == connections:

            # Calculate the coupling strength
            geomean = np.prod(maxtree.data) ** (1/connections)
            coupling_strength = geomean / math.factorial(connections)

            # Include the state if the coupling strength is above threshold
            if coupling_strength >= threshold:
                truncated_basis.append(state)
                index_map.append(idx)

    # Convert basis to NumPy array
    truncated_basis = np.array(truncated_basis)

    print("Truncated basis created.")
    print(f"Original dimension: {len(basis)}")
    print(f"Truncated dimension: {len(truncated_basis)}")
    print(f"Elapsed time: {time.time() - time_start:.4f} seconds.")
    print()

    return truncated_basis, index_map

def truncate_basis_by_coupling(
    basis: np.ndarray,
    J_couplings: np.ndarray,
    threshold: float,
    method: Literal["weakest_link", "network_strength"] = "weakest_link"
) -> tuple[np.ndarray, list]:
    """
    Removes basis states based on the scalar J-couplings. Whenever there exists
    a J-coupling network of sufficient strength between spins that constitute a
    product state, the particular state is kept in the basis. Otherwise, the
    state is removed. The coupling strength is evaluated either by the weakest
    link or by the overall network strength.

    Parameters
    ----------
    basis : ndarray
        A two-dimensional array where each row contains integers that represent
        a Kronecker product of single-spin irreducible spherical tensors.
    J_couplings : ndarray
        A two-dimensional array that contains the scalar J-couplings between
        the spins in Hz.
    threshold : float
        Coupling strength must be above this value in order for the product
        state to be considered in the basis set.
    method : {"weakest_link", "network_strength"}
        Decides how the importance of a product state is evaluated. Weakest link
        method considers a J-coupling network invalid based on the smallest J-
        coupling within that network. Network strength method calculates the
        effective coupling as a geometric mean scaled by the factorial of the
        number of couplings within the network.

    Returns
    -------
    truncated_basis : ndarray
        A two-dimensional array containing the truncated basis set.
    index_map : list
        List that contains an index map from the original basis to the truncated
        basis.
    """
    if method == "weakest_link":
        return truncate_basis_by_coupling_weakest_link(
            basis = basis,
            J_couplings = J_couplings,
            threshold = threshold
        )
    elif method == "network_strength":
        return truncate_basis_by_coupling_network_strength(
            basis = basis,
            J_couplings = J_couplings,
            threshold = threshold
        )
    else:
        raise ValueError(f"Invalid truncation method {method}.")
    
def truncate_basis_by_zte(
    basis: np.ndarray,
    L: np.ndarray | sp.csc_array,
    rho: np.ndarray | sp.csc_array,
    time_step: float,
    nsteps: int,
    zero_zte: float,
    zero_expm_vec: float
) -> tuple[np.ndarray, list]:
    """
    Removes basis states using the Zero-Track Elimination (ZTE) described in:

    Kuprov, I. (2008):
    https://doi.org/10.1016/j.jmr.2008.08.008

    Parameters
    ----------
    basis : ndarray
        A two-dimensional array where each row contains integers that represent
        a Kronecker product of single-spin irreducible spherical tensors.
    L : ndarray or csc_array
        Liouvillian superoperator, L = -iH - R + K.
    rho : ndarray or csc_array
        Initial spin density vector.
    time_step : float
        Time step of the propagation within the ZTE.
    nsteps : int
        Number of steps to take in the ZTE.
    zero_zte : float
        If state population is below this value, it is dropped from the basis.
    zero_expm_vec: float
        Convergence criterion to be used when calculating the action of matrix
        exponential of the Liouvillian to the state vector.

    Returns
    -------
    truncated_basis : ndarray
        A two-dimensional array containing the truncated basis set.
    index_map : list
        List that contains an index map from the original basis to the truncated
        basis.
    """
    print("Truncating the basis set using zero-track elimination.")
    time_start = time.time()

    # Create empty vector for the maximum values of rho
    rho_max = abs(np.array(rho))

    # Scale the zero value of the ZTE to take into account different norms
    scaling_zv = abs(rho).max()
    zero_zte = zero_zte / scaling_zv

    # Propagate for few steps
    for i in range(nsteps):
        print(f"ZTE step {i+1} of {nsteps}...")
        with HidePrints():
            rho = expm_vec(L*time_step, rho, zero_expm_vec)
            rho_max = np.maximum(rho_max, abs(rho))

    # Obtain indices of states that should remain
    index_map = list(np.where(rho_max > zero_zte)[0])

    # Obtain the truncated basis
    truncated_basis = basis[index_map]

    print("Truncated basis created.")
    print(f"Original dimension: {len(basis)}")
    print(f"Truncated dimension: {len(truncated_basis)}")
    print(f"Elapsed time: {time.time() - time_start:.4f} seconds.")
    print()

    return truncated_basis, index_map

def truncate_basis_by_indices(
    basis: np.ndarray,
    indices: list | np.ndarray
) -> np.ndarray:
    """
    Truncate the basis set to include only the basis states specified by the
    `indices` supplied by the user.

    Parameters
    ----------
    basis : ndarray
        A two-dimensional array where each row contains integers that represent
        a Kronecker product of single-spin irreducible spherical tensors.
    indices : list or ndarray
        List of indices that specify which basis states to retain.

    Returns
    -------
    truncated_basis : ndarray
        A two-dimensional array containing the truncated basis set.
    """
    print("Truncating the basis set based on supplied indices.")
    time_start = time.time()

    # Sort the indices
    indices = np.sort(indices)

    # Obtain the truncated basis
    truncated_basis = basis[indices]

    print("Truncated basis created.")
    print(f"Original dimension: {len(basis)}")
    print(f"Truncated dimension: {len(truncated_basis)}")
    print(f"Elapsed time: {time.time() - time_start:.4f} seconds.")
    print()

    return truncated_basis