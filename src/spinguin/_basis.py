"""
basis.py

Defines the `Basis` class, which manages the basis set for a spin system. 
This includes constructing the basis set, modifying it, and enabling efficient searches. 
The basis set is automatically created when a spin system is initialized.
"""

# For referencing the SpinSystem class
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from spinguin._spin_system import SpinSystem

# Imports
import numpy as np
import time
import math
from itertools import product, combinations
from scipy.sparse import csc_array
from spinguin import _la
from typing import Iterator, Tuple

class Basis:
    """
    Represents the basis set for a spin system. Responsible for constructing the basis set,
    making modifications, and enabling efficient searching.
    """

    def __init__(self, spin_system: SpinSystem):
        """
        Initializes the basis set for the `spin_system`.

        Parameters
        ----------
        spin_system : SpinSystem
            A SpinSystem object containing the details of the system.

        Attributes
        ----------
        arr : numpy.ndarray
            A two-dimensional array where each row contains integers that
            represent the orthogonal product operators that construct the basis set.
            Set directly by `make_basis`.
        dict : dict
            A dictionary where the keys are tuples of integers that
            represent the product operators. The values are the indices of
            the specific operators. Set directly by `make_basis`.

        TODO: Yleiset funktiot kantafilttereille.
        """

        # Create the basis
        self.make_basis(spin_system)

    @property
    def dim(self) -> int:
        """Returns the dimension (number of states) of the basis set."""
        return self.arr.shape[0]
    
    @property
    def nspins(self) -> int:
        """Returns the number of spins in the system."""
        return self.arr.shape[1]
    
    @property
    def uid(self) -> int:
        """Returns a unique identifier for the basis set."""
        return hash(self.arr.tobytes())
        
    def make_basis(self, spin_system: SpinSystem):
        """
        Constructs the basis set from all possible product operator combinations.
        The following attributes will be assigned:

        self.arr : numpy.ndarray
        self.dict : dict

        This method is called during the initialization of a basis.

        Parameters
        ----------
        spin_system : SpinSystem
            The spin system for which the basis is being constructed.
        """

        # Extract the necessary information from the spin system
        size = spin_system.size
        max_spin_order = spin_system.max_spin_order

        # Get all possible subsystems of the specified maximum spin order
        indices = [i for i in range(size)]
        subsystems = combinations(indices, max_spin_order)

        # Create an empty dictionary for the basis set
        basis = {}

        # Iterate through all subsystems
        state_index = 0
        for subsystem in subsystems:

            # Get the basis for the subsystem
            sub_basis = self.make_subsystem_basis(spin_system, subsystem)

            # Iterate through the states in the subsystem basis
            for state in sub_basis:

                # Add state to the basis set if not already added
                if state not in basis:
                    basis[state] = state_index
                    state_index += 1

        # Convert dictionary to NumPy array
        basis = np.array(list(basis.keys()))

        # Sort the basis (index of the first spin changes the slowest)
        sorted_indices = np.lexsort(tuple(basis[:, i] for i in reversed(range(basis.shape[1]))))
        basis = basis[sorted_indices]

        # Create a dictionary of the basis set for fast searching
        self.dict = {tuple(row): idx for idx, row in enumerate(basis)}
        
        # Assign to instance variable
        self.arr = basis

    def make_subsystem_basis(self, spin_system: SpinSystem, subsystem: tuple) -> Iterator:
        """
        Generates the basis set for a given subsystem.

        Parameters
        ----------
        spin_system : SpinSystem
            The spin system containing the subsystem.
        subsystem : tuple
            Indices of the spins involved in the subsystem.

        Returns
        -------
        basis : Iterator
            An iterator over the basis set for the given subsystem, represented as tuples.

            For example, identity operator and z-operator for the 3rd spin:
            [(0, 0, 0), (0, 0, 2), ...]
        """

        # Extract the necessary information from the spin system
        size = spin_system.size
        mults = spin_system.mults

        # Define all possible spin operators for each spin
        operators = []

        # Loop through every spin in the full system
        for spin in range(size):

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
    
def state_idx(spin_system: SpinSystem, op_def: tuple) -> int:
    """
    Finds the index corresponding to the given operator definition.

    Parameters
    ----------
    spin_system : SpinSystem
        The spin system containing the basis.
    op_def : tuple
        Tuple of integers that describes the operator of interest.

    Returns
    -------
    idx : int
        Index of the given state in the basis set.
    """

    # Get the index
    idx = spin_system.basis.dict[op_def]

    return idx

# def ZQ_basis(spin_system: SpinSystem) -> list:
#     """
#     Modifies the existing basis by retaining only the zero-quantum (ZQ) terms.

#     Assigns the `ZQ_map` variable for the basis, which contains the index
#     mapping from the original basis to the zero-quantum basis.

#     Parameters
#     ----------
#     spin_system : SpinSystem
#         The spin system for which the zero-quantum basis is being created.

#     Returns
#     -------
#     ZQ_map : list
#         Index mapping from the original basis set to the modified basis set. This index map
#         is used to convert the operators to the modified basis.
#     """

#     print("Constructing the zero-quantum basis.")
#     time_start = time.time()

#     # Create an empty dictionary for the new basis
#     basis_dict = {}

#     # Create an empty list for the mapping from old to new basis
#     ZQ_map = []

#     # Iterate over the basis
#     i = 0
#     for state, idx in spin_system.basis.dict.items():

#         # Check if coherence order is zero
#         if coherence_order(state) == 0:

#             # Assign state to the ZQ basis and increment index
#             basis_dict[state] = i
#             i += 1

#             # Assign index to the ZQ map
#             ZQ_map.append(idx)

#     # Convert basis to NumPy array
#     basis = np.array(list(basis_dict.keys()))

#     # Save the basis
#     spin_system.basis.arr = basis
#     spin_system.basis.dict = basis_dict

#     print("Zero-quantum basis created.")
#     print(f"Elapsed time: {time.time() - time_start:.4f} seconds.") # NOTE: Perttu's edit
#     print()

#     return ZQ_map

# NOTE: Perttu's edit
def ZQ_basis_map(spin_system: SpinSystem) -> list:
    """
    Generates the 'ZQ_basis_map' variable for the basis, which contains the index
    mapping from the original basis to the zero-quantum basis.

    Parameters
    ----------
    spin_system : SpinSystem
        The spin system for which the zero-quantum basis is being created.

    Returns
    -------
    ZQ_basis_map : list
        Index mapping from the original basis set to the modified basis set. This index map
        is used to convert the operators to the modified basis.
    """

    print("Constructing the zero-quantum basis.")
    time_start = time.time()

    # Create an empty dictionary for the new basis
    basis_dict = {}

    # Create an empty list for the mapping from old to new basis
    ZQ_basis_map = []

    # Iterate over the basis
    i = 0
    for state, idx in spin_system.basis.dict.items():

        # Check if coherence order is zero
        if coherence_order(state) == 0:

            # Assign state to the ZQ basis and increment index
            basis_dict[state] = i
            i += 1

            # Assign index to the ZQ map
            ZQ_basis_map.append(idx)

    # Convert basis to NumPy array
    basis = np.array(list(basis_dict.keys()))

    # Save the basis
    spin_system.basis.arr = basis
    spin_system.basis.dict = basis_dict

    print("Zero-quantum basis created.")
    print(f"Elapsed time: {time.time() - time_start:.4f} seconds.") # NOTE: Perttu's edit
    print()

    return ZQ_basis_map

def ZQ_filter(spin_system: SpinSystem, A: csc_array | np.ndarray, ZQ_map: list) -> csc_array | np.ndarray:
    """
    Filters a superoperator or state vector to retain only the zero-quantum coherence (ZQC) terms.
    The zero-quantum basis must have been created prior to calling this function.

    Parameters
    ----------
    spin_system : SpinSystem
        The spin system containing the basis.
    A : csc_array or numpy.ndarray
        Any superoperator or state vector written in the original basis.
    ZQ_map : list
        Index mapping from the original basis to the zero-quantum basis.

    Returns
    -------
    A : csc_array or numpy.ndarray
        The filtered operator or state vector with only ZQC terms retained.
    """

    print("Applying zero-quantum coherence filter.")
    time_start = time.time()

    # Process density vectors
    if _la.isvector(A):

        # Apply the filter
        A = A[ZQ_map]

    # Process superoperators
    else:

        # Apply the filter
        A = A[np.ix_(ZQ_map, ZQ_map)]

    print("Zero-quantum coherence filter applied.")
    print(f"Elapsed time: {time.time() - time_start:.4f} seconds.") # NOTE: Perttu's edit
    print()

    return A
    
def lq_to_idx(l: int, q: int) -> int:
    """
    Returns the index of a single-spin irreducible spherical tensor operator
    determined by rank l and projection q.

    Parameters
    ----------
    l : int
        Operator rank.
    q : int
        Operator projection.

    Returns
    -------
    idx : int
        Index of the operator.
    """

    # Get the operator index
    idx = l**2 + l - q

    return idx

def idx_to_lq(idx: int) -> Tuple[int, int]:
    """
    Converts the given operator index to rank l and projection q.

    Parameters
    ----------
    idx : int
        Index that describes the irreducible spherical tensor.

    Returns
    -------
    l : int
        Operator rank.
    q : int
        Operator projection.
    """

    # Calculate l
    l = math.ceil(-1 + math.sqrt(1 + idx))

    # Calculate q
    q = l**2 + l - idx
    
    return l, q

def coherence_order(op_def: tuple) -> int:
    """
    Determines the coherence order of a given operator defined by `op_def`.

    Parameters
    ----------
    op_def : tuple
        Contains the indices that describe the product operator.

    Returns
    -------
    order : int
        Coherence order of the operator.
    """

    # Initialize the coherence order
    order = 0

    # Iterate over the product operator and sum the q values together
    for op in op_def:
        _, q = idx_to_lq(op)
        order += q

    return order

# Assign each operator string to the corresponding integer
op_def_table = {
    'E': ([0], [1]),
    'I_+': ([1], [-np.sqrt(2)]),
    'I_z': ([2], [1]),
    'I_-': ([3], [np.sqrt(2)]),
    'I_x': ([1, 3], [-np.sqrt(2)/2, np.sqrt(2)/2]),
    'I_y': ([1, 3], [-np.sqrt(2)/(2j), -np.sqrt(2)/(2j)])
}

# Assign spherical tensors to the table
# NOTE: Hardcoded for up to l=9
for l in range(10):
    for q in range(-l, l + 1):
        op = f"T_{l}{q}"
        idx = lq_to_idx(l, q)
        op_def_table[op] = ([idx], [1])

def str_to_op_def(spin_system: SpinSystem, operators: list, spins: list) -> Tuple[list, list]:
    """
    Converts a product operator described by lists of strings and spin indices to the
    tuple(s) of integers that define the operator.

    Parameters
    ----------
    spin_system : SpinSystem
        The spin system containing the basis.
    operators : list
        List of operators that describe the product operator. Operators that are not
        specified will be treated as unit operators.
        Example: ['I_z', 'I_+']
    spins : list
        Indices of the spins. Must match the length of `operators`.
        Example: [0, 2]

    Returns
    -------
    op_defs : list of tuples
        A list that contains tuples, which describe the requested operator with integers.
        Example: [(2, 0, 1)]
    coeffs : list of floats
        Coefficients that account for the different norms of operator relations.
    """

    # Get the size of the operator definitions
    size = spin_system.size

    # Fill the rest of the operator array with unit operators
    operators_full = np.array(['E' for _ in range(size)], dtype='<U5')
    for op, spin in zip(operators, spins):
        operators_full[spin] = op

    # Create empty lists of lists to hold the op_defs and the coefficients
    op_defs = [[]]
    coeffs = [[]]

    # Iterate over all of the operator strings
    for op in operators_full:

        # Get the corresponding integers and coefficients
        op_ints, op_coeffs = op_def_table[op]

        # Add each possible value
        op_defs = [op_def + [op_int] for op_def in op_defs for op_int in op_ints]
        coeffs = [coeff + [op_coeff] for coeff in coeffs for op_coeff in op_coeffs]

    # Convert the operator definition to tuple
    op_defs = [tuple(op_def) for op_def in op_defs]

    # Calculate the coefficients
    coeffs = [np.prod(coeff) for coeff in coeffs]

    return op_defs, coeffs