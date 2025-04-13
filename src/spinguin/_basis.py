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
import re
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

def truncate_basis_by_coherence(spin_system: SpinSystem, coherence_orders: list) -> list:
    """
    Truncates the basis set of the `SpinSystem` object by retaining only the product operators
    in the basis that correspond to coherence orders specified in the `coherence_orders` list.
    The function generates also an index map from the original basis to the truncated basis.
    This map can be used to transform superoperators or state vectors to the new basis by
    using the `project_to_truncated_basis()` function.

    Parameters
    ----------
    spin_system : SpinSystem
        The spin system whose basis set will be truncated.
    coherence_orders : list
        List of coherence orders to be retained in the basis.

    Returns
    -------
    index_map : list
        List that contains an index map from the original basis to the truncated basis.

    TODO: Funktion ja input parametrien nimeäminen? Onko Pertulla ideoita?
    """

    print(f"Truncating the basis set. The following coherence orders are retained: {coherence_orders}")
    time_start = time.time()

    # Create an empty dictionary for the new basis
    basis_dict = {}

    # Create an empty list for the mapping from old to new basis
    index_map = []

    # Iterate over the basis
    i = 0
    for state, idx in spin_system.basis.dict.items():

        # Check if coherence order is in the list
        if coherence_order(state) in coherence_orders:

            # Assign state to the truncated basis and increment index
            basis_dict[state] = i
            i += 1

            # Assign index to the index map
            index_map.append(idx)

    # Convert basis to NumPy array
    basis = np.array(list(basis_dict.keys()))

    # Save the basis
    spin_system.basis.arr = basis
    spin_system.basis.dict = basis_dict

    print("Truncated basis created.")
    print(f"Elapsed time: {time.time() - time_start:.4f} seconds.")
    print()

    return index_map

def transform_to_truncated_basis(index_map: list, *objs: csc_array | np.ndarray) -> csc_array | np.ndarray | Tuple[csc_array, ...] | Tuple[np.ndarray, ...]:
    """
    Transforms superoperators or state vectors to a truncated basis using the `index_map`, which
    is obtained from `truncate_basis_by_coherence()` function. Multiple objects can be transformed
    simultaneously.

    Parameters
    ----------
    index_map : list
        Index mapping from the original basis to the truncated basis.
    objs : csc_array or numpy.ndarray
        Superoperators or state vectors written in the original basis.

    Returns
    -------
    transformed_objs : csc_array or numpy.ndarray
        Superoperators or state vectors transformed into the truncated basis.
    TODO: Funktion ja input parametrien nimeäminen? Onko Pertulla ideoita?
    """

    print("Transforming superoperators or state vectors into a truncated basis.")
    time_start = time.time()

    # Empty list for transformed objects
    transformed_objs = []

    # Perform the transformation to each given superoperator or state vector
    for obj in objs:

        # Process state vectors
        if _la.isvector(obj):
            transformed_objs.append(obj[index_map])

        # Process superoperators
        else:
            transformed_objs.append(obj[np.ix_(index_map, index_map)])

    # Do not return a tuple, if only one state vector or superoperator is transformed
    if len(transformed_objs) == 1:
        transformed_objs = transformed_objs[0]

    print("Transformation completed.")
    print(f"Elapsed time: {time.time() - time_start:.4f} seconds.")
    print()

    return transformed_objs
    
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

def parse_operator_string(spin_system: SpinSystem, operator: str):
    """
    Parses operator strings and returns their definitions in the basis set as well as their corresponding coefficients.
    The operator string must follow the rules below:

    - Cartesian and ladder operators: I(component,index). Example: I(x,4) --> Creates x-operator for spin at index 4.
    - Spherical tensor operators: T(l,q,index). Example: T(1,-1,3) --> Creates operator with l=1, q=-1 for spin at index 3.
    - Product operators have `*` in between the single-spin operators: I(z,0) * I(z,1)
    - Sums of operators have `+` in between the operators: I(x,0) + I(x,1)
    - The unit operator is not typed. Example: I(z,2) will generate E*I_z in case of a two-spin system. 

    Whitespace will be ignored in the input.

    Parameters
    ----------
    spin_system : SpinSystem
        The spin system for which the operator is going to be generated.
    operator : str
        String that defines the operator to be generated.

    Returns
    -------
    op_defs : list of tuples
        A list that contains tuples, which describe the requested operator with integers.
        Example: [(2, 0, 1)] --> T_1_0 * E * T_1_1
    coeffs : list of floats
        Coefficients that account for the different norms of operator relations.
    """

    # Create empty lists of lists to hold the operator definitions and the coefficients
    op_defs = []
    coeffs = []

    # Remove spaces from the user input
    operator = "".join(operator.split())

    # Split the user input into separate product operators
    prod_ops = re.split(r'(?<=\))\+', operator)

    # Process each product operator separately
    for prod_op in prod_ops:

        # Start from a unit operator
        op = np.array(['E' for _ in range(spin_system.size)], dtype='<U5')

        # Separate the terms in the product operator
        op_terms = prod_op.split('*')

        # Process each term separately
        for op_term in op_terms:

            # Obtain the component and index
            component_and_index = re.search(r'\(([^)]*)\)', op_term).group(1).split(',')

            # Handle Cartesian and ladder operators
            if op_term[0] == 'I':
                component = component_and_index[0]
                index = int(component_and_index[1])
                op[index] = f"I_{component}"

            # Handle spherical tensor operators
            elif op_term[0] == 'T':
                l = component_and_index[0]
                q = component_and_index[1]
                index = int(component_and_index[2])
                op[index] = f"T_{l}_{q}"

            # Other input types are not supported
            else:
                raise ValueError(f"Cannot parse the following invalid operator: {op_term}")

        # Create empty lists of lists to hold the current operator definitions and coefficients
        op_defs_curr = [[]]
        coeffs_curr = [[]]

        # Iterate over all of the operator strings
        for o in op:

            # Get the corresponding integers and coefficients
            match o:

                case 'E':
                    op_ints = [0]
                    op_coeffs = [1]

                case 'I_+':
                    op_ints = [1]
                    op_coeffs = [-np.sqrt(2)]

                case 'I_z':
                    op_ints = [2]
                    op_coeffs = [1]

                case 'I_-':
                    op_ints = [3]
                    op_coeffs = [np.sqrt(2)]

                case 'I_x':
                    op_ints = [1, 3]
                    op_coeffs = [-np.sqrt(2)/2, np.sqrt(2)/2]

                case 'I_y':
                    op_ints = [1, 3]
                    op_coeffs = [-np.sqrt(2)/(2j), -np.sqrt(2)/(2j)]

                # Default case handles spherical tensors
                case _:
                    o = o.split('_')
                    l = int(o[1])
                    q = int(o[2])
                    idx = lq_to_idx(l, q)
                    op_ints = [idx]
                    op_coeffs = [1]

            # Add each possible value
            op_defs_curr = [op_def + [op_int] for op_def in op_defs_curr for op_int in op_ints]
            coeffs_curr = [coeff + [op_coeff] for coeff in coeffs_curr for op_coeff in op_coeffs]

        # Convert the operator definition to tuple
        op_defs_curr = [tuple(op_def) for op_def in op_defs_curr]

        # Calculate the coefficients
        coeffs_curr = [np.prod(coeff) for coeff in coeffs_curr]

        # Extend the total lists
        op_defs.extend(op_defs_curr)
        coeffs.extend(coeffs_curr)

    return op_defs, coeffs