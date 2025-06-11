"""
basis.py

This module provides functionality for constructing a basis set.
"""

# Imports
import numpy as np
import time
import re
import math
from itertools import product, combinations
from typing import Iterator
        
def make_basis(spins: np.ndarray, max_spin_order: int):
    """
    Constructs a Liouville-space basis set, where the basis is spanned by all
    possible Kronecker products of irreducible spherical tensor operators, up
    to the defined maximum spin order.

    The Kronecker products themselves are not calculated. Instead, the operators
    are expressed as sequences of integers, where each integer represents a
    spherical tensor operator of rank `l` and projection `q` using the following
    relation: `N = l^2 + l - q`. The indexing scheme has been adapted from:

    Hogben, H. J., Hore, P. J., & Kuprov, I. (2010)
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
    
def truncate_basis_by_coherence(basis: np.ndarray,
                                coherence_orders: list) -> list:
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
    print(f"Elapsed time: {time.time() - time_start:.4f} seconds.")
    print()

    return truncated_basis, index_map

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

def idx_to_lq(idx: int) -> tuple[int, int]:
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

def coherence_order(op_def: np.ndarray) -> int:
    """
    Determines the coherence order of a given operator defined by `op_def`.

    Parameters
    ----------
    op_def : ndarray
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

def spin_order(op_def: np.ndarray) -> int:
    """
    Finds out the spin order of a given operator defined by `op_def`.

    Parameters
    ----------
    op_def : ndarray
        Contains the indices that describe the product operator.

    Returns
    -------
    order : int
        Spin order of the operator
    """
    # Spin order is equal to the number of non-zeros
    order = np.count_nonzero(op_def)

    return order

def parse_operator_string(operator: str, nspins: int):
    """
    Parses operator strings and returns their definitions in the basis set as
    well as their corresponding coefficients. The operator string must
    follow the rules below:

    - Cartesian and ladder operators: `I(component,index)` or
      `I(component)`. Examples:

        - `I(x,4)` --> Creates x-operator for spin at index 4.
        - `I(x)`--> Creates x-operator for all spins.

    - Spherical tensor operators: `T(l,q,index)` or `T(l,q)`. Examples:

        - `T(1,-1,3)` --> \
          Creates operator with `l=1`, `q=-1` for spin at index 3.
        - `T(1, -1)` --> \
          Creates operator with `l=1`, `q=-1` for all spins.
        
    - Product operators have `*` in between the single-spin operators:
      `I(z,0) * I(z,1)`
    - Sums of operators have `+` in between the operators:
      `I(x,0) + I(x,1)`
    - Unit operators are ignored in the input. Interpretation of these
      two is identical: `E * I(z,1)`, `I(z,1)`
    
    Special case: An empty `operator` string is considered as unit operator.

    Whitespace will be ignored in the input.

    NOTE: Indexing starts from 0!

    Parameters
    ----------
    operator : str
        String that defines the operator to be generated.
    nspins : int
        Number of spins in the system.

    Returns
    -------
    op_defs : list of ndarray
        A list that contains arrays, which describe the requested operator with
        integers. Example: `[[2, 0, 1]]` --> `T_1_0 * E * T_1_1`
    coeffs : list of floats
        Coefficients that account for the different norms of operator relations.
    """

    # Create empty lists to hold the operator definitions and the coefficients
    op_defs = []
    coeffs = []

    # Remove spaces from the user input
    operator = "".join(operator.split())

    # Create unit operator if input string is empty
    if operator == "":
        op_def = np.array([0 for _ in range(nspins)])
        coeff = 1
        op_defs.append(op_def)
        coeffs.append(coeff)
        return op_defs, coeffs

    # Split the user input sum '+' into separate product operators
    prod_ops = []
    inside_parantheses = False
    start = 0
    for i, char in enumerate(operator):
        if char == '(':
            inside_parantheses = True
        elif char == ')':
            inside_parantheses = False
        elif char == '+' and not inside_parantheses:
            prod_ops.append(operator[start:i])
            start = i + 1
    prod_ops.append(operator[start:])

    # Replace inputs of kind I(z) --> Sum operator for all spins
    prod_ops_copy = []
    for prod_op in prod_ops:
        if '*' not in prod_op:

            # For unit operators, do nothing
            if prod_op[0] == 'E':
                prod_ops_copy.append(prod_op)

            # Handle Cartesian and ladder operators
            elif prod_op[0] == 'I':
                component = re.search(r'\(([^)]*)\)',
                                      prod_op).group(1).split(',')
                if len(component) == 1:
                    component = component[0]
                    for index in range(nspins):
                        prod_ops_copy.append(f"I({component},{index})")
                else:
                    prod_ops_copy.append(prod_op)

            # Handle spherical tensor operators
            elif prod_op[0] == 'T':
                component = re.search(r'\(([^)]*)\)',
                                      prod_op).group(1).split(',')
                if len(component) == 2:
                    l = component[0]
                    q = component[1]
                    for index in range(nspins):
                        prod_ops_copy.append(f"T({l},{q},{index})")
                else:
                    prod_ops_copy.append(prod_op)

            # Otherwise an unsupported operator
            else:
                raise ValueError("Cannot parse the following invalid"
                                 f"operator: {op_term}")

        # Keep operator as is, if the input contains '*'
        else:
            prod_ops_copy.append(prod_op)

    prod_ops = prod_ops_copy
                
    # Process each product operator separately
    for prod_op in prod_ops:

        # Start from a unit operator
        op = np.array(['E' for _ in range(nspins)], dtype='<U10')

        # Separate the terms in the product operator
        op_terms = prod_op.split('*')

        # Process each term separately
        for op_term in op_terms:

            # Handle unit operators (by default exist in the operator)
            if op_term[0] == 'E':
                pass

            # Handle Cartesian and ladder operators
            elif op_term[0] == 'I':
                component_and_index = re.search(r'\(([^)]*)\)',
                                                op_term).group(1).split(',')
                component = component_and_index[0]
                index = int(component_and_index[1])
                op[index] = f"I_{component}"

            # Handle spherical tensor operators
            elif op_term[0] == 'T':
                component_and_index = re.search(r'\(([^)]*)\)',
                                                op_term).group(1).split(',')
                l = component_and_index[0]
                q = component_and_index[1]
                index = int(component_and_index[2])
                op[index] = f"T_{l}_{q}"

            # Other input types are not supported
            else:
                raise ValueError("Cannot parse the following invalid"
                                 f"operator: {op_term}")

        # Create empty lists of lists to hold the current operator definitions
        # and coefficients
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
            op_defs_curr = [op_def + [op_int] for op_def in op_defs_curr
                            for op_int in op_ints]
            coeffs_curr = [coeff + [op_coeff] for coeff in coeffs_curr
                           for op_coeff in op_coeffs]

        # Convert the operator definition to NumPy
        op_defs_curr = [np.array(op_def) for op_def in op_defs_curr]

        # Calculate the coefficients
        coeffs_curr = [np.prod(coeff) for coeff in coeffs_curr]

        # Extend the total lists
        op_defs.extend(op_defs_curr)
        coeffs.extend(coeffs_curr)

    return op_defs, coeffs

def state_idx(basis: np.ndarray, op_def: np.ndarray) -> int:
    """
    Finds the index corresponding to the given operator definition.

    Parameters
    ----------
    basis : ndarray
        Two dimensional array containing the basis set that consists of rows of
        integers defining the products of irreducible spherical tensors.
    op_def : ndarray
        A one-dimensional array of integers that describes the operator of
        interest.

    Returns
    -------
    idx : int
        Index of the given state in the basis set.
    """

    # Check that the dimensions match
    if not basis.shape[1] == op_def.shape[0]:
        raise ValueError("Cannot find the index of state, as the dimensions do "
                         f"not match. 'basis': {basis.shape[1]}, "
                         f"'op_def': {op_def.shape[0]}")

    # Search for the state
    is_equal = np.all(basis == op_def, axis=1)
    idx = np.where(is_equal)[0]

    # Confirm that exactly one state was found
    if idx.shape[0] == 1:
        idx = idx[0]
    elif idx.shape[0] == 0:
        raise ValueError(f"Could not find the index of state: {op_def}.")
    else:
        raise ValueError("Multiple states in the basis match with the "
                         f"requested state: {op_def}")
    
    return idx