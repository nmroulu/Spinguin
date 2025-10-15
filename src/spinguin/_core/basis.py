"""
This module provides functionality for constructing a basis set.
"""

# Imports
import numpy as np
import scipy.sparse as sp
import time
import re
import math
from itertools import product, combinations
from typing import Iterator, Literal
from spinguin._core.la import eliminate_small, expm_vec
from spinguin._core.hide_prints import HidePrints
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree
        
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

def lq_to_idx(l: int, q: int) -> int:
    """
    Returns the index of a single-spin irreducible spherical tensor operator
    determined by rank `l` and projection `q`.

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
    Converts the given operator index to rank `l` and projection `q`.

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
    Determines the coherence order of a given product operator in the basis set,
    defined by an array of integers `op_def`.

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

def _is_valid_single_spin_operator(operator: str, nspins: int):
    """
    Checks that the given operator string is a valid single-spin operator. For
    example, `operator = 'I(z,0)` is valid, whereas `I(z,y)` is invalid.

    Parameters
    ----------
    operator : str
        A string that defines the single-spin operator.
    nspins : int
        Number of spins in the system.

    Returns
    -------
    valid : bool
        True if the operator is valid, False otherwise.
    """
    # Empty strings are always invalid
    if len(operator) == 0:
        return False
    
    # Handle unit operators (must be of length 1)
    elif operator[0] == 'E' and len(operator) == 1:
        return True
        
    # Handle Cartesian, raising, and lowering operators
    elif (
        operator[0] == 'I' and
        operator.count('(') == 1 and
        operator.count(')') == 1 and
        operator.count(',') == 1 and
        operator[1] == '(' and
        operator[-1] == ')'
    ):

        # Find the operator component and index
        component = operator[2:-1].split(',')[0]
        index = operator[2:-1].split(',')[1]

        # Check the component and index
        if (
            component in {'x', 'y', 'z', '+', '-'} and
            index.isdigit() and
            int(index) < nspins
        ):
            return True
        else:
            return False
    
    # Handle spherical tensor operators
    elif (
        operator[0] == 'T' and
        operator.count('(') == 1 and
        operator.count(')') == 1 and
        operator.count(',') == 2 and
        operator[1] == '(' and
        operator[-1] == ')'
    ):
        # Find rank, projection and index
        l = operator[2:-1].split(',')[0]
        q = operator[2:-1].split(',')[1]
        index = operator[2:-1].split(',')[2]

        # Check that they all are integers
        try:
            l = int(l)
            q = int(q)
            index = int(index)
        except ValueError:
            return False
        
        # Check the rank, projection and index
        if (
            l >= 0 and
            l >= abs(q) and
            index < nspins
        ):
            return True
        else:
            return False
        
    # Otherwise not a valid operator
    else:
        return False
    
def _is_valid_product_operator(operator: str, nspins: int):
    """
    Checks whether the given operator string is a valid product operator. For
    example, `operator = I(z,0)*I(z,1)` and `operator = I(z,0)` are valid.

    Parameters
    ----------
    operator : str
        A string that defines the product operator.
    nspins : int
        Number of spins in the system.

    Returns
    -------
    valid : bool
        True if the operator is valid, False otherwise.
    """
    # Split the product operator and consider each operator at time
    ops = operator.split('*')
    for op in ops:
        if not _is_valid_single_spin_operator(op, nspins):
            return False
    return True

def _is_valid_sum_operator(operator: str):
    """
    Checks that the given operator string is a valid representation of a sum
    operator. For example, `operator = I(x)` is valid, whereas
    `operator = I(x,2)` is invalid.

    Parameters
    ----------
    operator : str
        A string that defines the sum operator.
    
    Returns
    -------
    valid : bool
        True if the operator is valid, False otherwise.
    """
    # Empty strings are always invalid
    if len(operator) == 0:
        return False

    # Cartesian, raising, and lowering operators (check all possible cases)
    elif (
        operator[0] == 'I' and
        operator in {'I(x)', 'I(y)', 'I(z)', 'I(+)', 'I(-)'}
    ):
        return True

    # Spherical tensor operators
    elif (
        operator[0] == 'T' and
        operator.count('(') == 1 and
        operator.count(')') == 1 and
        operator.count(',') == 1 and
        operator[1] == '(' and
        operator[-1] == ')'
    ):
        # Obtain the rank and projection
        try:
            l = int(operator[2:-1].split(',')[0])
            q = int(operator[2:-1].split(',')[1])
        except ValueError:
            return False

        # Check the rank and projection
        if (l >= 0 and l >= abs(q)):
            return True
        else:
            return False

    # Otherwise the operator is invalid
    else:
        return False

def _split_sum_operator(operator: str):
    """
    Splits the input operator into individual operators.

    Parameters
    ----------
    operator : str
        A string that defines the operator to be generated.

    Returns
    -------
    ops : list
        A list of strings, where each string is a single operator or a product
        operator.
    """
    # Split the given product operator into single-spin operators
    ops = []
    status = "start_operator"
    for i, char in enumerate(operator):

        match status:

            case "start_operator":
                if char == 'E':
                    start = i
                    status = "product_or_sum"
                elif char == 'I' or char == 'T':
                    start = i
                    status = "leftbracket"
                else:
                    raise ValueError(f"invalid operator {char}")
                
            case "leftbracket":
                if char == '(':
                    status = "operator_details"
                else:
                    raise ValueError(f"invalid operator {operator[start:i]}")
                
            case "operator_details":
                if char == ')':
                    status = "product_or_sum"
                elif not (
                    char in {'0','1','2','3','4','5','6','7','8','9'} or
                    char in {'x', 'y', 'z', '+', '-', ','}
                ):
                    raise ValueError(f"invalid operator {operator[start:i]}")

            case "product_or_sum":
                if char == '*':
                    status = "continue_operator"
                elif char == '+':
                    ops.append(operator[start:i])
                    status = "start_operator"
                else:
                    raise ValueError(f"invalid operator {operator[start:i]}")

            case "continue_operator":
                if char == 'E':
                    status = "product_or_sum"
                elif char == 'I' or char == 'T':
                    status = "leftbracket"
                else:
                    raise ValueError(f"invalid operator {operator[start:i]}")

    if status == "product_or_sum":
        ops.append(operator[start:])
    else:
        raise ValueError(f"invalid operator {operator[start:]}")

    return ops

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

    # Create the unit operator if the input string is empty or "E"
    if operator == "" or operator == "E":
        op_def = np.array([0 for _ in range(nspins)])
        coeff = 1
        op_defs.append(op_def)
        coeffs.append(coeff)
        return op_defs, coeffs

    # Split the user input sum '+' into separate product operators
    prod_ops = _split_sum_operator(operator)

    # Ensure that each operator is valid
    prod_ops_copy = []
    for prod_op in prod_ops:

        # Replace inputs of kind I(z) --> Sum operator for all spins
        if _is_valid_sum_operator(prod_op):

            # Handle Cartesian and ladder operators
            if prod_op[0] == 'I':
                component = prod_op[2]
                for index in range(nspins):
                    prod_ops_copy.append(f"I({component},{index})")

            # Handle spherical tensor operators
            elif prod_op[0] == 'T':
                l = prod_op[2:-1].split(',')[0]
                q = prod_op[2:-1].split(',')[1]
                for index in range(nspins):
                    prod_ops_copy.append(f"T({l},{q},{index})")

        # Keep product operators as is
        elif _is_valid_product_operator(prod_op, nspins):
            prod_ops_copy.append(prod_op)

        # Raise an error for invalid operators
        else:
            raise ValueError(f"invalid operator {prod_op}")

    # Use the newly created list
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
                component = op_term[2:-1].split(',')[0]
                index = int(op_term[2:-1].split(',')[1])
                op[index] = f"I_{component}"

            # Handle spherical tensor operators
            elif op_term[0] == 'T':
                l = op_term[2:-1].split(',')[0]
                q = op_term[2:-1].split(',')[1]
                index = int(op_term[2:-1].split(',')[2])
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
    Finds the index of the state defined by the `op_def` in the basis set.

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