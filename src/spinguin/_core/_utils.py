"""
This module contains utility functions that are shared across several modules.
"""
# Imports
import math
import numpy as np
import re

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