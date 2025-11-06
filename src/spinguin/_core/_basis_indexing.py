"""
This module contains functions related to the indexing of states in the basis
set.
"""

# Imports
import math
import numpy as np

def lq_to_idx(l: int, q: int) -> int:
    """
    Spherical tensor operators are indexed using integers 0, 1, 2, etc. This
    function returns the index of a single-spin irreducible spherical tensor
    operator that is determined by rank `l` and projection `q`.

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
    Spherical tensor operators are indexed using integers 0, 1, 2, etc. This
    function converts the given index `idx` to the rank `l` and projection `q`
    of the spherical tensor operator.

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
    which is defined by an array of integers `op_def`.

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
    Basis set operators are defined by an array of integers where each integer
    represents a spherical tensor operator. This function finds out the spin
    order of a given basis set operator defined by `op_def`.

    Parameters
    ----------
    op_def : ndarray
        Contains the indices that describe the product operator.

    Returns
    -------
    order : int
        Spin order of the operator.
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