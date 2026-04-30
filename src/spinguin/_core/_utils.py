"""
Shared utility helpers for tensor indices, operator parsing, basis analysis,
and other small tasks.

This module provides small helper functions for irreducible-tensor index
conversions, operator-string parsing, and simple basis-set analysis.
"""

import math
import re

import numpy as np


def _extract_arguments(operator_term: str) -> list[str]:
    """
    Extract the comma-separated arguments of an operator term.

    Parameters
    ----------
    operator_term : str
        Operator term such as ``I(z,0)`` or ``T(1,-1,2)``.

    Returns
    -------
    list of str
        Arguments extracted from inside the parentheses.

    Raises
    ------
    ValueError
        Raised if the operator term does not contain a valid argument list.
    """

    # Extract the comma-separated arguments enclosed in parentheses.
    match = re.search(r'\(([^)]*)\)', operator_term)
    if match is None:
        raise ValueError(
            f"Cannot parse the following invalid operator: {operator_term}."
        )

    return match.group(1).split(',')


def _split_sum_terms(operator: str) -> list[str]:
    """
    Split an operator string into top-level additive terms.

    Parameters
    ----------
    operator : str
        Operator string without whitespace.

    Returns
    -------
    list of str
        Additive terms of the operator string.
    """

    # Track the additive terms and the current parenthesis depth.
    prod_ops = []
    parenthesis_depth = 0
    start = 0

    # Split only at plus signs that are outside parentheses.
    for i, char in enumerate(operator):
        if char == '(':
            parenthesis_depth += 1
        elif char == ')':
            parenthesis_depth -= 1
        elif char == '+' and parenthesis_depth == 0:
            prod_ops.append(operator[start:i])
            start = i + 1

    # Append the final term after the last split point.
    prod_ops.append(operator[start:])

    return prod_ops


def _expand_global_operator(prod_op: str, nspins: int) -> list[str]:
    """
    Expand single-term shorthand operators that apply to all spins.

    Parameters
    ----------
    prod_op : str
        One additive operator term.
    nspins : int
        Number of spins in the system.

    Returns
    -------
    list of str
        Expanded operator terms.
    """

    # Keep product operators unchanged because they are already explicit.
    if '*' in prod_op:
        return [prod_op]

    # Leave the unit operator unchanged.
    if prod_op[0] == 'E':
        return [prod_op]

    # Expand Cartesian and ladder operators that omit the spin index.
    if prod_op[0] == 'I':
        component = _extract_arguments(prod_op)
        if len(component) == 1:
            return [f"I({component[0]},{index})" for index in range(nspins)]
        return [prod_op]

    # Expand spherical tensor operators that omit the spin index.
    if prod_op[0] == 'T':
        component = _extract_arguments(prod_op)
        if len(component) == 2:
            l = component[0]
            q = component[1]
            return [f"T({l},{q},{index})" for index in range(nspins)]
        return [prod_op]

    # Reject unsupported operator expressions explicitly.
    raise ValueError(
        f"Cannot parse the following invalid operator: {prod_op}."
    )


def _operator_component_to_terms(
    operator_component: str,
) -> tuple[list[int], list[complex]]:
    """
    Convert a single-spin operator label to basis indices and coefficients.

    Parameters
    ----------
    operator_component : str
        Single-spin operator label such as ``E``, ``I_x``, or ``T_1_0``.

    Returns
    -------
    op_ints : list of int
        Basis indices corresponding to the operator component.
    op_coeffs : list of complex
        Coefficients associated with the basis expansion.
    """

    # Map the operator component to basis indices and coefficients.
    match operator_component:
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
            op_coeffs = [-np.sqrt(2) / 2, np.sqrt(2) / 2]
        case 'I_y':
            op_ints = [1, 3]
            op_coeffs = [-np.sqrt(2) / (2j), -np.sqrt(2) / (2j)]
        case _:
            l_str, q_str = operator_component.split('_')[1:3]
            idx = lq_to_idx(int(l_str), int(q_str))
            op_ints = [idx]
            op_coeffs = [1]

    return op_ints, op_coeffs


def coherence_order(op_def: np.ndarray) -> int:
    """
    Determine the coherence order of a product operator.

    Parameters
    ----------
    op_def : ndarray
        Contains the indices that describe the product operator.

    Returns
    -------
    order : int
        Coherence order of the operator.
    """

    # Initialise the total coherence order.
    order = 0

    # Sum the tensor projections of all single-spin factors.
    for op in op_def:
        _, q = idx_to_lq(op)
        order += q

    return order


def idx_to_lq(idx: int) -> tuple[int, int]:
    """
    Convert a tensor index to rank ``l`` and projection ``q``.

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

    # Determine the tensor rank from the quadratic index relation.
    l = math.ceil(-1 + math.sqrt(1 + idx))

    # Determine the tensor projection from the index definition.
    q = l ** 2 + l - idx

    return l, q


def lq_to_idx(l: int, q: int) -> int:
    """
    Return the index of a single-spin irreducible spherical tensor operator.

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

    # Convert the tensor rank and projection to the linear index.
    idx = l ** 2 + l - q

    return idx


def parse_operator_string(
    operator: str,
    nspins: int,
) -> tuple[list[np.ndarray], list[complex]]:
    """
    Parse an operator string into basis definitions and coefficients.

    Parameters
    ----------
    operator : str
        The operator string must follow the rules below:

        - Cartesian or ladder operator at specific index or for all spins::

            operator = "I(component, index)"
            operator = "I(component)"

        - Spherical tensor operator at specific index or for all spins::

            operator = "T(l, q, index)"
            operator = "T(l, q)"

        - Product operators::

            operator = "I(component1, index1) * I(component2, index2)"

        - Sum of operators::

            operator = "I(component1, index1) + I(component2, index2)"

        - Unit operators are ignored in the input. These are identical::

            operator = "E * I(component, index)"
            operator = "I(component, index)"

        Special case: An empty ``operator`` string is considered as the unit
        operator.

        Whitespace is ignored in the input.

        Note that indexing starts from 0.
    nspins : int
        Number of spins in the system.

    Returns
    -------
    op_defs : list of ndarray
        Arrays that describe the requested operator with integers. Example:
        ``[[2, 0, 1]]`` corresponds to ``T_1_0 * E * T_1_1``.
    coeffs : list of complex
        Coefficients that account for the different norms of operator
        relations.
    """

    # Initialise the lists of operator definitions and coefficients.
    op_defs = []
    coeffs = []

    # Remove all whitespace from the input string.
    operator = "".join(operator.split())

    # Return the unit operator when the input string is empty.
    if operator == "":
        op_def = np.array([0 for _ in range(nspins)])
        coeff = 1
        op_defs.append(op_def)
        coeffs.append(coeff)
        return op_defs, coeffs

    # Split the additive (+) terms.
    prod_ops = _split_sum_terms(operator)

    # Expand shorthand single-spin notation.
    prod_ops = [
        expanded_op
        for prod_op in prod_ops
        for expanded_op in _expand_global_operator(prod_op, nspins)
    ]

    # Process each product operator separately.
    for prod_op in prod_ops:

        # Start from the all-unit product operator.
        op = np.array(["E" for _ in range(nspins)], dtype="<U10")

        # Split the product into single-spin operator terms.
        op_terms = prod_op.split('*')

        # Parse and place each single-spin operator term.
        for op_term in op_terms:

            # Keep explicit unit operators unchanged.
            if op_term[0] == 'E':
                pass

            # Parse Cartesian and ladder operators.
            elif op_term[0] == 'I':
                component_and_index = _extract_arguments(op_term)
                if len(component_and_index) != 2:
                    raise ValueError(
                        f"Cannot parse the following invalid operator: "
                        f"{op_term}."
                    )
                component = component_and_index[0]
                index = int(component_and_index[1])
                op[index] = f"I_{component}"

            # Parse spherical tensor operators.
            elif op_term[0] == 'T':
                component_and_index = _extract_arguments(op_term)
                if len(component_and_index) != 3:
                    raise ValueError(
                        f"Cannot parse the following invalid operator: "
                        f"{op_term}."
                    )
                l = component_and_index[0]
                q = component_and_index[1]
                index = int(component_and_index[2])
                op[index] = f"T_{l}_{q}"

            # Reject unsupported operator terms explicitly.
            else:
                raise ValueError(
                    f"Cannot parse the following invalid operator: {op_term}."
                )

        # Initialise the current basis-expansion lists.
        op_defs_curr = [[]]
        coeffs_curr = [[]]

        # Expand each single-spin factor into basis indices and coefficients.
        for operator_component in op:
            op_ints, op_coeffs = _operator_component_to_terms(
                operator_component
            )

            # Append all allowed basis choices for the current spin factor.
            op_defs_curr = [
                op_def + [op_int]
                for op_def in op_defs_curr
                for op_int in op_ints
            ]
            coeffs_curr = [
                coeff + [op_coeff]
                for coeff in coeffs_curr
                for op_coeff in op_coeffs
            ]

        # Convert the operator definitions to NumPy arrays.
        op_defs_curr = [np.array(op_def) for op_def in op_defs_curr]

        # Multiply the single-spin coefficients into total coefficients.
        coeffs_curr = [np.prod(coeff) for coeff in coeffs_curr]

        # Append the current contributions to the total expansion.
        op_defs.extend(op_defs_curr)
        coeffs.extend(coeffs_curr)

    return op_defs, coeffs


def spin_order(op_def: np.ndarray) -> int:
    """
    Determine the spin order (number of non-unit single-spin operators)
    of a product operator.

    Parameters
    ----------
    op_def : ndarray
        Contains the indices that describe the product operator.

    Returns
    -------
    order : int
        Spin order of the operator.
    """

    # Count the number of non-unit single-spin factors.
    return np.count_nonzero(op_def)