"""
Tests for single-spin and multi-spin operator construction.
"""

import math
import unittest

import numpy as np
import scipy.sparse as sp

import spinguin as sg
from spinguin._core._la import comm
from spinguin._core._operators import op_from_string, op_prod


class TestOperators(unittest.TestCase):
    """
    Test operator factories and operator-string parsing.
    """

    def _to_dense(
        self,
        operator,
    ):
        """
        Return a dense representation of an operator when needed.

        Parameters
        ----------
        operator : array-like or sparse matrix
            Operator to convert.

        Returns
        -------
        ndarray
            Dense representation of the operator.
        """

        # Convert sparse operators to dense arrays when necessary.
        if hasattr(operator, "toarray"):
            return operator.toarray()

        return operator

    def _assert_allclose(
        self,
        operator,
        reference,
        atol=1e-8,
    ):
        """
        Check that an operator matches its reference representation.

        Parameters
        ----------
        operator : array-like or sparse matrix
            Tested operator.
        reference : array-like or sparse matrix
            Reference operator.
        atol : float, optional
            Absolute tolerance used in the numerical comparison.

        Returns
        -------
        None
            The assertion is evaluated in place.
        """

        # Compare both operators in dense form.
        self.assertTrue(
            np.allclose(
                self._to_dense(operator),
                self._to_dense(reference),
                atol=atol,
            )
        )

    def _build_cartesian_operator_map(
        self,
        spin,
    ):
        """
        Return standard single-spin operators for one spin quantum number.

        Parameters
        ----------
        spin : float
            Spin quantum number.

        Returns
        -------
        dict
            Mapping from operator labels to their matrix representations.
        """

        # Collect the standard Cartesian and ladder operators.
        return {
            "E": sg.op_E(spin),
            "x": sg.op_Sx(spin),
            "y": sg.op_Sy(spin),
            "z": sg.op_Sz(spin),
            "+": sg.op_Sp(spin),
            "-": sg.op_Sm(spin),
        }

    def test_op_S(self):
        """
        Test spin operators against hard-coded values and commutators.

        The operators `E`, `Sx`, `Sy`, `Sz`, `Sp`, and `Sm` are compared
        against explicit matrices and then checked against the angular
        momentum commutation relations.
        """

        # Reset parameters to defaults.
        sg.parameters.default()

        # Define reference operators for selected spin quantum numbers.
        E = {
            1 / 2: np.array([[1, 0], [0, 1]]),
            1: np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
            3 / 2: np.array(
                [
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ]
            ),
        }

        Sx = {
            1 / 2: np.array([[0, 1 / 2], [1 / 2, 0]]),
            1: np.array(
                [
                    [0, 1 / math.sqrt(2), 0],
                    [1 / math.sqrt(2), 0, 1 / math.sqrt(2)],
                    [0, 1 / math.sqrt(2), 0],
                ]
            ),
            3 / 2: np.array(
                [
                    [0, math.sqrt(3) / 2, 0, 0],
                    [math.sqrt(3) / 2, 0, 1, 0],
                    [0, 1, 0, math.sqrt(3) / 2],
                    [0, 0, math.sqrt(3) / 2, 0],
                ]
            ),
        }

        Sy = {
            1/2: np.array([[0, -1j/2],
                           [1j/2, 0]]),
            1: np.array([[0, 1/(1j*math.sqrt(2)), 0],
                         [-1/(1j*math.sqrt(2)), 0, 1/(1j*math.sqrt(2))],
                         [0, -1/(1j*math.sqrt(2)), 0]]),
            3/2: np.array([[0, math.sqrt(3)/(2j), 0, 0],
                           [-math.sqrt(3)/(2j), 0, -1j, 0],
                           [0, 1j, 0, math.sqrt(3)/(2j)],
                           [0, 0, -math.sqrt(3)/(2j), 0]])
        }

        Sz = {
            1/2: np.array([[1/2, 0],
                           [0, -1/2]]),
            1: np.array([[1, 0, 0],
                         [0, 0, 0],
                         [0, 0, -1]]),
            3/2: np.array([[3/2, 0, 0, 0],
                           [0, 1/2, 0, 0],
                           [0, 0, -1/2, 0],
                           [0, 0, 0, -3/2]])
        }

        Sp = {
            1/2: np.array([[0, 1],
                           [0, 0]]),
            1: np.array([[0, math.sqrt(2), 0],
                         [0, 0, math.sqrt(2)],
                         [0, 0, 0]]),
            3/2: np.array([[0, math.sqrt(3), 0, 0],
                           [0, 0, 2, 0],
                           [0, 0, 0, math.sqrt(3)],
                           [0, 0, 0, 0]])
        }

        Sm = {
            1/2: np.array([[0, 0],
                           [1, 0]]),
            1: np.array([[0, 0, 0],
                         [math.sqrt(2), 0, 0],
                         [0, math.sqrt(2), 0]]),
            3/2: np.array([[0, 0, 0, 0],
                           [math.sqrt(3), 0, 0, 0],
                           [0, 2, 0, 0],
                           [0, 0, math.sqrt(3), 0]])
        }

        # Compare the operator values using the dense backend.
        sg.parameters.sparse_operator = False
        for spin, op in E.items():
            self._assert_allclose(sg.op_E(spin), op)
        for spin, op in Sx.items():
            self._assert_allclose(sg.op_Sx(spin), op)
        for spin, op in Sy.items():
            self._assert_allclose(sg.op_Sy(spin), op)
        for spin, op in Sz.items():
            self._assert_allclose(sg.op_Sz(spin), op)
        for spin, op in Sp.items():
            self._assert_allclose(sg.op_Sp(spin), op)
        for spin, op in Sm.items():
            self._assert_allclose(sg.op_Sm(spin), op)

        # Compare the operator values using the sparse backend.
        sg.parameters.sparse_operator = True
        for spin, op in E.items():
            self._assert_allclose(sg.op_E(spin), op)
        for spin, op in Sx.items():
            self._assert_allclose(sg.op_Sx(spin), op)
        for spin, op in Sy.items():
            self._assert_allclose(sg.op_Sy(spin), op)
        for spin, op in Sz.items():
            self._assert_allclose(sg.op_Sz(spin), op)
        for spin, op in Sp.items():
            self._assert_allclose(sg.op_Sp(spin), op)
        for spin, op in Sm.items():
            self._assert_allclose(sg.op_Sm(spin), op)

        # List the spin quantum numbers used in the commutator tests.
        spins = [1 / 2, 1, 3 / 2, 2, 5 / 2, 3, 7 / 2, 4, 9 / 2, 5, 11 / 2]

        # Test the commutation relations using the dense backend.
        sg.parameters.sparse_operator = False
        for spin in spins:

            # Check the standard angular momentum commutators.
            self._assert_allclose(comm(sg.op_E(spin), sg.op_E(spin)), 0)
            self._assert_allclose(
                comm(sg.op_Sx(spin), sg.op_Sy(spin)),
                1j * sg.op_Sz(spin),
            )
            self._assert_allclose(
                comm(sg.op_Sy(spin), sg.op_Sz(spin)),
                1j * sg.op_Sx(spin),
            )
            self._assert_allclose(
                comm(sg.op_Sz(spin), sg.op_Sx(spin)),
                1j * sg.op_Sy(spin),
            )
            self._assert_allclose(
                comm(sg.op_Sp(spin), sg.op_Sz(spin)),
                -sg.op_Sp(spin),
            )
            self._assert_allclose(
                comm(sg.op_Sm(spin), sg.op_Sz(spin)),
                sg.op_Sm(spin),
            )

        # Test the commutation relations using the sparse backend.
        sg.parameters.sparse_operator = True
        for spin in spins:

            # Check the same commutators using sparse operators.
            self._assert_allclose(comm(sg.op_E(spin), sg.op_E(spin)), 0)
            self._assert_allclose(
                comm(sg.op_Sx(spin), sg.op_Sy(spin)),
                1j * sg.op_Sz(spin),
            )
            self._assert_allclose(
                comm(sg.op_Sy(spin), sg.op_Sz(spin)),
                1j * sg.op_Sx(spin),
            )
            self._assert_allclose(
                comm(sg.op_Sz(spin), sg.op_Sx(spin)),
                1j * sg.op_Sy(spin),
            )
            self._assert_allclose(
                comm(sg.op_Sp(spin), sg.op_Sz(spin)),
                -sg.op_Sp(spin),
            )
            self._assert_allclose(
                comm(sg.op_Sm(spin), sg.op_Sz(spin)),
                sg.op_Sm(spin),
            )

    def test_op_T(self):
        """
        Test commutation relations for spherical tensor operators.
        """
        # Reset parameters to defaults.
        sg.parameters.default()

        # List the spin quantum numbers to test.
        spins = [1 / 2, 1, 3 / 2, 2, 5 / 2, 3, 7 / 2, 4, 9 / 2, 5, 11 / 2]

        # Test the commutation relations using the dense backend
        sg.parameters.sparse_operator = False
        for spin in spins:

            # Go through all the possible ranks and projections
            for l in range(0, int(2*spin+1)):
                for q in range(-l, l+1):

                    # Check the defining commutation relations.
                    self._assert_allclose(
                        comm(sg.op_Sz(spin), sg.op_T(spin, l, q)),
                        q * sg.op_T(spin, l, q),
                    )
                    self._assert_allclose(
                        comm(
                            sg.op_Sx(spin) @ sg.op_Sx(spin)
                            + sg.op_Sy(spin) @ sg.op_Sy(spin)
                            + sg.op_Sz(spin) @ sg.op_Sz(spin),
                            sg.op_T(spin, l, q),
                        ),
                        0,
                    )
                    if not q == -l:
                        self._assert_allclose(
                            comm(sg.op_Sm(spin), sg.op_T(spin, l, q)),
                            math.sqrt(l *(l + 1) - q *(q - 1))
                            * sg.op_T(spin, l, q - 1),
                        )
                    if not q == l:
                        self._assert_allclose(
                            comm(sg.op_Sp(spin), sg.op_T(spin, l, q)),
                            math.sqrt(l *(l + 1) - q *(q + 1))
                            * sg.op_T(spin, l, q + 1),
                        )

        # Test the commutation relations using the sparse backend
        sg.parameters.sparse_operator = True
        for spin in spins:

            # Go through all the possible ranks and projections
            for l in range(0, int(2*spin+1)):
                for q in range(-l, l+1):

                    # Check the defining commutation relations.
                    self._assert_allclose(
                        comm(sg.op_Sz(spin), sg.op_T(spin, l, q)),
                        q * sg.op_T(spin, l, q),
                    )
                    self._assert_allclose(
                        comm(
                            sg.op_Sx(spin) @ sg.op_Sx(spin)
                            + sg.op_Sy(spin) @ sg.op_Sy(spin)
                            + sg.op_Sz(spin) @ sg.op_Sz(spin),
                            sg.op_T(spin, l, q),
                        ),
                        0,
                    )
                    if not q == -l:
                        self._assert_allclose(
                            comm(sg.op_Sm(spin), sg.op_T(spin, l, q)),
                            math.sqrt(l *(l + 1) - q *(q - 1))
                            * sg.op_T(spin, l, q - 1),
                        )
                    # NOTE: Tolerance of allclose has to be increased a bit
                    if not q == l:
                        self._assert_allclose(
                            comm(sg.op_Sp(spin), sg.op_T(spin, l, q)),
                            math.sqrt(l *(l + 1) - q *(q + 1))
                            * sg.op_T(spin, l, q + 1),
                            atol=1e-7,
                        )

    def test_op_prod(self):
        """
        Test the construction of product operators for different spin quantum
        numbers.
        """
        # Reset parameters to defaults.
        sg.parameters.default()

        # Create a representative set of spin quantum numbers.
        spins = np.array([1 / 2, 1, 3 / 2])

        # Try all possible product operator combinations with the dense backend
        sg.parameters.sparse_operator = False
        for i in range(int(2*spins[0]+1)):
            l_i, q_i = sg.idx_to_lq(i)
            op_i = sg.op_T(spins[0], l_i, q_i)

            for j in range(int(2*spins[1]+1)):
                l_j, q_j = sg.idx_to_lq(j)
                op_j = sg.op_T(spins[1], l_j, q_j)

                for k in range(int(2*spins[2]+1)):
                    l_k, q_k = sg.idx_to_lq(k)
                    op_k = sg.op_T(spins[2], l_k, q_k)

                    # Create the operator using inbuilt function with the unit
                    # operator included
                    op_def = np.array([i, j, k])
                    oper = op_prod(op_def, spins, include_unit=True)

                    # Create the reference operator manually
                    oper_ref = np.kron(op_i, np.kron(op_j, op_k))

                    # Compare the constructed and manual operators.
                    self._assert_allclose(oper, oper_ref)

                    # Create the operator using inbuilt function without
                    # including the unit operator
                    oper = op_prod(op_def, spins, include_unit=False)
                
                    # Create the reference operator manually
                    oper_ref = np.array([[1]])
                    if i != 0:
                        oper_ref = np.kron(oper_ref, op_i)
                    if j != 0:
                        oper_ref = np.kron(oper_ref, op_j)
                    if k != 0:
                        oper_ref = np.kron(oper_ref, op_k)

                    # Compare the constructed and manual operators.
                    self._assert_allclose(oper, oper_ref)

        # Try all possible product operator combinations with the sparse backend
        sg.parameters.sparse_operator = True
        for i in range(int(2*spins[0]+1)):
            l_i, q_i = sg.idx_to_lq(i)
            op_i = sg.op_T(spins[0], l_i, q_i)

            for j in range(int(2*spins[1]+1)):
                l_j, q_j = sg.idx_to_lq(j)
                op_j = sg.op_T(spins[1], l_j, q_j)

                for k in range(int(2*spins[2]+1)):
                    l_k, q_k = sg.idx_to_lq(k)
                    op_k = sg.op_T(spins[2], l_k, q_k)

                    # Create the operator using inbuilt function with the unit
                    # operator included
                    op_def = np.array([i, j, k])
                    oper = op_prod(op_def, spins, include_unit=True)

                    # Create the reference operator manually
                    oper_ref = sp.kron(op_i, sp.kron(op_j, op_k))

                    # Compare the constructed and manual operators.
                    self._assert_allclose(oper, oper_ref)

                    # Create the operator using inbuilt function without
                    # including the unit operator
                    oper = op_prod(op_def, spins, include_unit=False)
                
                    # Create the reference operator manually
                    oper_ref = sp.csc_array([[1]])
                    if i != 0:
                        oper_ref = sp.kron(oper_ref, op_i)
                    if j != 0:
                        oper_ref = sp.kron(oper_ref, op_j)
                    if k != 0:
                        oper_ref = sp.kron(oper_ref, op_k)

                    # Compare the constructed and manual operators.
                    self._assert_allclose(oper, oper_ref)

    def test_op_T_coupled(self):
        """
        Test the coupled spherical tensor operators for two spins.
        """
        # Reset parameters to defaults.
        sg.parameters.default()

        # List the spin quantum numbers to test.
        spins = [1 / 2, 1, 3 / 2, 2, 5 / 2, 3, 7 / 2, 4, 9 / 2, 5, 11 / 2]

        # Test relations given in:
        # Eq. 254-262, Man: Cartesian and Spherical Tensors in NMR Hamiltonian.
        # Using dense arrays
        sg.parameters.sparse_operator = False
        for s1 in spins:
            for s2 in spins:

                # Get the two-spin operators
                SxIx = np.kron(sg.op_Sx(s1), sg.op_Sx(s2))
                SxIy = np.kron(sg.op_Sx(s1), sg.op_Sy(s2))
                SxIz = np.kron(sg.op_Sx(s1), sg.op_Sz(s2))
                SyIx = np.kron(sg.op_Sy(s1), sg.op_Sx(s2))
                SyIy = np.kron(sg.op_Sy(s1), sg.op_Sy(s2))
                SyIz = np.kron(sg.op_Sy(s1), sg.op_Sz(s2))
                SzIx = np.kron(sg.op_Sz(s1), sg.op_Sx(s2))
                SzIy = np.kron(sg.op_Sz(s1), sg.op_Sy(s2))
                SzIz = np.kron(sg.op_Sz(s1), sg.op_Sz(s2))

                # Check the coupled spherical tensor identities.
                self._assert_allclose(
                    sg.op_T_coupled(0, 0, 1, s1, 1, s2),
                    -1 / np.sqrt(3) * (SxIx + SyIy + SzIz),
                )
                self._assert_allclose(
                    sg.op_T_coupled(1, 1, 1, s1, 1, s2),
                    1 / 2 * (SzIx - SxIz + 1j * (SzIy - SyIz)),
                )
                self._assert_allclose(
                    sg.op_T_coupled(1, 0, 1, s1, 1, s2),
                    1j / np.sqrt(2) * (SxIy - SyIx),
                )
                self._assert_allclose(
                    sg.op_T_coupled(1, -1, 1, s1, 1, s2),
                    1 / 2 * (SzIx - SxIz - 1j * (SzIy - SyIz)),
                )
                self._assert_allclose(
                    sg.op_T_coupled(2, 2, 1, s1, 1, s2),
                    1 / 2 * (SxIx - SyIy + 1j * (SxIy + SyIx)),
                )
                self._assert_allclose(
                    sg.op_T_coupled(2, 1, 1, s1, 1, s2),
                    -1 / 2 * (SxIz + SzIx + 1j * (SyIz + SzIy)),
                )
                self._assert_allclose(
                    sg.op_T_coupled(2, 0, 1, s1, 1, s2),
                    1 / np.sqrt(6) * (-SxIx + 2 * SzIz - SyIy),
                )
                self._assert_allclose(
                    sg.op_T_coupled(2, -1, 1, s1, 1, s2),
                    1 / 2 * (SxIz + SzIx - 1j * (SyIz + SzIy)),
                )
                self._assert_allclose(
                    sg.op_T_coupled(2, -2, 1, s1, 1, s2),
                    1 / 2 * (SxIx - SyIy - 1j * (SxIy + SyIx)),
                )

        # Repeat the test for sparse arrays
        sg.parameters.sparse_operator = True
        for s1 in spins:
            for s2 in spins:

                # Get the two-spin operators
                SxIx = sp.kron(sg.op_Sx(s1), sg.op_Sx(s2))
                SxIy = sp.kron(sg.op_Sx(s1), sg.op_Sy(s2))
                SxIz = sp.kron(sg.op_Sx(s1), sg.op_Sz(s2))
                SyIx = sp.kron(sg.op_Sy(s1), sg.op_Sx(s2))
                SyIy = sp.kron(sg.op_Sy(s1), sg.op_Sy(s2))
                SyIz = sp.kron(sg.op_Sy(s1), sg.op_Sz(s2))
                SzIx = sp.kron(sg.op_Sz(s1), sg.op_Sx(s2))
                SzIy = sp.kron(sg.op_Sz(s1), sg.op_Sy(s2))
                SzIz = sp.kron(sg.op_Sz(s1), sg.op_Sz(s2))

                # Check the same identities using sparse operators.
                self._assert_allclose(
                    sg.op_T_coupled(0, 0, 1, s1, 1, s2),
                    -1 / np.sqrt(3) * (SxIx + SyIy + SzIz),
                )
                self._assert_allclose(
                    sg.op_T_coupled(1, 1, 1, s1, 1, s2),
                    1 / 2 * (SzIx - SxIz + 1j * (SzIy - SyIz)),
                )
                self._assert_allclose(
                    sg.op_T_coupled(1, 0, 1, s1, 1, s2),
                    1j / np.sqrt(2) * (SxIy - SyIx),
                )
                self._assert_allclose(
                    sg.op_T_coupled(1, -1, 1, s1, 1, s2),
                    1 / 2 * (SzIx - SxIz - 1j * (SzIy - SyIz)),
                )
                self._assert_allclose(
                    sg.op_T_coupled(2, 2, 1, s1, 1, s2),
                    1 / 2 * (SxIx - SyIy + 1j * (SxIy + SyIx)),
                )
                self._assert_allclose(
                    sg.op_T_coupled(2, 1, 1, s1, 1, s2),
                    -1 / 2 * (SxIz + SzIx + 1j * (SyIz + SzIy)),
                )
                self._assert_allclose(
                    sg.op_T_coupled(2, 0, 1, s1, 1, s2),
                    1 / np.sqrt(6) * (-SxIx + 2 * SzIz - SyIy),
                )
                self._assert_allclose(
                    sg.op_T_coupled(2, -1, 1, s1, 1, s2),
                    1 / 2 * (SxIz + SzIx - 1j * (SyIz + SzIy)),
                )
                self._assert_allclose(
                    sg.op_T_coupled(2, -2, 1, s1, 1, s2),
                    1 / 2 * (SxIx - SyIy - 1j * (SxIy + SyIx)),
                )

    def test_op_from_string(self):
        """
        A test for creating the Hilbert-space operators using the operators
        string.
        """
        # Reset parameters to defaults.
        sg.parameters.default()

        # Create a representative set of spin quantum numbers.
        spins = np.array([1 / 2, 1, 3 / 2])

        # Operators to test
        test_opers = ['E', 'x', 'y', 'z', '+', '-']

        # Collect the single-spin operators in dense format.
        sg.parameters.sparse_operator = False
        opers = {}
        for spin in spins:
            opers.update(
                {
                    (label, spin): operator
                    for label, operator in self._build_cartesian_operator_map(
                        spin
                    ).items()
                }
            )

        # Try all possible product operator combinations using the dense format
        for i in test_opers:
            if i == "E":
                op_i = "E"
            else:
                op_i = f"I({i}, 0)"

            for j in test_opers:
                if j == "E":
                    op_j = "E"
                else:
                    op_j = f"I({j}, 1)"

                for k in test_opers:
                    if k == "E":
                        op_k = "E"
                    else:
                        op_k = f"I({k}, 2)"

                    # Create the operator using inbuilt function
                    op_string = f"{op_i} * {op_j} * {op_k}"
                    oper = op_from_string(spins, op_string)

                    # Create the reference operator
                    oper_ref = np.kron(
                        opers[(i, spins[0])], np.kron(
                            opers[(j, spins[1])], opers[(k, spins[2])]
                        ))

                    # Compare the parsed operator with the reference operator.
                    self._assert_allclose(oper, oper_ref)

        # Test creating a sum operator for all spins using input type I(x)
        for oper in test_opers:
            if oper != "E":      

                op_string = f"I({oper})"
                op_string_ref = f"I({oper},0) + I({oper},1) + I({oper},2)"

                oper = op_from_string(spins, op_string)
                oper_ref = op_from_string(spins, op_string_ref)

                # Compare the shorthand and explicit sum forms.
                self._assert_allclose(oper, oper_ref)

        # Test creating a sum operator for all spins using input type T(l,q)
        for l in range(0, 2):
            for q in range(-l, l+1):

                op_string = f"T({l},{q})"
                op_string_ref = f"T({l},{q},0) + T({l},{q},1) + T({l},{q},2)"

                oper = op_from_string(spins, op_string)
                oper_ref = op_from_string(spins, op_string_ref)

                # Compare the shorthand and explicit sum forms.
                self._assert_allclose(oper, oper_ref)

        # Perform the same tests using the sparse format
        sg.parameters.sparse_operator = True
        for i in test_opers:
            if i == "E":
                op_i = "E"
            else:
                op_i = f"I({i}, 0)"

            for j in test_opers:
                if j == "E":
                    op_j = "E"
                else:
                    op_j = f"I({j}, 1)"

                for k in test_opers:
                    if k == "E":
                        op_k = "E"
                    else:
                        op_k = f"I({k}, 2)"

                    # Create the operator using inbuilt function
                    op_string = f"{op_i} * {op_j} * {op_k}"
                    oper = op_from_string(spins, op_string)

                    # Create the reference operator
                    oper_ref = sp.kron(
                        opers[(i, spins[0])], sp.kron(
                            opers[(j, spins[1])], opers[(k, spins[2])]
                        ))

                    # Compare the parsed operator with the reference operator.
                    self._assert_allclose(oper, oper_ref)

        # Test creating a sum operator for all spins using input type I(x)
        for oper in test_opers:
            if oper != "E":      

                op_string = f"I({oper})"
                op_string_ref = f"I({oper},0) + I({oper},1) + I({oper},2)"

                oper = op_from_string(spins, op_string)
                oper_ref = op_from_string(spins, op_string_ref)

                # Compare the shorthand and explicit sum forms.
                self._assert_allclose(oper, oper_ref)

        # Test creating a sum operator for all spins using input type T(l,q)
        for l in range(0, 2):
            for q in range(-l, l+1):

                op_string = f"T({l},{q})"
                op_string_ref = f"T({l},{q},0) + T({l},{q},1) + T({l},{q},2)"

                oper = op_from_string(spins, op_string)
                oper_ref = op_from_string(spins, op_string_ref)

                # Compare the shorthand and explicit sum forms.
                self._assert_allclose(oper, oper_ref)

    def test_operator(self):
        """
        Test the user-friendly operator function.
        """
        # Reset parameters to defaults.
        sg.parameters.default()

        # Create a representative three-spin system.
        ss = sg.SpinSystem(["1H", "14N", "23Na"])

        # Test using the operator string with the dense backend
        sg.parameters.sparse_operator = False
        op = sg.operator(ss, "I(x, 0) * I(y, 1) * I(z, 2)")
        op_ref = np.kron(
            sg.op_Sx(ss.spins[0]), np.kron(
                sg.op_Sy(ss.spins[1]), sg.op_Sz(ss.spins[2])
            ))
        self._assert_allclose(op, op_ref)

        # Test using the operator string with the sparse backend
        sg.parameters.sparse_operator = True
        op = sg.operator(ss, "I(x, 0) * I(y, 1) * I(z, 2)")
        op_ref = sp.kron(
            sg.op_Sx(ss.spins[0]), sp.kron(
                sg.op_Sy(ss.spins[1]), sg.op_Sz(ss.spins[2])
            ))
        self._assert_allclose(op, op_ref)

        # Test using the operator array with the dense backend
        sg.parameters.sparse_operator = False
        op1 = sg.operator(ss, [1, 2, 3])
        op2 = sg.operator(ss, (1, 2, 3))
        op3 = sg.operator(ss, np.array([1, 2, 3]))
        op_ref = np.kron(
            sg.op_T(ss.spins[0], 1, 1), np.kron(
                sg.op_T(ss.spins[1], 1, 0), sg.op_T(ss.spins[2], 1, -1)
            ))
        self._assert_allclose(op1, op_ref)
        self._assert_allclose(op2, op_ref)
        self._assert_allclose(op3, op_ref)

        # Test using the operator array with the sparse backend
        sg.parameters.sparse_operator = True
        op1 = sg.operator(ss, [1, 2, 3])
        op2 = sg.operator(ss, (1, 2, 3))
        op3 = sg.operator(ss, np.array([1, 2, 3]))
        op_ref = sp.kron(
            sg.op_T(ss.spins[0], 1, 1), sp.kron(
                sg.op_T(ss.spins[1], 1, 0), sg.op_T(ss.spins[2], 1, -1)
            ))
        self._assert_allclose(op1, op_ref)
        self._assert_allclose(op2, op_ref)
        self._assert_allclose(op3, op_ref)

        # Wrong input type should raise an error
        with self.assertRaises(ValueError):
            sg.operator(ss, 1)

        # Incorrect array length raises an error
        with self.assertRaises(ValueError):
            sg.operator(ss, [1, 2])

        # Incorrect array dimension raises an error
        with self.assertRaises(ValueError):
            sg.operator(ss, [[1, 2, 3]])