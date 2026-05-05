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
from ._helpers import build_spin_system


class TestOperators(unittest.TestCase):
    """
    Test operator factories and operator-string parsing.
    """

    # List the spin quantum numbers used in the commutator tests.
    SPINS = [1/2, 1, 3/2]

    def test_op_S(self):
        """
        The operators `E`, `Sx`, `Sy`, `Sz`, `Sp`, and `Sm` are compared
        the angular momentum commutation relations.
        """
        # Reset parameters to defaults
        sg.parameters.default()

        # Test the commutation relations using the dense backend.
        sg.parameters.sparse_operator = False
        for spin in self.SPINS:

            # Check the standard angular momentum commutators.
            self.assertTrue(np.allclose(comm(sg.op_E(spin), sg.op_E(spin)), 0))
            self.assertTrue(np.allclose(
                comm(sg.op_Sx(spin), sg.op_Sy(spin)),
                1j*sg.op_Sz(spin)
            ))
            self.assertTrue(np.allclose(
                comm(sg.op_Sy(spin), sg.op_Sz(spin)),
                1j*sg.op_Sx(spin)
            ))
            self.assertTrue(np.allclose(
                comm(sg.op_Sz(spin), sg.op_Sx(spin)),
                1j*sg.op_Sy(spin)
            ))
            self.assertTrue(np.allclose(
                comm(sg.op_Sp(spin), sg.op_Sz(spin)),
                -sg.op_Sp(spin)
            ))
            self.assertTrue(np.allclose(
                comm(sg.op_Sm(spin), sg.op_Sz(spin)),
                sg.op_Sm(spin)
            ))

        # Test the commutation relations using the sparse backend.
        sg.parameters.sparse_operator = True
        for spin in self.SPINS:

            # Test commutation relations
            self.assertTrue(np.allclose(
                comm(sg.op_E(spin), sg.op_E(spin)).toarray(),
                0
            ))
            self.assertTrue(np.allclose(
                comm(sg.op_Sx(spin), sg.op_Sy(spin)).toarray(),
                1j*sg.op_Sz(spin).toarray()
            ))
            self.assertTrue(np.allclose(
                comm(sg.op_Sy(spin), sg.op_Sz(spin)).toarray(),
                1j*sg.op_Sx(spin).toarray()
            ))
            self.assertTrue(np.allclose(
                comm(sg.op_Sz(spin), sg.op_Sx(spin)).toarray(),
                1j*sg.op_Sy(spin).toarray()
            ))
            self.assertTrue(np.allclose(
                comm(sg.op_Sp(spin), sg.op_Sz(spin)).toarray(),
                -sg.op_Sp(spin).toarray()
            ))
            self.assertTrue(np.allclose(
                comm(sg.op_Sm(spin), sg.op_Sz(spin)).toarray(),
                sg.op_Sm(spin).toarray()
            ))

    def test_op_T(self):
        """
        Test commutation relations for spherical tensor operators.
        """
        # Reset parameters to defaults
        sg.parameters.default()

        # Test the commutation relations using the dense backend.
        sg.parameters.sparse_operator = False
        for spin in self.SPINS:

            # Go through all the possible ranks and projections
            for l in range(0, int(2*spin+1)):
                for q in range(-l, l+1):

                    # Check the defining commutation relations.
                    self.assertTrue(np.allclose(
                        comm(sg.op_Sz(spin), sg.op_T(spin, l, q)),
                        q*sg.op_T(spin, l, q)
                    ))
                    self.assertTrue(np.allclose(
                        comm(
                            sg.op_Sx(spin)@sg.op_Sx(spin) + \
                            sg.op_Sy(spin)@sg.op_Sy(spin) + \
                            sg.op_Sz(spin)@sg.op_Sz(spin),
                            sg.op_T(spin, l, q)
                        ), 
                        0
                    ))
                    if not q == -l:
                        self.assertTrue(np.allclose(
                            comm(sg.op_Sm(spin), sg.op_T(spin, l, q)),
                            math.sqrt(l*(l + 1) - q*(q - 1)) * sg.op_T(spin, l, q - 1)
                        ))
                    if not q == l:
                        self.assertTrue(np.allclose(
                            comm(sg.op_Sp(spin), sg.op_T(spin, l, q)),
                            math.sqrt(l*(l + 1) - q*(q + 1))
                            * sg.op_T(spin, l, q + 1)
                        ))

        # Test the commutation relations using the sparse backend
        sg.parameters.sparse_operator = True
        for spin in self.SPINS:

            # Go through all the possible ranks and projections
            for l in range(0, int(2*spin+1)):
                for q in range(-l, l+1):

                    # Check the defining commutation relations.
                    self.assertTrue(np.allclose(
                        comm(sg.op_Sz(spin), sg.op_T(spin, l, q)).toarray(),
                        q*sg.op_T(spin, l, q).toarray()
                    ))
                    self.assertTrue(np.allclose(
                        comm(
                            sg.op_Sx(spin)@sg.op_Sx(spin) + \
                            sg.op_Sy(spin)@sg.op_Sy(spin) + \
                            sg.op_Sz(spin)@sg.op_Sz(spin),
                            sg.op_T(spin, l, q)
                        ).toarray(), 0
                    ))
                    if not q == -l:
                        self.assertTrue(np.allclose(
                            comm(sg.op_Sm(spin), sg.op_T(spin, l, q)).toarray(),
                            math.sqrt(l*(l+1) - q*(q-1)) * \
                            sg.op_T(spin, l, q-1).toarray()
                        ))
                    # NOTE: Tolerance of allclose has to be increased a bit
                    if not q == l:
                        self.assertTrue(np.allclose(
                            comm(sg.op_Sp(spin), sg.op_T(spin, l, q)).toarray(),
                            math.sqrt(l*(l+1) - q*(q+1))
                            * sg.op_T(spin, l, q+1).toarray(),
                            atol = 1e-7
                        ))

    def test_operator_1(self):
        """
        Test the construction of product operators using the integer array
        input.
        """
        # Reset parameters to defaults.
        sg.parameters.default()

        # Create a test spin system.
        ss = build_spin_system(["1H", "14N", "23Na"], 3)

        # Wrong input type should raise an error
        with self.assertRaises(ValueError):
            sg.operator(ss, 1)

        # Incorrect array length raises an error
        with self.assertRaises(ValueError):
            sg.operator(ss, [1, 2])

        # Incorrect array dimension raises an error
        with self.assertRaises(ValueError):
            sg.operator(ss, [[1, 2, 3]])

        def _test_product_operators() -> None:
            for i in range(ss.mults[0]):
                l_i, q_i = sg.idx_to_lq(i)
                op_i = sg.op_T(ss.spins[0], l_i, q_i)

                for j in range(ss.mults[1]):
                    l_j, q_j = sg.idx_to_lq(j)
                    op_j = sg.op_T(ss.spins[1], l_j, q_j)

                    for k in range(ss.mults[2]):
                        l_k, q_k = sg.idx_to_lq(k)
                        op_k = sg.op_T(ss.spins[2], l_k, q_k)

                        # Create the operator using inbuilt function
                        op_def = [i, j, k]
                        oper = sg.operator(ss, op_def)

                        # Create the reference operator manually
                        if sg.parameters.sparse_operator:
                            oper_ref = sp.kron(op_i, sp.kron(op_j, op_k))
                        else:
                            oper_ref = np.kron(op_i, np.kron(op_j, op_k))

                        if sg.parameters.sparse_operator:
                            oper = oper.toarray()
                            oper_ref = oper_ref.toarray()

                        # Compare the constructed and manual operators.
                        self.assertTrue(np.allclose(oper, oper_ref))

        # Test all product operator combinations with dense and sparse backends
        for sparse in [False, True]:
            sg.parameters.sparse_operator = sparse
            _test_product_operators()


    def test_operator_2(self):
        """
        Test the construction of product operators using the string input.
        """
        # Reset parameters to defaults
        sg.parameters.default()

        # Create a test spin system.
        ss = build_spin_system(["1H", "14N"], 2)

        # Create a set of operators to test
        sg.parameters.sparse_operator = False
        test_opers = []
        for spin in range(ss.nspins):
            opers = {}
            opers["E"] = sg.op_E(ss.spins[spin])
            opers[f"I(x, {spin})"] = sg.op_Sx(ss.spins[spin])
            opers[f"I(y, {spin})"] = sg.op_Sy(ss.spins[spin])
            opers[f"I(z, {spin})"] = sg.op_Sz(ss.spins[spin])
            opers[f"I(+, {spin})"] = sg.op_Sp(ss.spins[spin])
            opers[f"I(-, {spin})"] = sg.op_Sm(ss.spins[spin])
            for l in range(2*spin + 1):
                for q in range(-l, l + 1):
                    opers[f"T({l}, {q}, {spin})"] = \
                        sg.op_T(ss.spins[spin], l, q)
            test_opers.append(opers)

        def _test_single_spin_operators() -> None:
            for spin in range(ss.nspins):
                for op_k, op_v in test_opers[spin].items():

                    # Create the operator using inbuilt function
                    oper = sg.operator(ss, op_k)

                    # Create the reference operator
                    if spin == 0:
                        oper_ref = np.kron(op_v, test_opers[1]["E"])
                    else:
                        oper_ref = np.kron(test_opers[0]["E"], op_v)

                    # Compare the parsed operator with the reference operator.
                    if sg.parameters.sparse_operator:
                        oper = oper.toarray()
                    self.assertTrue(np.allclose(oper, oper_ref))

        def _test_product_operators() -> None:
            for op1_k, op1_v in test_opers[0].items():
                for op2_k, op2_v in test_opers[1].items():

                    # Create the operator using inbuilt function
                    op_string = f"{op1_k} * {op2_k}"
                    oper = sg.operator(ss, op_string)

                    # Create the reference operator
                    oper_ref = np.kron(op1_v, op2_v)

                    # Compare the parsed operator with the reference operator.
                    if sg.parameters.sparse_operator:
                        oper = oper.toarray()
                    self.assertTrue(np.allclose(oper, oper_ref))

        def _test_sum_operators() -> None:
            # Collect operators to be tested
            cases = []

            # Cartesian + ladder
            for oper in ["x", "y", "z", "+", "-"]:
                case = {}
                case["inp1"] = f"I({oper})"
                case["inp2"] = f"I({oper}, {0}) + I({oper}, {1})"
                case["ref"] = [f"I({oper}, {0})", f"I({oper}, {1})"]
                cases.append(case)

            # Spherical tensor
            for l in range(2):
                for q in range(-l, l+1):
                    case = {}
                    case["inp1"] = f"T({l}, {q})"
                    case["inp2"] = f"T({l}, {q}, 0) + T({l}, {q}, 1)"
                    case["ref"] =  [f"T({l}, {q}, 0)", f"T({l}, {q}, 1)"]
                    cases.append(case)

            for case in cases:
                # Create the operator using inbuilt function
                oper1 = sg.operator(ss, case["inp1"])
                oper2 = sg.operator(ss, case["inp2"])

                # Create the reference operator
                oper_ref = sum(
                    sg.operator(ss, case["ref"][i]) for i in range(ss.nspins)
                )

                if sg.parameters.sparse_operator:
                    oper1 = oper1.toarray()
                    oper2 = oper2.toarray()
                    oper_ref = oper_ref.toarray()

                # Compare the parsed operator with the reference operator.
                self.assertTrue(np.allclose(oper1, oper_ref), msg=case["inp1"])
                self.assertTrue(np.allclose(oper2, oper_ref), msg=case["inp2"])

        # Try all operator inputs with dense and sparse backends.
        for sparse in [False, True]:
            sg.parameters.sparse_operator = sparse
            _test_single_spin_operators()
            _test_product_operators()
            _test_sum_operators()


    def test_op_T_coupled(self):
        """
        Test the coupled spherical tensor operators for two spins.
        """
        # Reset the parameters to defaults
        sg.parameters.default()

        # Spin quantum numbers to test
        spins = [1/2, 1, 3/2, 2, 5/2, 3, 7/2, 4, 9/2, 5, 11/2]

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

                # Test the relations
                self.assertTrue(np.allclose(
                    sg.op_T_coupled(0, 0, 1, s1, 1, s2),
                    -1 / np.sqrt(3) * (SxIx + SyIy + SzIz)
                ))
                self.assertTrue(np.allclose(
                    sg.op_T_coupled(1, 1, 1, s1, 1, s2),
                    1 / 2 * (SzIx - SxIz + 1j * (SzIy - SyIz))
                ))
                self.assertTrue(np.allclose(
                    sg.op_T_coupled(1, 0, 1, s1, 1, s2),
                    1j / np.sqrt(2) * (SxIy - SyIx)
                ))
                self.assertTrue(np.allclose(
                    sg.op_T_coupled(1, -1, 1, s1, 1, s2),
                    1 / 2 * (SzIx - SxIz - 1j * (SzIy - SyIz))
                ))
                self.assertTrue(np.allclose(
                    sg.op_T_coupled(2, 2, 1, s1, 1, s2),
                    1 / 2 * (SxIx - SyIy + 1j * (SxIy + SyIx))
                ))
                self.assertTrue(np.allclose(
                    sg.op_T_coupled(2, 1, 1, s1, 1, s2),
                    -1 / 2 * (SxIz + SzIx + 1j * (SyIz + SzIy))
                ))
                self.assertTrue(np.allclose(
                    sg.op_T_coupled(2, 0, 1, s1, 1, s2),
                    1 / np.sqrt(6) * (-SxIx + 2 * SzIz - SyIy)
                ))
                self.assertTrue(np.allclose(
                    sg.op_T_coupled(2, -1, 1, s1, 1, s2),
                    1 / 2 * (SxIz + SzIx - 1j * (SyIz + SzIy))
                ))
                self.assertTrue(np.allclose(
                    sg.op_T_coupled(2, -2, 1, s1, 1, s2),
                    1 / 2 * (SxIx - SyIy - 1j * (SxIy + SyIx))
                ))

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

                # Test the relations
                self.assertTrue(np.allclose(
                    sg.op_T_coupled(0, 0, 1, s1, 1, s2).toarray(),
                    -1 / np.sqrt(3) * (SxIx + SyIy + SzIz).toarray()
                ))
                self.assertTrue(np.allclose(
                    sg.op_T_coupled(1, 1, 1, s1, 1, s2).toarray(),
                    1 / 2 * (SzIx - SxIz + 1j * (SzIy - SyIz)).toarray()
                ))
                self.assertTrue(np.allclose(
                    sg.op_T_coupled(1, 0, 1, s1, 1, s2).toarray(),
                    1j / np.sqrt(2) * (SxIy - SyIx).toarray()
                ))
                self.assertTrue(np.allclose(
                    sg.op_T_coupled(1, -1, 1, s1, 1, s2).toarray(),
                    1 / 2 * (SzIx - SxIz - 1j * (SzIy - SyIz)).toarray()
                ))
                self.assertTrue(np.allclose(
                    sg.op_T_coupled(2, 2, 1, s1, 1, s2).toarray(),
                    1 / 2 * (SxIx - SyIy + 1j * (SxIy + SyIx)).toarray()
                ))
                self.assertTrue(np.allclose(
                    sg.op_T_coupled(2, 1, 1, s1, 1, s2).toarray(),
                    -1 / 2 * (SxIz + SzIx + 1j * (SyIz + SzIy)).toarray()
                ))
                self.assertTrue(np.allclose(
                    sg.op_T_coupled(2, 0, 1, s1, 1, s2).toarray(),
                    1 / np.sqrt(6) * (-SxIx + 2 * SzIz - SyIy).toarray()
                ))
                self.assertTrue(np.allclose(
                    sg.op_T_coupled(2, -1, 1, s1, 1, s2).toarray(),
                    1 / 2 * (SxIz + SzIx - 1j * (SyIz + SzIy)).toarray()
                ))
                self.assertTrue(np.allclose(
                    sg.op_T_coupled(2, -2, 1, s1, 1, s2).toarray(),
                    1 / 2 * (SxIx - SyIy - 1j * (SxIy + SyIx)).toarray()
                ))