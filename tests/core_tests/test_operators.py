"""
Tests for single-spin and multi-spin operator construction.
"""

import math
import unittest
from typing import Callable

import numpy as np
import scipy.sparse as sp

import spinguin as sg
from spinguin._core._la import comm
from ._helpers import build_spin_system


class TestOperators(unittest.TestCase):
    """
    Test operator factories and operator-string parsing.
    """

    # List the spin quantum numbers used in the commutator tests.
    SPINS = [1/2, 1, 3/2]

    def test_op_S(self):
        """
        The operators `E`, `Sx`, `Sy`, `Sz`, `Sp`, and `Sm` are compared with
        the angular momentum commutation relations.
        """
        def _test_op_S(zeros: Callable, sparse: bool) -> None:
            sg.parameters.sparse_operator = sparse

            # Test with each spin quantum number
            for spin in self.SPINS:

                # Build single-spin operators for the current spin
                E  = sg.op_E(spin)
                Sx = sg.op_Sx(spin)
                Sy = sg.op_Sy(spin)
                Sz = sg.op_Sz(spin)
                Sp = sg.op_Sp(spin)
                Sm = sg.op_Sm(spin)

                # Map the commutators to their expected results
                relations = [
                    (comm(E,  E),  zeros(E.shape)),
                    (comm(Sx, Sy), 1j * Sz),
                    (comm(Sy, Sz), 1j * Sx),
                    (comm(Sz, Sx), 1j * Sy),
                    (comm(Sp, Sz), -Sp),
                    (comm(Sm, Sz), Sm),
                ]

                # Test all relations
                for actual, expected in relations:
                    if sparse:
                        actual = actual.toarray()
                        expected = expected.toarray()
                    self.assertTrue(np.allclose(actual, expected))

        # Reset parameters to defaults
        sg.parameters.default()

        # Test with dense and sparse backends
        _test_op_S(zeros=np.zeros, sparse=False)
        _test_op_S(zeros=sp.csc_array, sparse=True)


    def test_op_T(self):
        """
        Test commutation relations for spherical tensor operators.
        """
        # Reset parameters to defaults
        sg.parameters.default()

        def _test_op_T(zeros: Callable, sparse: bool) -> None:
            sg.parameters.sparse_operator = sparse

            # Test with each spin quantum number
            for spin in self.SPINS:

                # Calculate single-spin operators for the current spin
                Sx = sg.op_Sx(spin)
                Sy = sg.op_Sy(spin)
                Sz = sg.op_Sz(spin)
                Sp = sg.op_Sp(spin)
                Sm = sg.op_Sm(spin)
                S2 = Sx @ Sx + Sy @ Sy + Sz @ Sz

                # Test all ranks and projections
                for l in range(0, int(2*spin + 1)):
                    for q in range(-l, l + 1):

                        T_lq = sg.op_T(spin, l, q)

                        # Define the commutation relations.
                        relations = [
                            (comm(Sz, T_lq), q * T_lq),
                            (comm(S2, T_lq), zeros(T_lq.shape)),
                        ]
                        if q != -l:
                            T_lqm1 = sg.op_T(spin, l, q - 1)
                            relations.append((
                                comm(sg.op_Sm(spin), T_lq),
                                math.sqrt(l*(l + 1) - q*(q - 1)) * T_lqm1
                            ))
                        if not q == l:
                            T_lqp1 = sg.op_T(spin, l, q + 1)
                            relations.append((
                                comm(sg.op_Sp(spin), T_lq),
                                math.sqrt(l*(l + 1) - q*(q + 1)) * T_lqp1 
                            ))

                        # Test the relations
                        for actual, expected in relations:
                            if sparse:
                                actual = actual.toarray()
                                expected = expected.toarray()
                            # NOTE: Tolerance of allclose increased slightly
                            self.assertTrue(
                                np.allclose(actual, expected, atol=1e-7)
                            )

        # Test with dense and sparse backends
        _test_op_T(zeros=np.zeros, sparse=False)
        _test_op_T(zeros=sp.csc_array, sparse=True)


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

        def _test_product_operators(kron: Callable, sparse: bool) -> None:
            sg.parameters.sparse_operator = sparse

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
                        oper_ref = kron(op_i, kron(op_j, op_k))

                        if sparse:
                            oper = oper.toarray()
                            oper_ref = oper_ref.toarray()

                        # Compare the constructed and manual operators.
                        self.assertTrue(np.allclose(oper, oper_ref))

        # Test all product operator combinations with dense and sparse backends
        _test_product_operators(kron=np.kron, sparse=False)
        _test_product_operators(kron=sp.kron, sparse=True)


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
        for i in range(ss.nspins):
            ops = {}

            # Unit operator
            ops["E"] = sg.op_E(ss.spins[i])

            # Cartesian operators
            ops[f"I(x, {i})"] = sg.op_Sx(ss.spins[i])
            ops[f"I(y, {i})"] = sg.op_Sy(ss.spins[i])
            ops[f"I(z, {i})"] = sg.op_Sz(ss.spins[i])

            # Ladder operators
            ops[f"I(+, {i})"] = sg.op_Sp(ss.spins[i])
            ops[f"I(-, {i})"] = sg.op_Sm(ss.spins[i])

            # Spherical tensor operators
            for l in range(int(2*ss.spins[i] + 1)):
                for q in range(-l, l + 1):
                    ops[f"T({l}, {q}, {i})"] = sg.op_T(ss.spins[i], l, q)
            
            test_opers.append(ops)

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
        Test the coupled spherical tensor operators for two spins using the
        relations given in:
        
        Eq. 254-262, Man: Cartesian and Spherical Tensors in NMR Hamiltonian.
        """
        def _test_op_T_coupled(kron: Callable, sparse: bool) -> None:
            sg.parameters.sparse_operator = sparse

            # Test all spin quantum number combinations
            for s1 in self.SPINS:
                for s2 in self.SPINS:

                    # Get the two-spin operators
                    SxIx = kron(sg.op_Sx(s1), sg.op_Sx(s2))
                    SxIy = kron(sg.op_Sx(s1), sg.op_Sy(s2))
                    SxIz = kron(sg.op_Sx(s1), sg.op_Sz(s2))
                    SyIx = kron(sg.op_Sy(s1), sg.op_Sx(s2))
                    SyIy = kron(sg.op_Sy(s1), sg.op_Sy(s2))
                    SyIz = kron(sg.op_Sy(s1), sg.op_Sz(s2))
                    SzIx = kron(sg.op_Sz(s1), sg.op_Sx(s2))
                    SzIy = kron(sg.op_Sz(s1), sg.op_Sy(s2))
                    SzIz = kron(sg.op_Sz(s1), sg.op_Sz(s2))

                    # Write the relations for each (l, q)
                    relations = [
                        ((0,  0), -1 / np.sqrt(3) * (SxIx + SyIy + SzIz)),
                        ((1,  1),  1 / 2 * (SzIx - SxIz + 1j * (SzIy - SyIz))),
                        ((1,  0),  1j / np.sqrt(2) * (SxIy - SyIx)),
                        ((1, -1),  1 / 2 * (SzIx - SxIz - 1j * (SzIy - SyIz))),
                        ((2,  2),  1 / 2 * (SxIx - SyIy + 1j * (SxIy + SyIx))),
                        ((2,  1), -1 / 2 * (SxIz + SzIx + 1j * (SyIz + SzIy))),
                        ((2,  0),  1 / np.sqrt(6) * (-SxIx + 2 * SzIz - SyIy)),
                        ((2, -1),  1 / 2 * (SxIz + SzIx - 1j * (SyIz + SzIy))),
                        ((2, -2),  1 / 2 * (SxIx - SyIy - 1j * (SxIy + SyIx))),
                    ]

                    # Test all relations
                    for (l, q), op_T_coupled_ref in relations:

                        # Create the operator using inbuilt function
                        op_T_coupled = sg.op_T_coupled(l, q, 1, s1, 1, s2)

                        if sparse:
                            op_T_coupled = op_T_coupled.toarray()
                            op_T_coupled_ref = op_T_coupled_ref.toarray()

                        # Compare the parsed operator with the reference.
                        self.assertTrue(
                            np.allclose(op_T_coupled, op_T_coupled_ref)
                        )

        # Reset the parameters to defaults
        sg.parameters.default()

        # Test with dense and sparse backends
        _test_op_T_coupled(kron=np.kron, sparse=False)
        _test_op_T_coupled(kron=sp.kron, sparse=True)