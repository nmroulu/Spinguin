import unittest
import numpy as np
import math
from spinguin._core._utils import (
    state_idx,
    parse_operator_string,
    idx_to_lq,
    lq_to_idx,
    coherence_order
)
from spinguin._core._basis import (
    make_basis,
    truncate_basis_by_coherence
)

class TestBasis(unittest.TestCase):

    def test_make_basis_1(self):
        """
        Test the creation of the basis set for a spin system
        against a hard-coded result.
        """

        # Hard-coded result for comparison
        basis_ref = np.array([[0, 0],
                              [0, 1],
                              [0, 2],
                              [0, 3],
                              [1, 0],
                              [2, 0],
                              [3, 0]])
        
        # Test system
        spins = np.array([1/2, 1/2])
        max_spin_order = 1

        # Create the basis
        basis = make_basis(spins, max_spin_order)

        # Compare the generated basis with the hard-coded result
        self.assertTrue(np.array_equal(basis, basis_ref))

    def test_make_basis_2(self):
        """
        Test the creation of a basis set using large spin system.
        """

        # Larger test system
        spins = np.array([1/2, 1/2, 1/2, 1/2, 1/2, 1/2, 1/2])
        nspins = spins.shape[0]

        # Compare the dimension of the basis set to a reference value for
        # different spin orders
        for max_so in range(1, nspins):
            basis = make_basis(spins, max_so)
            dim_ref = sum(
                [math.comb(nspins, k) * 3**k for k in range(max_so + 1)])
            dim = basis.shape[0]
            self.assertEqual(dim, dim_ref)

    def test_state_idx(self):
        """
        Test the mapping of states to their corresponding indices.
        """
        # Make an arbitrary basis set
        basis = np.array([
            [0, 0, 0],
            [0, 0, 1],
            [0, 2, 0],
            [3, 1, 1],
            [5, 0, 1]
        ])

        # Test searching the index
        self.assertEqual(state_idx(basis, np.array([0, 0, 0])), 0)
        self.assertEqual(state_idx(basis, np.array([0, 0, 1])), 1)
        self.assertEqual(state_idx(basis, np.array([0, 2, 0])), 2)
        self.assertEqual(state_idx(basis, np.array([3, 1, 1])), 3)
        self.assertEqual(state_idx(basis, np.array([5, 0, 1])), 4)

        # Test searching for a non-existent state
        self.assertRaises(ValueError, state_idx, basis, np.array([9, 9, 9]))

        # Test searching state with wrong dimension
        self.assertRaises(ValueError, state_idx, basis, np.array([0]))
    
        # Test searching for a state duplicate
        basis[2] = [0, 0, 0]
        self.assertRaises(ValueError,
                          state_idx, basis, op_def=np.array([0, 0, 0]))

    def test_parse_operator_string(self):
        """
        Test parsing the operator strings.
        """

        # Set number of spins
        nspins = 3

        # Test empty operator string
        operator = ""
        op_defs, coeffs = parse_operator_string(operator, nspins)
        self.assertTrue(len(op_defs) == 1)
        self.assertTrue(len(coeffs) == 1)
        self.assertTrue(np.array_equal(op_defs[0], np.array([0, 0, 0])))
        self.assertTrue(coeffs[0] == 1)

        # Test Cartesian operator
        operator = "I(x, 0)"
        op_defs, coeffs = parse_operator_string(operator, nspins)
        self.assertTrue(len(op_defs) == 2)
        self.assertTrue(len(coeffs) == 2)
        self.assertTrue(np.array_equal(op_defs[0], np.array([1, 0, 0])))
        self.assertTrue(np.array_equal(op_defs[1], np.array([3, 0, 0])))
        self.assertTrue(coeffs[0] == -np.sqrt(2) / 2)
        self.assertTrue(coeffs[1] == np.sqrt(2) / 2)

        # Test ladder operator
        operator = "I(+, 1)"
        op_defs, coeffs = parse_operator_string(operator, nspins)
        self.assertTrue(len(op_defs) == 1)
        self.assertTrue(len(coeffs) == 1)
        self.assertTrue(np.array_equal(op_defs[0], np.array([0, 1, 0])))
        self.assertTrue(coeffs[0] == -np.sqrt(2))

        # Test spherical tensor operator
        operator = "T(1, -1, 2)"
        op_defs, coeffs = parse_operator_string(operator, nspins)
        self.assertTrue(len(op_defs) == 1)
        self.assertTrue(len(coeffs) == 1)
        self.assertTrue(np.array_equal(op_defs[0], np.array([0, 0, 3])))
        self.assertTrue(coeffs[0] == 1)

        # Test product operator
        operator = "I(z, 0) * I(z, 1)"
        op_defs, coeffs = parse_operator_string(operator, nspins)
        self.assertTrue(len(op_defs) == 1)
        self.assertTrue(len(coeffs) == 1)
        self.assertTrue(np.array_equal(op_defs[0], np.array([2, 2, 0])))
        self.assertTrue(coeffs[0] == 1)

        # Test sum of operators
        operator = "I(x, 0) + I(x, 1)"
        op_defs, coeffs = parse_operator_string(operator, nspins)
        self.assertTrue(len(op_defs) == 4)
        self.assertTrue(len(coeffs) == 4)
        self.assertTrue(np.array_equal(op_defs[0], np.array([1, 0, 0])))
        self.assertTrue(np.array_equal(op_defs[1], np.array([3, 0, 0])))
        self.assertTrue(np.array_equal(op_defs[2], np.array([0, 1, 0])))
        self.assertTrue(np.array_equal(op_defs[3], np.array([0, 3, 0])))
        self.assertTrue(coeffs[0] == -np.sqrt(2) / 2)
        self.assertTrue(coeffs[1] == np.sqrt(2) / 2)
        self.assertTrue(coeffs[2] == -np.sqrt(2) / 2)
        self.assertTrue(coeffs[3] == np.sqrt(2) / 2)

    def test_idx_conversions(self):
        """
        Test the conversion between index and (l, q) representations.
        """
        # Convert back and forth and compare
        for l in range(10):
            for q in range(-l, l + 1):
                self.assertEqual(idx_to_lq(lq_to_idx(l, q)), (l, q))    
        for idx in range(100):
            self.assertEqual(lq_to_idx(*idx_to_lq(idx)), idx)

    def test_coherence_order(self):
        """
        Test the calculation of coherence orders for given states.
        """
        # Test against states and their known coherence orders
        self.assertEqual(coherence_order(np.array([0])), 0)
        self.assertEqual(coherence_order(np.array([1])), 1)
        self.assertEqual(coherence_order(np.array([2])), 0)
        self.assertEqual(coherence_order(np.array([3])), -1)
        self.assertEqual(coherence_order(np.array([4])), 2)
        self.assertEqual(coherence_order(np.array([5])), 1)
        self.assertEqual(coherence_order(np.array([6])), 0)
        self.assertEqual(coherence_order(np.array([7])), -1)
        self.assertEqual(coherence_order(np.array([8])), -2)
        self.assertEqual(coherence_order(np.array([1, 4])), 3)
        self.assertEqual(
            coherence_order(np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])), 0)

    def test_truncate_basis_by_coherence(self):
        """
        Test the creation of the truncated basis.
        """
        # Example system
        spins = np.array([1/2, 1/2, 1/2, 1/2, 1])
        max_spin_order = 4

        # Create a basis set with all coherence orders 
        basis_org = make_basis(spins, max_spin_order)

        # Truncate the basis (retain coherence orders of -2, 0, and 1)
        coherence_orders = [-2, 0, 1]
        basis_truncated, index_map = truncate_basis_by_coherence(
                                        basis_org, coherence_orders)

        # Check that only the coherence orders [-2, 0, 1] remain and that the
        # index map is correct
        for idx, op in enumerate(basis_org):
            if coherence_order(op) in coherence_orders:
                self.assertEqual(index_map[state_idx(basis_truncated, op)], idx)
            else:
                self.assertRaises(ValueError, state_idx, basis_truncated, op)