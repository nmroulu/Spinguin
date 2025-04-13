import unittest
import numpy as np
import math
import copy
from spinguin._spin_system import SpinSystem
from spinguin._basis import idx_to_lq, lq_to_idx, coherence_order, state_idx, transform_to_truncated_basis, truncate_basis_by_coherence, parse_operator_string
from spinguin._operators import superoperator

class TestBasis(unittest.TestCase):

    def test_make_basis(self):
        """
        Test the creation of the basis set for a spin system.
        """
        # Hard-coded result for comparison
        basis_arr = np.array([[0, 0],
                              [0, 1],
                              [0, 2],
                              [0, 3],
                              [1, 0],
                              [2, 0],
                              [3, 0]])
        basis_dict = {
            (0, 0): 0,
            (0, 1): 1,
            (0, 2): 2,
            (0, 3): 3,
            (1, 0): 4,
            (2, 0): 5,
            (3, 0): 6
        }
        
        # Test system
        isotopes = np.array(['1H', '1H'])
        spin_system = SpinSystem(isotopes, max_spin_order=1)

        # Compare the generated basis with the hard-coded result
        self.assertTrue(np.array_equal(basis_arr, spin_system.basis.arr))
        self.assertEqual(basis_dict, spin_system.basis.dict)

        # Larger test system
        isotopes = np.array(['1H', '1H', '1H', '1H', '1H', '1H', '1H'])

        # Compare the size of the basis set to a calculated value for different spin orders
        for max_so in range(1, isotopes.size):
            spin_system = SpinSystem(isotopes, max_spin_order=max_so)
            size = sum([math.comb(isotopes.size, k) * 3**k for k in range(max_so + 1)])
            self.assertEqual(size, spin_system.basis.dim)

    def test_state_idx(self):
        """
        Test the mapping of states to their corresponding indices.
        """
        # Test system
        isotopes = np.array(['1H', '1H'])
        spin_system = SpinSystem(isotopes, max_spin_order=1)

        # Test against known values
        self.assertEqual(state_idx(spin_system, (0, 0)), 0)
        self.assertEqual(state_idx(spin_system, (0, 1)), 1)
        self.assertEqual(state_idx(spin_system, (0, 2)), 2)
        self.assertEqual(state_idx(spin_system, (0, 3)), 3)
        self.assertEqual(state_idx(spin_system, (1, 0)), 4)
        self.assertEqual(state_idx(spin_system, (2, 0)), 5)
        self.assertEqual(state_idx(spin_system, (3, 0)), 6)

    def test_parse_operator_string(self):
        """
        Test parsing the operator strings.
        TODO: Better test.
        """

        # Test system
        isotopes = np.array(['1H', '1H'])
        spin_system = SpinSystem(isotopes)

        print(parse_operator_string(spin_system, "I(z, 0) * I(z, 1) + I(+, 1) + I(x, 0) + T(1, 0, 0)"))

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
        self.assertEqual(coherence_order((0,)), 0)
        self.assertEqual(coherence_order((1,)), 1)
        self.assertEqual(coherence_order((2,)), 0)
        self.assertEqual(coherence_order((3,)), -1)
        self.assertEqual(coherence_order((4,)), 2)
        self.assertEqual(coherence_order((5,)), 1)
        self.assertEqual(coherence_order((6,)), 0)
        self.assertEqual(coherence_order((7,)), -1)
        self.assertEqual(coherence_order((8,)), -2)
        self.assertEqual(coherence_order((1, 4)), 3)
        self.assertEqual(coherence_order((0, 1, 2, 3, 4, 5, 6, 7, 8)), 0)

    def test_truncate_basis_by_coherence(self):
        """
        Test the creation of the truncated basis.
        """
        # Example system
        isotopes = np.array(['1H', '1H', '1H', '1H', '14N'])
        spin_system = SpinSystem(isotopes, max_spin_order=4)

        # Save the original basis
        basis_org = copy.copy(spin_system.basis)

        # Truncate the basis (retain coherence orders of -2, 0, and 1)
        coherence_orders = [-2, 0, 1]
        index_map = truncate_basis_by_coherence(spin_system, coherence_orders)

        # Check that only the coherence orders [-2, 0, 1] remain and that the index map is correct
        for op, idx in basis_org.dict.items():
            if op in spin_system.basis.dict:
                self.assertTrue(coherence_order(op) in coherence_orders)
                self.assertEqual(index_map[spin_system.basis.dict[op]], idx)

    def test_transform_to_truncated_basis(self):
        """
        Test the transformation of superoperators to truncated basis.
        """
        # Example systems
        isotopes = np.array(['1H', '1H', '1H', '1H', '14N'])
        spin_system = SpinSystem(isotopes, max_spin_order=4)
        spin_system_tr = SpinSystem(isotopes, max_spin_order=4)
        index_map = truncate_basis_by_coherence(spin_system_tr, [-2, 0, 1])

        # Operators to test
        operators = ['x', 'y', 'z', '+', '-']

        # Try all possible combinations
        for i in operators:
            for j in operators:
                for k in operators:

                    # Create the superoperators in original and truncated basis
                    sop = superoperator(spin_system, f"I({i},0) * I({j},1) * I({k}, 2)")
                    sop_tr = superoperator(spin_system_tr, f"I({i},0) * I({j},1) * I({k}, 2)")

                    # Applying the transformation should result in the same result
                    self.assertTrue(np.allclose(sop_tr.toarray(), transform_to_truncated_basis(index_map, sop).toarray()))