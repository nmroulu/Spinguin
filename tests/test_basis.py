import unittest
import numpy as np
import math
import copy
from spinguin._spin_system import SpinSystem
from spinguin._basis import idx_to_lq, lq_to_idx, coherence_order, state_idx, str_to_op_def, ZQ_basis, ZQ_filter
from spinguin._operators import superoperator

class TestBasis(unittest.TestCase):

    def test_make_basis(self):

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

        # Compare
        self.assertTrue(np.array_equal(basis_arr, spin_system.basis.arr))
        self.assertEqual(basis_dict, spin_system.basis.dict)

        # Larger test system
        isotopes = np.array(['1H', '1H', '1H', '1H', '1H', '1H', '1H'])

        # Compare the size of the basis set to a calculated value for different spin orders
        for max_so in range(1, isotopes.size):
            spin_system = SpinSystem(isotopes, max_spin_order=max_so)
            size = sum([math.comb(isotopes.size, k) * 3**k for k in range(max_so+1)])
            self.assertEqual(size, spin_system.basis.dim)

    def test_state_idx(self):

        # Test system
        isotopes = np.array(['1H', '1H'])
        spin_system = SpinSystem(isotopes, max_spin_order=1)

        # Test against known values
        self.assertEqual(state_idx(spin_system, (0,0)), 0)
        self.assertEqual(state_idx(spin_system, (0,1)), 1)
        self.assertEqual(state_idx(spin_system, (0,2)), 2)
        self.assertEqual(state_idx(spin_system, (0,3)), 3)
        self.assertEqual(state_idx(spin_system, (1,0)), 4)
        self.assertEqual(state_idx(spin_system, (2,0)), 5)
        self.assertEqual(state_idx(spin_system, (3,0)), 6)

    def test_str_to_op_def(self):

        # Test system
        isotopes = np.array(['1H'])
        spin_system = SpinSystem(isotopes)

        # Test against known values
        self.assertEqual(str_to_op_def(spin_system, ['E'], [0]), ([(0,)], [1]))
        self.assertEqual(str_to_op_def(spin_system, ['I_+'], [0]), ([(1,)], [-math.sqrt(2)]))
        self.assertEqual(str_to_op_def(spin_system, ['I_z'], [0]), ([(2,)], [1]))
        self.assertEqual(str_to_op_def(spin_system, ['I_-'], [0]), ([(3,)], [math.sqrt(2)]))
        self.assertEqual(str_to_op_def(spin_system, ['I_x'], [0]), ([(1,), (3,)], [-math.sqrt(2)/2, math.sqrt(2)/2]))
        self.assertEqual(str_to_op_def(spin_system, ['I_y'], [0]), ([(1,), (3,)], [-math.sqrt(2)/(2j), -math.sqrt(2)/(2j)]))
        self.assertEqual(str_to_op_def(spin_system, ['T_00'], [0]), ([(0,)], [1]))
        self.assertEqual(str_to_op_def(spin_system, ['T_11'], [0]), ([(1,)], [1]))
        self.assertEqual(str_to_op_def(spin_system, ['T_10'], [0]), ([(2,)], [1]))
        self.assertEqual(str_to_op_def(spin_system, ['T_1-1'], [0]), ([(3,)], [1]))

    def test_idx_conversions(self):

        # Convert back and forth and compare
        for l in range(10):
            for q in range(-l, l+1):
                self.assertEqual(idx_to_lq(lq_to_idx(l,q)), (l, q))    
        for idx in range(100):
            self.assertEqual(lq_to_idx(*idx_to_lq(idx)), idx)

    def test_coherence_order(self):

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

    def test_ZQ_basis(self):

        # Example system
        isotopes = np.array(['1H', '1H', '1H'])
        spin_system = SpinSystem(isotopes, max_spin_order=2)

        # Save the original basis
        basis_org = copy.copy(spin_system.basis)

        # Convert to ZQ basis
        ZQ_basis(spin_system)

        # Check that only the ZQ terms remain and that the index map is correct
        for op, idx in basis_org.dict.items():
            if op in spin_system.basis.dict:
                
                self.assertEqual(coherence_order(op), 0)
                self.assertEqual(spin_system.basis.ZQ_map[spin_system.basis.dict[op]], idx)

    def test_ZQ_filter(self):

        # Example systems
        isotopes = np.array(['1H', '1H', '1H'])
        spin_system = SpinSystem(isotopes, max_spin_order=2)
        spin_system_ZQ = SpinSystem(isotopes, max_spin_order=2)
        ZQ_basis(spin_system_ZQ)

        # Operators to test
        operators = ['E', 'I_x', 'I_y', 'I_z', 'I_+', 'I_-']

        # Try all possible combinations
        for i in operators:
            for j in operators:
                for k in operators:

                    # Create the superoperators
                    sop = superoperator(spin_system, [i, j, k], [0, 1, 2])
                    sop_ZQ = superoperator(spin_system_ZQ, [i, j, k], [0, 1, 2])

                    # Applying the ZQ-filter should result in same result
                    self.assertTrue(np.allclose(sop_ZQ.toarray(), ZQ_filter(spin_system_ZQ, sop).toarray()))