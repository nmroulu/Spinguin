import unittest
import numpy as np
import math
import spinguin as sg

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
        spin_system = sg.SpinSystem(['1H', '1H'])
        
        # Create the basis
        spin_system.basis.max_spin_order = 1
        spin_system.basis.build()

        # Compare the generated basis with the hard-coded result
        self.assertTrue(np.array_equal(spin_system.basis.basis, basis_ref))

    def test_make_basis_2(self):
        """
        Test the creation of a basis set using large spin system.
        """

        # Large test system
        spin_system = sg.SpinSystem(['1H', '1H', '1H', '1H', '1H', '1H', '1H'])
        nspins = spin_system.nspins

        # Compare the dimension of the basis set to a reference value for
        # different spin orders
        for max_so in range(1, nspins):
            spin_system.basis.max_spin_order = max_so
            spin_system.basis.build()
            dim_ref = sum(
                [math.comb(nspins, k) * 3**k for k in range(max_so + 1)]
            )
            self.assertEqual(spin_system.basis.dim, dim_ref)

    def test_state_idx(self):
        """
        Test the mapping of states to their corresponding indices.
        """
        # Create a spin system
        spin_system = sg.SpinSystem(["14N", "1H", "1H"])

        # Create a basis set
        spin_system.basis.max_spin_order = 2
        spin_system.basis.build()

        # Test searching the index against hard-coded values with different
        # input types
        self.assertEqual(spin_system.basis.indexof([0, 1, 0]), 4)
        self.assertEqual(spin_system.basis.indexof((1, 0, 3)), 19)
        self.assertEqual(spin_system.basis.indexof(np.array([8, 0, 3])), 68)

        # Test searching for a non-existent state
        with self.assertRaises(ValueError):
            spin_system.basis.indexof([9, 9, 9])

        # Test searching state with too small dimension
        with self.assertRaises(ValueError):
            spin_system.basis.indexof([0])

        # Test searching state with too large dimension
        with self.assertRaises(ValueError):
            spin_system.basis.indexof([0, 1, 2, 0])

    def test_truncate_basis_by_coherence(self):
        """
        Test the creation of the truncated basis using coherence order as the
        selection criterion.
        """
        # Example system
        spin_system = sg.SpinSystem(['1H', '1H', '1H', '1H', '14N'])

        # Create a basis set with all coherence orders 
        spin_system.basis.max_spin_order = 4
        spin_system.basis.build()

        # Save the original basis set for further testing
        basis_org = spin_system.basis.basis.copy()

        # Create a superoperator and a state in the full basis set
        oper_org = sg.superoperator(spin_system, "I(z,0) * I(+,1) * I(-,2)")
        state_org = sg.state(spin_system, "I(+,1) * I(z,3) * I(-,4)")

        # Truncate the basis (retain coherence orders of -2, 0, and 1)
        # Obtain also the superoperator and the state in the truncated basis
        coherence_orders = [-2, 0, 1]
        oper_org_tr, state_org_tr = spin_system.basis.truncate_by_coherence(
            coherence_orders, oper_org, state_org
        )

        # Obtain the superoperator and state directly in the truncated basis
        oper_tr = sg.superoperator(spin_system, "I(z,0) * I(+,1) * I(-,2)")
        state_tr = sg.state(spin_system, "I(+,1) * I(z,3) * I(-,4)")

        # Verify that the superoperators and states are equal
        self.assertTrue(np.allclose(oper_org_tr.toarray(), oper_tr.toarray()))
        self.assertTrue(np.allclose(state_org_tr, state_tr))

        # Check that only the coherence orders [-2, 0, 1] remain in the basis
        for op_def in basis_org:

            # Should not raise an error
            if sg.coherence_order(op_def) in coherence_orders:
                spin_system.basis.indexof(op_def)

            # Raises an error
            else:
                with self.assertRaises(ValueError):
                    spin_system.basis.indexof(op_def)