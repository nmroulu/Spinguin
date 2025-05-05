import unittest
import numpy as np
import scipy.sparse as sp
from spinguin.system.spin_system import SpinSystem
from spinguin.qm.operators import op_prod
from spinguin.qm.superoperators import sop_prod, sop_prod_ref, superoperator, sop_T_coupled
from spinguin.utils.la import cartesian_tensor_to_spherical_tensor

class TestSuperoperators(unittest.TestCase):

    def test_structure_coefficients(self):
        """
        Test creating a superoperator using the structure coefficients.
        """

        # Create a test spin system
        isotopes = np.array(['1H', '14N'])
        spin_system = SpinSystem(isotopes)

        # Test all product operators from the basis set
        for i in range(int(2*spin_system.spins[0]+1)):
            for j in range(int(2*spin_system.spins[1]+1)):

                # Set current operator definition
                op_def_i = (i, j)

                # Build left and right superoperator manually
                sop_L_ref = np.zeros((spin_system.basis.dim, spin_system.basis.dim), dtype=complex)
                sop_R_ref = np.zeros((spin_system.basis.dim, spin_system.basis.dim), dtype=complex)

                # Construct the operator
                op_i = op_prod(op_def_i, spin_system.spins, include_unit=True, sparse=False)

                # Loop over the operator bras
                for j in range(spin_system.basis.dim):

                    # Construct the operator bra
                    op_def_j = spin_system.basis[j]
                    op_j = op_prod(op_def_j, spin_system.spins, include_unit=True, sparse=False)

                    # Loop over the kets
                    for k in range(spin_system.basis.dim):

                        # Construct the operator ket
                        op_def_k = spin_system.basis[k]
                        op_k = op_prod(op_def_k, spin_system.spins, include_unit=True, sparse=False)

                        # Calculate the elements
                        norm = np.sqrt((op_j.conj().T @ op_j).trace() * (op_k.conj().T @ op_k).trace())
                        sop_L_ref[j, k] = (op_j.conj().T @ op_i @ op_k).trace() / norm
                        sop_R_ref[j, k] = (op_j.conj().T @ op_k @ op_i).trace() / norm

                # Build left and right superoperators using inbuilt function
                # that uses the structure coefficients
                sop_L_dense = sop_prod(op_def_i, spin_system.basis, spin_system.spins, "left", sparse=False)
                sop_R_dense = sop_prod(op_def_i, spin_system.basis, spin_system.spins, "right", sparse=False)
                sop_L_sparse = sop_prod(op_def_i, spin_system.basis, spin_system.spins, "left", sparse=True)
                sop_R_sparse = sop_prod(op_def_i, spin_system.basis, spin_system.spins, "right", sparse=True)

                # Compare
                self.assertTrue(np.allclose(sop_L_dense, sop_L_ref))
                self.assertTrue(np.allclose(sop_R_dense, sop_R_ref))
                self.assertTrue(np.allclose(sop_L_sparse.toarray(), sop_L_ref))
                self.assertTrue(np.allclose(sop_R_sparse.toarray(), sop_R_ref))

    def test_sop_prod(self):
        """
        Test the construction of superoperators at varying truncated basis sets.
        """

        # Define test spin systems
        systems = []

        # Assign isotopes
        systems.append(np.array(['1H']))
        systems.append(np.array(['1H', '14N']))

        # Test all systems
        for isotopes in systems:

            # Test all possible spin orders
            for max_so in range(1, isotopes.size + 1):

                # Initialize the spin system
                spin_system = SpinSystem(isotopes, max_spin_order=max_so)

                # Test all possible operators
                for op in spin_system.basis.arr:

                    # Create reference superoperators using an "idiot-proof" function
                    sop_L_ref = sop_prod_ref(op, spin_system.basis, spin_system.spins, 'left')
                    sop_R_ref = sop_prod_ref(op, spin_system.basis, spin_system.spins, 'right')
                    sop_comm_ref = sop_prod_ref(op, spin_system.basis, spin_system.spins, 'comm')

                    # Create superoperators using the inbuilt function
                    sop_L_sparse = sop_prod(op, spin_system.basis, spin_system.spins, 'left', sparse=True)
                    sop_L_dense = sop_prod(op, spin_system.basis, spin_system.spins, 'left', sparse=False)
                    sop_R_sparse = sop_prod(op, spin_system.basis, spin_system.spins, 'right', sparse=True)
                    sop_R_dense = sop_prod(op, spin_system.basis, spin_system.spins, 'right', sparse=False)
                    sop_comm_sparse = sop_prod(op, spin_system.basis, spin_system.spins, 'comm', sparse=True)
                    sop_comm_dense = sop_prod(op, spin_system.basis, spin_system.spins, 'comm', sparse=False)

                    # Compare
                    self.assertTrue(np.allclose(sop_L_sparse.toarray(), sop_L_ref))
                    self.assertTrue(np.allclose(sop_L_dense, sop_L_ref))
                    self.assertTrue(np.allclose(sop_R_sparse.toarray(), sop_R_ref))
                    self.assertTrue(np.allclose(sop_R_dense, sop_R_ref))
                    self.assertTrue(np.allclose(sop_comm_sparse.toarray(), sop_comm_ref))
                    self.assertTrue(np.allclose(sop_comm_dense, sop_comm_ref))

    def test_sop_prod_cache(self):
        """
        Test caching behavior of the sop_prod function when the basis changes.
        """

        # Example system
        isotopes = np.array(['1H', '1H', '1H'])
        spin_system = SpinSystem(isotopes)

        # Create an operator, change the basis, and create an operator again
        op_def = (2, 0, 0)
        Iz = sop_prod(op_def, spin_system.basis, spin_system.spins, side='comm', sparse=False)
        spin_system.basis.truncate_by_coherence([0])
        Iz_ZQ = sop_prod(op_def, spin_system.basis, spin_system.spins, side='comm', sparse=False)

        # Resulting shapes should be different
        self.assertNotEqual(Iz.shape, Iz_ZQ.shape)

    def test_superoperator(self):
        """
        Test the superoperator function against a few hard-coded cases.
        """

        # Example system
        isotopes = np.array(['1H', '1H'])
        spin_system = SpinSystem(isotopes)

        # Test the superoperator function
        self.assertTrue(np.allclose(superoperator(spin_system, "I(z,0)", "left", sparse=True).toarray(),
                                    sop_prod((2, 0), spin_system.basis, spin_system.spins, "left", sparse=True).toarray()))
        self.assertTrue(np.allclose(superoperator(spin_system, "I(z,0)", "right", sparse=False),
                                    sop_prod((2, 0), spin_system.basis, spin_system.spins, "right", sparse=False)))
        self.assertTrue(np.allclose(superoperator(spin_system, "I(z,0)", "comm", sparse=True).toarray(),
                                    sop_prod((2, 0), spin_system.basis, spin_system.spins, "comm", sparse=True).toarray()))
        self.assertTrue(np.allclose(superoperator(spin_system, "I(z,0) + I(z,1)", "comm", sparse=True).toarray(),
                                    (sop_prod((2, 0), spin_system.basis, spin_system.spins, "comm", sparse=True) \
                                     + sop_prod((0, 2), spin_system.basis, spin_system.spins, "comm", sparse=True)).toarray()))
        self.assertTrue(np.allclose(superoperator(spin_system, "I(+,0) * I(-,1)", "comm", sparse=True).toarray(),
                                    -2 * sop_prod((1, 3), spin_system.basis, spin_system.spins, "comm", sparse=True).toarray()))
        self.assertTrue(np.allclose(superoperator(spin_system, "I(x,0) + I(x,1)", "comm", sparse=True).toarray(), (
                                    - 1 / np.sqrt(2) * sop_prod((1, 0), spin_system.basis, spin_system.spins, "comm", sparse=True) \
                                    + 1 / np.sqrt(2) * sop_prod((3, 0), spin_system.basis, spin_system.spins, "comm", sparse=True) \
                                    - 1 / np.sqrt(2) * sop_prod((0, 1), spin_system.basis, spin_system.spins, "comm", sparse=True) \
                                    + 1 / np.sqrt(2) * sop_prod((0, 3), spin_system.basis, spin_system.spins, "comm", sparse=True)
                                    ).toarray()))
        
    def test_sop_T_coupled(self):
        """
        Test creating the Hamiltonian term using "Cartesian" superoperators and the
        coupled spherical tensor superoperator.
        """

        # Example system
        isotopes = np.array(['1H', '1H'])
        spin_system = SpinSystem(isotopes)

        # Make a random Cartesian interaction tensor
        A = np.random.rand(3, 3)

        # Cartesian spin operators
        I = np.array(['x', 'y', 'z'])
        
        # Perform the dot product manually
        left = np.zeros((spin_system.basis.dim, spin_system.basis.dim), dtype=complex)
        for i in range(A.shape[0]):
            for s in range(A.shape[1]):
                left += A[i, s] * superoperator(spin_system, f"I({I[i]},0) * I({I[s]},1)", 'comm', sparse=False)

        # Convert A to spherical tensors
        A = cartesian_tensor_to_spherical_tensor(A)

        # Use spherical tensors
        right_dense = np.zeros((spin_system.basis.dim, spin_system.basis.dim), dtype=complex)
        right_sparse = sp.csc_array((spin_system.basis.dim, spin_system.basis.dim), dtype=complex)
        for l in range(0, 3):
            for q in range(-l, l + 1):
                right_dense += (-1)**(q) * A[(l, q)] * sop_T_coupled(spin_system.basis, spin_system.spins, l, -q, 0, 1, sparse=False)
                right_sparse += (-1)**(q) * A[(l, q)] * sop_T_coupled(spin_system.basis, spin_system.spins, l, -q, 0, 1, sparse=True)

        # Both conventions should give the same result
        self.assertTrue(np.allclose(left, right_dense))
        self.assertTrue(np.allclose(left, right_sparse.toarray()))