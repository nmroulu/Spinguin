import unittest
import numpy as np
import scipy.sparse as sp
from spinguin._core._operators import op_prod
from spinguin._core.basis import make_basis, truncate_basis_by_coherence
from spinguin._core._superoperators import sop_prod, sop_prod_ref, \
    sop_from_string, sop_T_coupled, sop_to_truncated_basis
from spinguin._core.la import cartesian_tensor_to_spherical_tensor

class TestSuperoperators(unittest.TestCase):

    def test_structure_coefficients(self):
        """
        Test creating a superoperator using the structure coefficients.
        """

        # Create a test spin system
        spins = np.array([1/2, 1])
        nspins = spins.shape[0]
        basis = make_basis(spins, nspins)
        dim = basis.shape[0]

        # Test all product operators from the basis set
        for i in range(int(2*spins[0]+1)):
            for j in range(int(2*spins[1]+1)):

                # Set current operator definition
                op_def_i = np.array([i, j])

                # Build left and right superoperator manually
                sop_L_ref = np.zeros((dim, dim), dtype=complex)
                sop_R_ref = np.zeros((dim, dim), dtype=complex)

                # Construct the operator
                op_i = op_prod(op_def_i, spins, include_unit=True, sparse=False)

                # Loop over the operator bras
                for j in range(dim):

                    # Construct the operator bra
                    op_def_j = basis[j]
                    op_j = op_prod(op_def_j, spins, include_unit=True,
                                   sparse=False)

                    # Loop over the kets
                    for k in range(dim):

                        # Construct the operator ket
                        op_def_k = basis[k]
                        op_k = op_prod(op_def_k, spins, include_unit=True,
                                       sparse=False)

                        # Calculate the elements
                        norm = np.sqrt((op_j.conj().T @ op_j).trace() * \
                                       (op_k.conj().T @ op_k).trace())
                        sop_L_ref[j, k] = (op_j.conj().T @ op_i @ op_k).trace()
                        sop_L_ref[j, k] = sop_L_ref[j, k] / norm
                        sop_R_ref[j, k] = (op_j.conj().T @ op_k @ op_i).trace()
                        sop_R_ref[j, k] = sop_R_ref[j, k] / norm

                # Build left and right superoperators using inbuilt function
                # that uses the structure coefficients
                sop_L_dense = sop_prod(op_def_i, basis, spins, "left",
                                       sparse=False)
                sop_R_dense = sop_prod(op_def_i, basis, spins, "right",
                                       sparse=False)
                sop_L_sparse = sop_prod(op_def_i, basis, spins, "left",
                                        sparse=True)
                sop_R_sparse = sop_prod(op_def_i, basis, spins, "right",
                                        sparse=True)

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
        test_systems = [
            np.array([1/2]),
            np.array([1/2, 1])
        ]

        # Test all systems
        for spins in test_systems:

            # Test all possible spin orders
            for max_so in range(1, spins.shape[0] + 1):

                # Create a basis set
                basis = make_basis(spins, max_so)

                # Test all possible operators
                for op in basis:

                    # Create reference superoperators using an "idiot-proof"
                    # function
                    sop_L_ref = sop_prod_ref(op, basis, spins, 'left')
                    sop_R_ref = sop_prod_ref(op, basis, spins, 'right')
                    sop_comm_ref = sop_prod_ref(op, basis, spins, 'comm')

                    # Create superoperators using the inbuilt function
                    sop_L_sparse = sop_prod(op, basis, spins, 'left',
                                            sparse=True)
                    sop_L_dense = sop_prod(op, basis, spins, 'left',
                                           sparse=False)
                    sop_R_sparse = sop_prod(op, basis, spins, 'right',
                                            sparse=True)
                    sop_R_dense = sop_prod(op, basis, spins, 'right',
                                           sparse=False)
                    sop_comm_sparse = sop_prod(op, basis, spins, 'comm',
                                               sparse=True)
                    sop_comm_dense = sop_prod(op, basis, spins, 'comm',
                                              sparse=False)

                    # Compare
                    self.assertTrue(np.allclose(sop_L_sparse.toarray(),
                                                sop_L_ref))
                    self.assertTrue(np.allclose(sop_L_dense, sop_L_ref))
                    self.assertTrue(np.allclose(sop_R_sparse.toarray(),
                                                sop_R_ref))
                    self.assertTrue(np.allclose(sop_R_dense, sop_R_ref))
                    self.assertTrue(np.allclose(sop_comm_sparse.toarray(),
                                                sop_comm_ref))
                    self.assertTrue(np.allclose(sop_comm_dense, sop_comm_ref))

    def test_sop_prod_cache(self):
        """
        Test caching behavior of the sop_prod function when the basis changes.
        """

        # Example system
        spins = np.array([1/2, 1/2, 1/2])
        
        # Define an operator to be created
        op_def = np.array([2, 0, 0])

        # Create the operator in original basis
        basis = make_basis(spins, spins.shape[0])
        Iz = sop_prod(op_def, basis, spins, side='comm', sparse=False)

        # Truncate the basis and create the operator again
        ZQ_basis, _ = truncate_basis_by_coherence(basis, coherence_orders=[0])
        Iz_ZQ = sop_prod(op_def, ZQ_basis, spins, side='comm', sparse=False)

        # Resulting shapes should be different
        self.assertNotEqual(Iz.shape, Iz_ZQ.shape)

    def test_sop_from_string(self):
        """
        Test creating the superoperator from a string against a few hard-coded
        cases.
        """

        # Example system
        spins = np.array([1/2, 1/2])
        nspins = spins.shape[0]
        basis = make_basis(spins, max_spin_order=nspins)

        # Test the sop_from_string function
        self.assertTrue(np.allclose(
            sop_from_string("I(z,0)", basis, spins, "left",
                            sparse=True).toarray(),
            sop_prod(np.array([2, 0]), basis, spins, "left",
                     sparse=True).toarray()))
        self.assertTrue(np.allclose(
            sop_from_string("I(z,0)", basis, spins, "right", sparse=False),
            sop_prod(np.array([2, 0]), basis, spins, "right", sparse=False)))
        self.assertTrue(np.allclose(
            sop_from_string("I(z,0)", basis, spins, "comm",
                            sparse=True).toarray(),
            sop_prod(np.array([2, 0]), basis, spins, "comm",
                     sparse=True).toarray()))
        self.assertTrue(np.allclose(
            sop_from_string("I(z,0) + I(z,1)", basis, spins, "comm",
                            sparse=True).toarray(),
            (sop_prod(np.array([2, 0]), basis, spins, "comm", sparse=True) \
            + sop_prod(np.array([0, 2]), basis, spins, "comm",
                       sparse=True)).toarray()))
        self.assertTrue(np.allclose(
            sop_from_string("I(+,0) * I(-,1)", basis, spins, "comm",
                            sparse=True).toarray(),
            -2 * sop_prod(np.array([1, 3]), basis, spins, "comm",
                          sparse=True).toarray()))
        self.assertTrue(np.allclose(
            sop_from_string("I(x,0) + I(x,1)", basis, spins, "comm",
                            sparse=True).toarray(),
            (
            - 1 / np.sqrt(2) * sop_prod(np.array([1, 0]), basis, spins, "comm",
                                        sparse=True) \
            + 1 / np.sqrt(2) * sop_prod(np.array([3, 0]), basis, spins, "comm",
                                        sparse=True) \
            - 1 / np.sqrt(2) * sop_prod(np.array([0, 1]), basis, spins, "comm",
                                        sparse=True) \
            + 1 / np.sqrt(2) * sop_prod(np.array([0, 3]), basis, spins, "comm",
                                        sparse=True)
            ).toarray()))
        
    def test_sop_T_coupled(self):
        """
        Test creating the Hamiltonian term using "Cartesian" superoperators and
        the coupled spherical tensor superoperator.
        """

        # Example system
        spins = np.array([1/2, 1/2])
        basis = make_basis(spins, max_spin_order=spins.shape[0])
        dim = basis.shape[0]

        # Make a random Cartesian interaction tensor
        A = np.random.rand(3, 3)

        # Cartesian spin operators
        I = np.array(['x', 'y', 'z'])
        
        # Perform the dot product manually
        left = np.zeros((dim, dim), dtype=complex)
        for i in range(A.shape[0]):
            for s in range(A.shape[1]):
                left += A[i, s] * sop_from_string(f"I({I[i]},0) * I({I[s]},1)",
                                                  basis, spins, 'comm',
                                                  sparse=False)

        # Convert A to spherical tensors
        A = cartesian_tensor_to_spherical_tensor(A)

        # Use spherical tensors
        right_dense = np.zeros((dim, dim), dtype=complex)
        right_sparse = sp.csc_array((dim, dim), dtype=complex)
        for l in range(0, 3):
            for q in range(-l, l + 1):
                right_dense += (-1)**(q) * A[(l, q)] * \
                    sop_T_coupled(basis, spins, l, -q, 0, 1, sparse=False)
                right_sparse += (-1)**(q) * A[(l, q)] * \
                    sop_T_coupled(basis, spins, l, -q, 0, 1, sparse=True)

        # Both conventions should give the same result
        self.assertTrue(np.allclose(left, right_dense))
        self.assertTrue(np.allclose(left, right_sparse.toarray()))

    def test_sop_to_truncated_basis(self):
        """
        Test the transformation of superoperators to truncated basis.
        """
        # Example system
        spins = np.array([1/2, 1/2, 1/2, 1/2, 1])
        max_spin_order = 4
        basis = make_basis(spins, max_spin_order)

        # Make a truncated basis containing only coherence orders -2, 0, and 1
        coherence_orders = [-2, 0, 1]
        basis_truncated, index_map = truncate_basis_by_coherence(
            basis, coherence_orders)

        # Operators to test
        operators = ['E', 'x', 'y', 'z', '+', '-']

        # Try all possible combinations
        for i in operators:
            if i == "E":
                op_i = "E"
            else:
                op_i = f"I({i}, 0)"

            for j in operators:
                if j == "E":
                    op_j = "E"
                else:
                    op_j = f"I({j}, 1)"

                for k in operators:
                    if k == "E":
                        op_k = "E"
                    else:
                        op_k = f"I({k}, 2)"

                    # Create the operator string
                    op_string = f"{op_i} * {op_j} * {op_k}"

                    # Create the superoperators in original and transform to
                    # truncated basis
                    sop = sop_from_string(op_string, basis, spins, side="comm",
                                          sparse=True)
                    sop = sop_to_truncated_basis(index_map, sop)

                    # Create the superoperator directly to the truncated basis
                    # for reference
                    sop_ref = sop_from_string(op_string, basis_truncated, spins,
                                              side="comm", sparse=True)

                    # Compare
                    self.assertTrue(np.allclose(sop.toarray(),
                                                sop_ref.toarray()))