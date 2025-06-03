import unittest
import numpy as np
import itertools
from spinguin.core.chem import dissociate, associate, permute_spins
from spinguin.core.basis import make_basis
from spinguin.core.states import triplet_plus_state, alpha_state

class TestChemMethods(unittest.TestCase):

    def test_associate(self):
        """
        Test the associate function for combining spin systems.
        """

        # Define spin quantum numbers
        spins1 = np.array([1/2, 1/2])
        spins2 = np.array([1/2, 1/2, 1/2])
        spins3 = np.array([1/2, 1/2, 1/2, 1/2, 1/2])

        # Acquire the number of spins
        nspins1 = spins1.shape[0]
        nspins2 = spins2.shape[0]
        nspins3 = spins3.shape[0]

        # Define spin mappings after dissociation
        spinmap1 = np.array([0, 2])
        spinmap2 = np.array([1, 3, 4])

        # Test with basis sets of varying spin orders
        for max_spin_order1 in range(1, nspins1+1):
            basis1 = make_basis(spins1, max_spin_order1)

            for max_spin_order2 in range(1, nspins2+1):
                basis2 = make_basis(spins2, max_spin_order2)

                for max_spin_order3 in range(2, nspins3+1):
                    basis3 = make_basis(spins3, max_spin_order3)

                    # Create alpha states for the spin systems
                    rho1_dense = alpha_state(basis1, spins1, 0, sparse=False)
                    rho1_sparse = alpha_state(basis1, spins1, 0, sparse=True)
                    rho2_dense = alpha_state(basis2, spins2, 0, sparse=False)
                    rho2_sparse = alpha_state(basis2, spins2, 0, sparse=True)

                    # Perform association
                    rho3_dense_dense = associate(basis1, basis2, basis3,
                                                 rho1_dense, rho2_dense,
                                                 spinmap1, spinmap2)
                    rho3_dense_sparse = associate(basis1, basis2, basis3,
                                                  rho1_dense, rho2_sparse,
                                                  spinmap1, spinmap2)
                    rho3_sparse_dense = associate(basis1, basis2, basis3,
                                                  rho1_sparse, rho2_dense,
                                                  spinmap1, spinmap2)
                    rho3_sparse_sparse = associate(basis1, basis2, basis3,
                                                   rho1_sparse, rho2_sparse,
                                                   spinmap1, spinmap2)

                    # Create the expected state directly
                    rho3_ref = triplet_plus_state(basis3, spins3, 0, 1,
                                                  sparse=False)

                    # Compare
                    self.assertTrue(np.allclose(rho3_ref, rho3_dense_dense))
                    self.assertTrue(np.allclose(rho3_ref, rho3_dense_sparse))
                    self.assertTrue(np.allclose(rho3_ref, rho3_sparse_dense))
                    self.assertTrue(np.allclose(rho3_ref,
                                                rho3_sparse_sparse.toarray()))

    def test_dissociate(self):
        """
        Test the dissociate function for splitting spin systems.
        """

        # Define spin quantum numbers
        spins1 = np.array([1/2, 1/2])
        spins2 = np.array([1/2, 1/2, 1/2])
        spins3 = np.array([1/2, 1/2, 1/2, 1/2, 1/2])

        # Acquire the number of spins
        nspins1 = spins1.shape[0]
        nspins2 = spins2.shape[0]
        nspins3 = spins3.shape[0]

        # Define spin mappings after dissociation
        spinmap1 = np.array([0, 2])
        spinmap2 = np.array([1, 3, 4])

        # Test with basis sets of varying spin orders
        for max_spin_order1 in range(1, nspins1+1):
            basis1 = make_basis(spins1, max_spin_order1)

            for max_spin_order2 in range(1, nspins2+1):
                basis2 = make_basis(spins2, max_spin_order2)

                for max_spin_order3 in range(2, nspins3+1):
                    basis3 = make_basis(spins3, max_spin_order3)

                    # Create a triplet plus state for the combined system
                    rho3_dense = triplet_plus_state(basis3, spins3, 0, 1,
                                                    sparse=False)
                    rho3_sparse = triplet_plus_state(basis3, spins3, 0, 1,
                                                     sparse=False)

                    # Perform dissociation
                    rho1_dense, rho2_dense = dissociate(
                        basis1, basis2, basis3, spins1, spins2, rho3_dense,
                        spinmap1, spinmap2)
                    rho1_sparse, rho2_sparse = dissociate(
                        basis1, basis2, basis3, spins1, spins2, rho3_sparse,
                        spinmap1, spinmap2)

                    # Create the expected states directly
                    rho1_ref = alpha_state(basis1, spins1, 0, sparse=False)
                    rho2_ref = alpha_state(basis2, spins2, 0, sparse=False)

                    # Compare
                    self.assertTrue(np.allclose(rho1_ref, rho1_dense))
                    self.assertTrue(np.allclose(rho1_ref, rho1_sparse))
                    self.assertTrue(np.allclose(rho2_ref, rho2_dense))
                    self.assertTrue(np.allclose(rho2_ref, rho2_sparse))

    def test_permute_spins(self):
        """
        Test permuting spin indices.
        """

        # Define the spin system
        spins = np.array([1/2, 1/2, 1/2])
        max_spin_order = 2
        basis = make_basis(spins, max_spin_order)

        # Create an alpha state for the first spin
        rho = alpha_state(basis, spins, 0, sparse=False)

        # Get all possible permutations
        permutations = itertools.permutations([0, 1, 2])

        # Go through the permutations
        for perm in permutations:

            # Convert to NumPy array
            perm = np.array(perm)

            # Permute the original state
            rho_perm = permute_spins(basis, rho, perm)

            # Find where the first spin is going to be mapped
            idx = perm[0]

            # Create reference state
            rho_ref = alpha_state(basis, spins, idx, sparse=False)

            # Compare
            self.assertTrue(np.allclose(rho_ref, rho_perm))