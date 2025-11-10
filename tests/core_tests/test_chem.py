import unittest
import numpy as np
import itertools
import spinguin as sg

class TestChemMethods(unittest.TestCase):

    def test_associate(self):
        """
        Test the associate function for combining spin systems.
        """
        # Parameters for this test
        sg.parameters.default()

        # Define three spin systems
        ss1 = sg.SpinSystem(['1H', '1H'])
        ss2 = sg.SpinSystem(['1H', '1H', '1H'])
        ss3 = sg.SpinSystem(['1H', '1H', '1H', '1H', '1H'])

        # Build the basis sets
        ss1.basis.max_spin_order = 2
        ss2.basis.max_spin_order = 3
        ss3.basis.max_spin_order = 5
        ss1.basis.build()
        ss2.basis.build()
        ss3.basis.build()

        # Define spin mappings after association
        map1 = np.array([0, 2])
        map2 = np.array([1, 3, 4])

        # Create alpha states with dense and sparse formalisms
        # Create also a reference state
        sg.parameters.sparse_state = False
        rho1_d = sg.alpha_state(ss1, 0)
        rho2_d = sg.alpha_state(ss2, 0)
        rho3_ref = sg.triplet_plus_state(ss3, 0, 1)
        sg.parameters.sparse_state = True
        rho1_s = sg.alpha_state(ss1, 0)
        rho2_s = sg.alpha_state(ss2, 0)

        # Test the associate function using a mixture of different formalisms
        sg.parameters.sparse_state = False
        rho3_ddd = sg.associate(ss1, ss2, ss3, rho1_d, rho2_d, map1, map2)
        rho3_dsd = sg.associate(ss1, ss2, ss3, rho1_s, rho2_d, map1, map2)
        rho3_dds = sg.associate(ss1, ss2, ss3, rho1_d, rho2_s, map1, map2)
        rho3_dss = sg.associate(ss1, ss2, ss3, rho1_s, rho2_s, map1, map2)
        self.assertTrue(np.allclose(rho3_ddd, rho3_ref))
        self.assertTrue(np.allclose(rho3_dsd, rho3_ref))
        self.assertTrue(np.allclose(rho3_dds, rho3_ref))
        self.assertTrue(np.allclose(rho3_dss, rho3_ref))
        sg.parameters.sparse_state = True
        rho3_sdd = sg.associate(ss1, ss2, ss3, rho1_d, rho2_d, map1, map2)
        rho3_ssd = sg.associate(ss1, ss2, ss3, rho1_s, rho2_d, map1, map2)
        rho3_sds = sg.associate(ss1, ss2, ss3, rho1_d, rho2_s, map1, map2)
        rho3_sss = sg.associate(ss1, ss2, ss3, rho1_s, rho2_s, map1, map2)
        self.assertTrue(np.allclose(rho3_sdd.toarray(), rho3_ref))
        self.assertTrue(np.allclose(rho3_ssd.toarray(), rho3_ref))
        self.assertTrue(np.allclose(rho3_sds.toarray(), rho3_ref))
        self.assertTrue(np.allclose(rho3_sss.toarray(), rho3_ref))

        # Test with basis sets of varying spin orders (using dense formalism)
        sg.parameters.sparse_state = False
        for max_spin_order1 in range(1, ss1.nspins+1):
            ss1.basis.max_spin_order = max_spin_order1
            ss1.basis.build()

            for max_spin_order2 in range(1, ss2.nspins+1):
                ss2.basis.max_spin_order = max_spin_order2
                ss2.basis.build()

                for max_spin_order3 in range(2, ss3.nspins+1):
                    ss3.basis.max_spin_order = max_spin_order3
                    ss3.basis.build()

                    # Create alpha states for the spin systems
                    rho1 = sg.alpha_state(ss1, 0)
                    rho2 = sg.alpha_state(ss2, 0)

                    # Perform association
                    rho3 = sg.associate(ss1, ss2, ss3, rho1, rho2, map1, map2)

                    # Create the expected state directly
                    rho3_ref = sg.triplet_plus_state(ss3, 0, 1)

                    # Compare
                    self.assertTrue(np.allclose(rho3_ref, rho3))

    def test_dissociate(self):
        """
        Test the dissociate function for splitting spin systems.
        """
        # Parameters for this test
        sg.parameters.default()

        # Define three spin systems
        ss1 = sg.SpinSystem(['1H', '1H'])
        ss2 = sg.SpinSystem(['1H', '1H', '1H'])
        ss3 = sg.SpinSystem(['1H', '1H', '1H', '1H', '1H'])

        # Build the basis sets
        ss1.basis.max_spin_order = 2
        ss2.basis.max_spin_order = 3
        ss3.basis.max_spin_order = 5
        ss1.basis.build()
        ss2.basis.build()
        ss3.basis.build()

        # Define spin mappings after dissociation
        map1 = np.array([0, 2])
        map2 = np.array([1, 3, 4])

        # Create triplet plus states with dense and sparse formalisms
        # Create also reference states
        sg.parameters.sparse_state = False
        rho3_d = sg.triplet_plus_state(ss3, 0, 1)
        rho1_ref = sg.alpha_state(ss1, 0)
        rho2_ref = sg.alpha_state(ss2, 0)
        sg.parameters.sparse_state = True
        rho3_s = sg.triplet_plus_state(ss3, 0, 1)

        # Test the dissociate function using a mixture of different formalisms
        sg.parameters.sparse_state = False
        rho1_dd, rho2_dd = sg.dissociate(ss1, ss2, ss3, rho3_d, map1, map2)
        rho1_ds, rho2_ds = sg.dissociate(ss1, ss2, ss3, rho3_s, map1, map2)
        self.assertTrue(np.allclose(rho1_dd, rho1_ref))
        self.assertTrue(np.allclose(rho2_dd, rho2_ref))
        self.assertTrue(np.allclose(rho1_ds, rho1_ref))
        self.assertTrue(np.allclose(rho2_ds, rho2_ref))
        sg.parameters.sparse_state = True
        rho1_sd, rho2_sd = sg.dissociate(ss1, ss2, ss3, rho3_d, map1, map2)
        rho1_ss, rho2_ss = sg.dissociate(ss1, ss2, ss3, rho3_s, map1, map2)
        self.assertTrue(np.allclose(rho1_sd.toarray(), rho1_ref))
        self.assertTrue(np.allclose(rho2_sd.toarray(), rho2_ref))
        self.assertTrue(np.allclose(rho1_ss.toarray(), rho1_ref))
        self.assertTrue(np.allclose(rho2_ss.toarray(), rho2_ref))

        # Test with basis sets of varying spin orders (using dense formalism)
        sg.parameters.sparse_state = False
        for max_spin_order1 in range(1, ss1.nspins+1):
            ss1.basis.max_spin_order = max_spin_order1
            ss1.basis.build()

            for max_spin_order2 in range(1, ss2.nspins+1):
                ss2.basis.max_spin_order = max_spin_order2
                ss2.basis.build()

                for max_spin_order3 in range(2, ss3.nspins+1):
                    ss3.basis.max_spin_order = max_spin_order3
                    ss3.basis.build()

                    # Create a triplet plus state for the combined system
                    rho3 = sg.triplet_plus_state(ss3, 0, 1)

                    # Perform dissociation
                    rho1, rho2 = sg.dissociate(ss1, ss2, ss3, rho3, map1, map2)

                    # Create the expected states directly
                    rho1_ref = sg.alpha_state(ss1, 0)
                    rho2_ref = sg.alpha_state(ss2, 0)

                    # Compare
                    self.assertTrue(np.allclose(rho1_ref, rho1))
                    self.assertTrue(np.allclose(rho2_ref, rho2))

    def test_permute_spins(self):
        """
        Test permuting spin indices.
        """
        # Reset parameters to defaults
        sg.parameters.default()

        # Define the spin system
        ss = sg.SpinSystem(["1H", "1H", "1H"])

        # Build the basis set
        ss.basis.max_spin_order = 3
        ss.basis.build()

        # Create the alpha state using dense and sparse formalisms
        sg.parameters.sparse_state = False
        rho_d = sg.alpha_state(ss, 0)
        sg.parameters.sparse_state = True
        rho_s = sg.alpha_state(ss, 0)

        # Test permuting using dense and sparse formalisms
        sg.parameters.sparse_state = False
        rho_perm_dd = sg.permute_spins(ss, rho_d, [1, 2, 0])
        rho_perm_ds = sg.permute_spins(ss, rho_s, [1, 2, 0])
        self.assertTrue(np.allclose(rho_perm_dd, rho_perm_ds))
        sg.parameters.sparse_state = True
        rho_perm_sd = sg.permute_spins(ss, rho_d, [1, 2, 0])
        rho_perm_ss = sg.permute_spins(ss, rho_s, [1, 2, 0])
        self.assertTrue(np.allclose(
            rho_perm_sd.toarray(),
            rho_perm_ss.toarray()
        ))

        # Get all possible permutations
        permutations = list(itertools.permutations([0, 1, 2]))

        # Test with various basis sets (using the dense formalism)
        sg.parameters.sparse_state = False
        for max_so in range(1, ss.nspins+1):

            # Create the basis set
            ss.basis.max_spin_order = max_so
            ss.basis.build()

            # Create an alpha state for the first spin
            rho = sg.alpha_state(ss, 0)

            # Go through the permutations
            for perm in permutations:

                # Convert to NumPy array
                perm = np.array(perm)

                # Permute the original state
                rho_perm = sg.permute_spins(ss, rho, perm)

                # Find where the first spin is going to be mapped
                idx = perm[0]

                # Create reference state
                rho_ref = sg.alpha_state(ss, idx)

                # Compare
                self.assertTrue(np.allclose(rho_ref, rho_perm))