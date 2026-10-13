"""
Tests for spin-system association, dissociation, and spin permutation.
"""

import itertools
import unittest

import numpy as np

import spinguin as sg
from ._helpers import build_spin_system


class TestChemMethods(unittest.TestCase):
    """
    Test chemical spin-system manipulation utilities.
    """

    def test_associate(self):
        """
        Test the associate function for combining spin systems.
        """

        # Reset the global parameters before the test.
        sg.parameters.default()

        # Build the subsystem and combined-system test objects.
        ss1 = build_spin_system(["1H", "1H"], 2)
        ss2 = build_spin_system(["1H", "1H", "1H"], 3)
        ss3 = build_spin_system(["1H", "1H", "1H", "1H", "1H"], 5)

        # Define the spin mappings between the subsystems and the full system.
        map1 = np.array([0, 2])
        map2 = np.array([1, 3, 4])

        # Create dense input states and the dense reference state.
        sg.parameters.sparse_state = False
        rho1_d = sg.alpha_state(ss1, 0)
        rho2_d = sg.alpha_state(ss2, 0)
        rho3_ref = sg.triplet_plus_state(ss3, 0, 1)

        # Create sparse input states for mixed-formalism checks.
        sg.parameters.sparse_state = True
        rho1_s = sg.alpha_state(ss1, 0)
        rho2_s = sg.alpha_state(ss2, 0)

        # Test dense output with all combinations of dense and sparse inputs.
        sg.parameters.sparse_state = False
        dense_results = [
            sg.associate(ss1, ss2, ss3, rho1_d, rho2_d, map1, map2),
            sg.associate(ss1, ss2, ss3, rho1_s, rho2_d, map1, map2),
            sg.associate(ss1, ss2, ss3, rho1_d, rho2_s, map1, map2),
            sg.associate(ss1, ss2, ss3, rho1_s, rho2_s, map1, map2),
        ]
        for result in dense_results:
            self.assertTrue(np.allclose(result, rho3_ref))

        # Test sparse output with all combinations of dense and sparse inputs.
        sparse_results = [
            sg.associate(ss1, ss2, ss3, rho1_d, rho2_d, map1, map2),
            sg.associate(ss1, ss2, ss3, rho1_s, rho2_d, map1, map2),
            sg.associate(ss1, ss2, ss3, rho1_d, rho2_s, map1, map2),
            sg.associate(ss1, ss2, ss3, rho1_s, rho2_s, map1, map2),
        ]
        for result in sparse_results:
            self.assertTrue(np.allclose(result, rho3_ref))

        # Check that invalid spin maps raise errors.
        map1_w = [0, 2, 3]
        with self.assertRaises(ValueError):
            sg.associate(ss1, ss2, ss3, rho1_d, rho2_d, map1_w, map2)
        map2_w = [1, 3]
        with self.assertRaises(ValueError):
            sg.associate(ss1, ss2, ss3, rho1_d, rho2_d, map1, map2_w)
        map1_w = [0, 5]
        with self.assertRaises(ValueError):
            sg.associate(ss1, ss2, ss3, rho1_d, rho2_d, map1_w, map2)

        # Check dense association for multiple basis-set truncation levels.
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

                    # Create the subsystem states used in the association.
                    rho1 = sg.alpha_state(ss1, 0)
                    rho2 = sg.alpha_state(ss2, 0)

                    # Associate the subsystem states into the combined system.
                    rho3 = sg.associate(ss1, ss2, ss3, rho1, rho2, map1, map2)

                    # Create the reference state directly in the full system.
                    rho3_ref = sg.triplet_plus_state(ss3, 0, 1)

                    # Compare the associated state with the reference state.
                    self.assertTrue(np.allclose(rho3_ref, rho3))

    def test_dissociate(self):
        """
        Test the dissociate function for splitting spin systems.
        """

        # Reset the global parameters before the test.
        sg.parameters.default()

        # Build the subsystem and combined-system test objects.
        ss1 = build_spin_system(["1H", "1H"], 2)
        ss2 = build_spin_system(["1H", "1H", "1H"], 3)
        ss3 = build_spin_system(["1H", "1H", "1H", "1H", "1H"], 5)

        # Define the spin mappings between the subsystems and the full system.
        map1 = np.array([0, 2])
        map2 = np.array([1, 3, 4])

        # Create the dense combined-system state and dense references.
        sg.parameters.sparse_state = False
        rho3_d = sg.triplet_plus_state(ss3, 0, 1)
        rho1_ref = sg.alpha_state(ss1, 0)
        rho2_ref = sg.alpha_state(ss2, 0)

        # Create the sparse combined-system state for mixed checks.
        sg.parameters.sparse_state = True
        rho3_s = sg.triplet_plus_state(ss3, 0, 1)

        # Test dense output for dense and sparse combined-system inputs.
        sg.parameters.sparse_state = False
        rho1_dd, rho2_dd = sg.dissociate(ss1, ss2, ss3, rho3_d, map1, map2)
        rho1_ds, rho2_ds = sg.dissociate(ss1, ss2, ss3, rho3_s, map1, map2)
        self.assertTrue(np.allclose(rho1_dd, rho1_ref))
        self.assertTrue(np.allclose(rho2_dd, rho2_ref))
        self.assertTrue(np.allclose(rho1_ds, rho1_ref))
        self.assertTrue(np.allclose(rho2_ds, rho2_ref))

        # Test sparse output for dense and sparse combined-system inputs.
        sg.parameters.sparse_state = True
        rho1_sd, rho2_sd = sg.dissociate(ss1, ss2, ss3, rho3_d, map1, map2)
        rho1_ss, rho2_ss = sg.dissociate(ss1, ss2, ss3, rho3_s, map1, map2)
        self.assertTrue(np.allclose(rho1_sd.toarray(), rho1_ref))
        self.assertTrue(np.allclose(rho2_sd.toarray(), rho2_ref))
        self.assertTrue(np.allclose(rho1_ss.toarray(), rho1_ref))
        self.assertTrue(np.allclose(rho2_ss.toarray(), rho2_ref))

        # Check that invalid spin maps raise errors.
        map1_w = [0, 2, 3]
        with self.assertRaises(ValueError):
            sg.dissociate(ss1, ss2, ss3, rho3_d, map1_w, map2)
        map2_w = [1, 3]
        with self.assertRaises(ValueError):
            sg.dissociate(ss1, ss2, ss3, rho3_d, map1, map2_w)
        map1_w = [0, 5]
        with self.assertRaises(ValueError):
            sg.dissociate(ss1, ss2, ss3, rho3_d, map1_w, map2)

        # Check dense dissociation for multiple basis-set truncation levels.
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

                    # Create the combined-system state to be dissociated.
                    rho3 = sg.triplet_plus_state(ss3, 0, 1)

                    # Dissociate the combined state into subsystem states.
                    rho1, rho2 = sg.dissociate(ss1, ss2, ss3, rho3, map1, map2)

                    # Create the subsystem reference states directly.
                    rho1_ref = sg.alpha_state(ss1, 0)
                    rho2_ref = sg.alpha_state(ss2, 0)

                    # Compare the dissociated states with the references.
                    self.assertTrue(np.allclose(rho1_ref, rho1))
                    self.assertTrue(np.allclose(rho2_ref, rho2))

    def test_permute_spins(self):
        """
        Test permuting spin indices.
        """

        # Reset the global parameters before the test.
        sg.parameters.default()

        # Define the spin system
        ss = build_spin_system(["1H", "1H", "1H"], 3)

        # Create the alpha state using dense and sparse formalisms.
        sg.parameters.sparse_state = False
        rho_d = sg.alpha_state(ss, 0)
        sg.parameters.sparse_state = True
        rho_s = sg.alpha_state(ss, 0)

        # Test dense output for dense and sparse input states.
        sg.parameters.sparse_state = False
        rho_perm_dd = sg.permute_spins(ss, rho_d, [1, 2, 0])
        rho_perm_ds = sg.permute_spins(ss, rho_s, [1, 2, 0])
        self.assertTrue(np.allclose(rho_perm_dd, rho_perm_ds))

        # Test sparse output for dense and sparse input states.
        sg.parameters.sparse_state = True
        rho_perm_sd = sg.permute_spins(ss, rho_d, [1, 2, 0])
        rho_perm_ss = sg.permute_spins(ss, rho_s, [1, 2, 0])
        self.assertTrue(
            np.allclose(rho_perm_sd.toarray(), rho_perm_ss.toarray())
        )

        # Check that invalid permutation maps raise errors.
        with self.assertRaises(ValueError):
            sg.permute_spins(ss, rho_d, [0, 1])
        with self.assertRaises(ValueError):
            sg.permute_spins(ss, rho_d, [0, 1, 1])
        with self.assertRaises(ValueError):
            sg.permute_spins(ss, rho_d, [0, 1, 3])

        # Generate all possible spin permutations.
        permutations = list(itertools.permutations([0, 1, 2]))

        # Check dense permutation for multiple basis-set truncation levels.
        sg.parameters.sparse_state = False
        for max_so in range(1, ss.nspins+1):

            # Rebuild the basis set with the current spin-order limit.
            ss.basis.max_spin_order = max_so
            ss.basis.build()

            # Create an alpha state for the first spin.
            rho = sg.alpha_state(ss, 0)

            # Test the state under every permutation of the spin labels.
            for perm in permutations:

                # Convert the permutation to the expected array format.
                perm = np.array(perm)

                # Apply the permutation to the state.
                rho_perm = sg.permute_spins(ss, rho, perm)

                # Determine where the first spin is mapped by the permutation.
                idx = perm[0]

                # Construct the corresponding reference state directly.
                rho_ref = sg.alpha_state(ss, idx)

                # Compare the permuted and reference states.
                self.assertTrue(np.allclose(rho_ref, rho_perm))

    def test_elementary_reaction_1(self):
        """
        Test the elementary reaction with an association reaction.
        """

        # Reset the global parameters before the test.
        sg.parameters.default()

        # Build the subsystem and combined-system test objects.
        ss1 = sg.SpinSystem(["1H", "1H"])
        ss2 = sg.SpinSystem(["1H", "1H", "1H"])
        ss3 = sg.SpinSystem(["1H", "1H", "1H", "1H", "1H"])

        # Define the spin map between the subsystems and the full system.
        spin_map = {
            (0, 0): (0, 0),
            (0, 1): (0, 2),
            (1, 0): (0, 1),
            (1, 1): (0, 3),
            (1, 2): (0, 4)
        }

        # Test the elementary reaction for dense and sparse backends
        for sparse in [False, True]:
            sg.parameters.sparse_state = sparse

            # Test with various spin order combinations
            for max_spin_order1 in range(1, ss1.nspins+1):
                ss1.basis.max_spin_order = max_spin_order1
                ss1.basis.build()

                for max_spin_order2 in range(1, ss2.nspins+1):
                    ss2.basis.max_spin_order = max_spin_order2
                    ss2.basis.build()

                    for max_spin_order3 in range(2, ss3.nspins+1):
                        ss3.basis.max_spin_order = max_spin_order3
                        ss3.basis.build()

                        # Create the input and reference states
                        rho1 = sg.alpha_state(ss1, 0)
                        rho2 = sg.alpha_state(ss2, 0)
                        rho3_ref = sg.triplet_plus_state(ss3, 0, 1)

                        # Perform association using elementary reaction
                        rho3 = sg.elementary_reaction(
                            reactant_systems=[ss1, ss2],
                            product_systems=ss3,
                            reactant_rhos=[rho1, rho2],
                            spin_map=spin_map
                        )

                        # Convert to dense before comparison if required
                        if sparse:
                            rho3 = rho3.toarray()
                            rho3_ref = rho3_ref.toarray()

                        # Compare
                        self.assertTrue(np.allclose(rho3, rho3_ref))

    def test_elementary_reaction_2(self):
        """
        Test the elementary reaction with a dissociation reaction.
        """

        # Reset the global parameters before the test.
        sg.parameters.default()

        # Build the subsystem and combined-system test objects.
        ss1 = sg.SpinSystem(["1H", "1H"])
        ss2 = sg.SpinSystem(["1H", "1H", "1H"])
        ss3 = sg.SpinSystem(["1H", "1H", "1H", "1H", "1H"])

        # Define the spin map between the subsystems and the full system.
        spin_map = {
            (0, 0): (0, 0),
            (0, 2): (0, 1),
            (0, 1): (1, 0),
            (0, 3): (1, 1),
            (0, 4): (1, 2)
        }

        # Test the elementary reaction for dense and sparse backends
        for sparse in [False, True]:
            sg.parameters.sparse_state = sparse

            # Test with various spin order combinations
            for max_spin_order1 in range(1, ss1.nspins+1):
                ss1.basis.max_spin_order = max_spin_order1
                ss1.basis.build()

                for max_spin_order2 in range(1, ss2.nspins+1):
                    ss2.basis.max_spin_order = max_spin_order2
                    ss2.basis.build()

                    for max_spin_order3 in range(2, ss3.nspins+1):
                        ss3.basis.max_spin_order = max_spin_order3
                        ss3.basis.build()

                        # Create the input and reference states
                        rho1_ref = sg.alpha_state(ss1, 0)
                        rho2_ref = sg.alpha_state(ss2, 0)
                        rho3 = sg.triplet_plus_state(ss3, 0, 1)

                        # Perform dissociation using elementary reaction
                        rho1, rho2 = sg.elementary_reaction(
                            reactant_systems=ss3,
                            product_systems=[ss1, ss2],
                            reactant_rhos=rho3,
                            spin_map=spin_map
                        )

                        # Convert to dense before comparison if required
                        if sparse:
                            rho1 = rho1.toarray()
                            rho2 = rho2.toarray()
                            rho1_ref = rho1_ref.toarray()
                            rho2_ref = rho2_ref.toarray()

                        # Compare
                        self.assertTrue(np.allclose(rho1, rho1_ref))
                        self.assertTrue(np.allclose(rho2, rho2_ref))

    def test_elementary_reaction_3(self):
        """
        Test the elementary reaction with a reaction A + B -> C + D.
        """

        # Reset the global parameters before the test
        sg.parameters.default()

        # Build the subsystem and combined-system test objects
        ss1 = sg.SpinSystem(["1H", "1H"])
        ss2 = sg.SpinSystem(["1H", "1H", "1H"])
        ss3 = sg.SpinSystem(["1H", "1H", "1H"])
        ss4 = sg.SpinSystem(["1H", "1H"])

        # Define the spin map
        spin_map = {
            (0, 0): (0, 0),  # A(0) -> C(0)
            (0, 1): (1, 0),  # A(1) -> D(0)
            (1, 0): (0, 1),  # B(0) -> C(1)
            (1, 1): (0, 2),  # B(1) -> C(2)
            (1, 2): (1, 1)   # B(2) -> D(1)
        }

        # Test the elementary reaction for dense and sparse backends
        for sparse in [False, True]:
            sg.parameters.sparse_state = sparse

            # Test with various spin order combinations.
            for max_spin_order1 in range(1, ss1.nspins+1):
                ss1.basis.max_spin_order = max_spin_order1
                ss1.basis.build()

                for max_spin_order2 in range(2, ss2.nspins+1):
                    ss2.basis.max_spin_order = max_spin_order2
                    ss2.basis.build()

                    for max_spin_order3 in range(2, ss3.nspins+1):
                        ss3.basis.max_spin_order = max_spin_order3
                        ss3.basis.build()

                        for max_spin_order4 in range(1, ss4.nspins+1):
                            ss4.basis.max_spin_order = max_spin_order4
                            ss4.basis.build()

                            # Create the input states
                            rho1 = sg.alpha_state(ss1, 0)
                            rho2 = sg.triplet_plus_state(ss2, 0, 2)

                            # Create the reference product states
                            rho3_ref = sg.triplet_plus_state(ss3, 0, 1)
                            rho4_ref = sg.alpha_state(ss4, 1)

                            # Perform elementary reaction
                            rho3, rho4 = sg.elementary_reaction(
                                reactant_systems=[ss1, ss2],
                                product_systems=[ss3, ss4],
                                reactant_rhos=[rho1, rho2],
                                spin_map=spin_map
                            )

                            # Convert to dense before comparison if required
                            if sparse:
                                rho3 = rho3.toarray()
                                rho4 = rho4.toarray()
                                rho3_ref = rho3_ref.toarray()
                                rho4_ref = rho4_ref.toarray()

                            # Compare
                            self.assertTrue(np.allclose(rho3, rho3_ref))
                            self.assertTrue(np.allclose(rho4, rho4_ref))

    def test_elementary_reaction_4(self):
        """
        Test the elementary reaction error handling for invalid mappings.
        """
        # Reset the global parameters before the test.
        sg.parameters.default()

        # Build spin systems for the tests
        ss1 = build_spin_system(["1H", "1H"], 2)
        ss2 = build_spin_system(["1H", "1H", "1H"], 3)

        # Create an example input state
        rho1 = sg.alpha_state(ss1, 0)

        # Error messages to be expected
        spin_mismatch = "The number of reactant and product spins do not match"
        missing_reactants = "Not every reactant spin is mapped."
        missing_prodcuts = "Not every product spin is mapped."

        # Spin map with spin count mismatch
        spin_map= {
            (0, 0): (0, 0),
            (0, 1): (0, 1)
        }
        with self.assertRaisesRegex(ValueError, spin_mismatch):
            sg.elementary_reaction(
                reactant_systems=ss1,
                product_systems=ss2,
                reactant_rhos=rho1,
                spin_map=spin_map
            )

        # Missing reactant spin in the map
        spin_map = {
            (0, 0): (0, 0)
        }
        with self.assertRaisesRegex(ValueError, missing_reactants):
            sg.elementary_reaction(
                reactant_systems=ss1,
                product_systems=ss1,
                reactant_rhos=rho1,
                spin_map=spin_map
            )

        # Reactant spin is out of bounds
        spin_map = {
            (0, 0): (0, 0),
            (0, 2): (0, 1)
        }
        with self.assertRaisesRegex(ValueError, missing_reactants):
            sg.elementary_reaction(
                reactant_systems=ss1,
                product_systems=ss1,
                reactant_rhos=rho1,
                spin_map=spin_map
            )

        # Overlap in product spins
        spin_map = {
            (0, 0): (0, 0),
            (0, 1): (0, 0)
        }
        with self.assertRaisesRegex(ValueError, missing_prodcuts):
            sg.elementary_reaction(
                reactant_systems=ss1,
                product_systems=ss1,
                reactant_rhos=rho1,
                spin_map=spin_map
            )

        # Product spin is out of bounds
        spin_map = {
            (0, 0): (0, 0),
            (0, 1): (0, 2)
        }
        with self.assertRaisesRegex(ValueError, missing_prodcuts):
            sg.elementary_reaction(
                reactant_systems=ss1,
                product_systems=ss1,
                reactant_rhos=rho1,
                spin_map=spin_map
            )