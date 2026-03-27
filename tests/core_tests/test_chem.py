"""
Tests for spin-system association, dissociation, and spin permutation.
"""

import itertools
import unittest

import numpy as np

import spinguin as sg


class TestChemMethods(unittest.TestCase):
    """
    Test chemical spin-system manipulation utilities.
    """

    def _build_spin_system(
        self,
        isotopes,
        max_spin_order,
    ):
        """
        Create a spin system and build its basis set.

        Parameters
        ----------
        isotopes : sequence of str
            Isotope labels of the spin system.
        max_spin_order : int
            Maximum spin order used to build the basis set.

        Returns
        -------
        SpinSystem
            Spin system with a built basis set.
        """

        # Create the spin system used in the test.
        spin_system = sg.SpinSystem(isotopes)

        # Build the basis set with the requested truncation level.
        spin_system.basis.max_spin_order = max_spin_order
        spin_system.basis.build()

        return spin_system

    def _build_association_systems(
        self,
    ):
        """
        Create the three spin systems used in association tests.

        Returns
        -------
        tuple
            Two subsystem spin systems, the combined spin system, and the two
            spin maps.
        """

        # Build the two subsystem spin systems and the combined system.
        spin_system_1 = self._build_spin_system(["1H", "1H"], 2)
        spin_system_2 = self._build_spin_system(["1H", "1H", "1H"], 3)
        spin_system_3 = self._build_spin_system(
            ["1H", "1H", "1H", "1H", "1H"],
            5,
        )

        # Define the spin mappings between the subsystems and the full system.
        spin_map_1 = np.array([0, 2])
        spin_map_2 = np.array([1, 3, 4])

        return spin_system_1, spin_system_2, spin_system_3, spin_map_1, spin_map_2

    def _as_dense_array(
        self,
        state,
    ):
        """
        Convert a state object to a dense NumPy array when needed.

        Parameters
        ----------
        state : array-like or sparse matrix
            State representation returned by Spinguin.

        Returns
        -------
        ndarray
            Dense array representation of the state.
        """

        # Convert sparse objects to dense arrays when necessary.
        if hasattr(state, "toarray"):
            return state.toarray()

        return state

    def _assert_state_matches_reference(
        self,
        state,
        reference,
    ):
        """
        Check that a state matches a dense reference state.

        Parameters
        ----------
        state : array-like or sparse matrix
            State to be tested.
        reference : ndarray
            Dense reference state.

        Returns
        -------
        None
            The assertion is evaluated in place.
        """

        # Compare the tested state with the dense reference representation.
        self.assertTrue(np.allclose(self._as_dense_array(state), reference))

    def test_associate(self):
        """
        Test the associate function for combining spin systems.
        """

        # Reset the global parameters before the test.
        sg.parameters.default()

        # Build the subsystem and combined-system test objects.
        ss1, ss2, ss3, map1, map2 = self._build_association_systems()

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
            self._assert_state_matches_reference(result, rho3_ref)

        # Test sparse output with all combinations of dense and sparse inputs.
        sg.parameters.sparse_state = True
        sparse_results = [
            sg.associate(ss1, ss2, ss3, rho1_d, rho2_d, map1, map2),
            sg.associate(ss1, ss2, ss3, rho1_s, rho2_d, map1, map2),
            sg.associate(ss1, ss2, ss3, rho1_d, rho2_s, map1, map2),
            sg.associate(ss1, ss2, ss3, rho1_s, rho2_s, map1, map2),
        ]
        for result in sparse_results:
            self._assert_state_matches_reference(result, rho3_ref)

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
        for max_spin_order1 in range(1, ss1.nspins + 1):
            ss1.basis.max_spin_order = max_spin_order1
            ss1.basis.build()

            for max_spin_order2 in range(1, ss2.nspins + 1):
                ss2.basis.max_spin_order = max_spin_order2
                ss2.basis.build()

                for max_spin_order3 in range(2, ss3.nspins + 1):
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
        ss1, ss2, ss3, map1, map2 = self._build_association_systems()

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
        self._assert_state_matches_reference(rho1_dd, rho1_ref)
        self._assert_state_matches_reference(rho2_dd, rho2_ref)
        self._assert_state_matches_reference(rho1_ds, rho1_ref)
        self._assert_state_matches_reference(rho2_ds, rho2_ref)

        # Test sparse output for dense and sparse combined-system inputs.
        sg.parameters.sparse_state = True
        rho1_sd, rho2_sd = sg.dissociate(ss1, ss2, ss3, rho3_d, map1, map2)
        rho1_ss, rho2_ss = sg.dissociate(ss1, ss2, ss3, rho3_s, map1, map2)
        self._assert_state_matches_reference(rho1_sd, rho1_ref)
        self._assert_state_matches_reference(rho2_sd, rho2_ref)
        self._assert_state_matches_reference(rho1_ss, rho1_ref)
        self._assert_state_matches_reference(rho2_ss, rho2_ref)

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
        for max_spin_order1 in range(1, ss1.nspins + 1):
            ss1.basis.max_spin_order = max_spin_order1
            ss1.basis.build()

            for max_spin_order2 in range(1, ss2.nspins + 1):
                ss2.basis.max_spin_order = max_spin_order2
                ss2.basis.build()

                for max_spin_order3 in range(2, ss3.nspins + 1):
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

        # Build the spin system used throughout the permutation test.
        ss = self._build_spin_system(["1H", "1H", "1H"], 3)

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
        for max_so in range(1, ss.nspins + 1):

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