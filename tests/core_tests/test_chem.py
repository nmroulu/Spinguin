"""
Tests for spin-system association, dissociation, spin permutation, and
linear chemical exchange.
"""

import itertools
import unittest

import numpy as np
import scipy.linalg as la

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

class TestLinearExchange(unittest.TestCase):
    """
    Test linear chemical exchange between sites with identical bases against
    analytic results and the Bloch-McConnell equations.
    """

    def _single_spin_sites(
        self,
        chemical_shifts: list[float],
        T1: list[float],
        T2: list[float],
    ) -> list[sg.SpinSystem]:
        """
        Build single-spin 1H sites with phenomenological relaxation.

        Parameters
        ----------
        chemical_shifts : list of float
            Chemical shift of each site in ppm.
        T1 : list of float
            Longitudinal relaxation time of each site in s.
        T2 : list of float
            Transverse relaxation time of each site in s.

        Returns
        -------
        spin_systems : list of SpinSystem
            Spin systems of the sites.
        """

        # Build one spin system per site.
        spin_systems = []
        for delta, T1_site, T2_site in zip(chemical_shifts, T1, T2):
            ss = build_spin_system(["1H"], 1)
            ss.chemical_shifts = [delta]
            ss.relaxation.theory = "phenomenological"
            ss.relaxation.T1 = [T1_site]
            ss.relaxation.T2 = [T2_site]
            spin_systems.append(ss)

        return spin_systems

    def test_equilibrium_populations(self):
        """
        Test the equilibrium concentrations and populations of the binding
        equilibrium A + B <-> AB.
        """

        # Define the association constant and the total concentrations.
        K_a = 1e3
        c_A0 = 1e-3
        c_B0 = 2e-3

        # Check the mass balance and the association constant.
        c_A, c_B, c_AB = sg.equilibrium_concentrations(K_a, c_A0, c_B0)
        self.assertAlmostEqual(c_A + c_AB, c_A0)
        self.assertAlmostEqual(c_B + c_AB, c_B0)
        self.assertAlmostEqual(c_AB / (c_A*c_B), K_a)

        # Check that the populations are normalised and correctly ordered.
        p_A, p_AB = sg.equilibrium_two_site_populations(K_a, c_A0, c_B0)
        self.assertAlmostEqual(p_A + p_AB, 1)
        self.assertAlmostEqual(p_AB / p_A, c_AB / c_A)

    def test_two_site_exchange_matrix(self):
        """
        Test the two-site exchange matrix.
        """

        # Construct the exchange matrix.
        k_ex = 25.0
        populations = np.array([0.3, 0.7])
        K = sg.two_site_exchange_matrix(k_ex, populations)

        # The total population is conserved and the equilibrium is stationary.
        self.assertTrue(np.allclose(K.sum(axis=0), 0))
        self.assertTrue(np.allclose(K @ populations, 0))

        # The forward and backward rate constants sum to the exchange rate.
        self.assertAlmostEqual(K[1, 0] + K[0, 1], k_ex)

    def test_exchange_validation(self):
        """
        Test that invalid exchanging sites and exchange matrices are rejected.
        """

        # Reset the global parameters before the test.
        sg.parameters.default()
        sg.parameters.magnetic_field = 1.0

        # Build two compatible sites and their Liouvillians.
        sites = [build_spin_system(["1H"], 1), build_spin_system(["1H"], 1)]
        Ls = [sg.liouvillian(sg.hamiltonian(ss)) for ss in sites]
        K = sg.two_site_exchange_matrix(10.0, [0.5, 0.5])

        # Sites with different isotopes are rejected.
        mixed = [sites[0], build_spin_system(["13C"], 1)]
        with self.assertRaises(ValueError):
            sg.exchange_liouvillian(mixed, Ls, K)

        # Sites with different bases are rejected.
        truncated = [
            build_spin_system(["1H", "1H"], 2),
            build_spin_system(["1H", "1H"], 1),
        ]
        with self.assertRaises(ValueError):
            sg.exchange_state(
                truncated,
                [sg.unit_state(ss) for ss in truncated],
                [0.5, 0.5],
            )

        # An exchange matrix of the wrong shape is rejected.
        with self.assertRaises(ValueError):
            sg.exchange_liouvillian(sites, Ls, np.zeros((3, 3)))

        # An exchange matrix that does not conserve populations is rejected.
        with self.assertRaises(ValueError):
            sg.exchange_liouvillian(sites, Ls, np.array([[-1, 0], [1, 0.5]]))

        # Negative rate constants are rejected.
        with self.assertRaises(ValueError):
            sg.exchange_liouvillian(sites, Ls, np.array([[1, -1], [-1, 1]]))

        # A mismatched number of Liouvillians is rejected.
        with self.assertRaises(ValueError):
            sg.exchange_liouvillian(sites, Ls[:1], K)

    def test_exchange_equilibrium_and_kinetics(self):
        """
        Test that the population-weighted thermal equilibrium is stationary
        under exchange with thermalised relaxation, and that the site
        populations are conserved during the evolution.
        """

        # Set the global parameters used in the test.
        sg.parameters.default()
        sg.parameters.magnetic_field = 1.0
        sg.parameters.temperature = 295

        # Define three-site kinetics that satisfy detailed balance, with the
        # rate constants k[i, j] = F[i, j] / p[j] from symmetric fluxes F.
        populations = np.array([0.5, 0.3, 0.2])
        F = np.array([[0, 1.5, 1.0], [1.5, 0, 1.2], [1.0, 1.2, 0]])
        k = F / populations
        K = k - np.diag(k.sum(axis=0))

        # Test with dense and sparse backends.
        for sparse in [False, True]:
            sg.parameters.sparse_superoperator = sparse
            sg.parameters.sparse_state = sparse

            # Build J-coupled two-spin sites with different chemical shifts.
            sites = []
            for shifts in [[1.0, 3.0], [1.5, 3.2], [0.8, 2.5]]:
                ss = build_spin_system(["1H", "1H"], 2)
                ss.chemical_shifts = shifts
                ss.J_couplings = [[0, 0], [7.0, 0]]
                ss.relaxation.theory = "phenomenological"
                ss.relaxation.T1 = [2.0, 1.5]
                ss.relaxation.T2 = [0.8, 0.5]
                sites.append(ss)

            # Build the Liouvillian of the exchanging ensemble.
            Ls = [
                sg.liouvillian(sg.hamiltonian(ss), sg.relaxation(ss))
                for ss in sites
            ]
            L = sg.exchange_liouvillian(sites, Ls, K)

            # The weighted thermal equilibrium must be stationary, up to the
            # small differences between the thermal polarisations of the sites.
            rho_eq = sg.exchange_state(
                sites,
                [sg.equilibrium_state(ss) for ss in sites],
                populations,
            )
            rho_t = sg.propagator(L, 3.0) @ rho_eq
            for ss, rho_eq_s, rho_t_s in zip(
                sites,
                sg.split_exchange_state(sites, rho_eq),
                sg.split_exchange_state(sites, rho_t),
            ):
                Mz_eq = sg.measure(ss, rho_eq_s, "I(z)")
                Mz_t = sg.measure(ss, rho_t_s, "I(z)")
                self.assertLess(abs(Mz_t / Mz_eq - 1), 1e-5)

            # The site populations must be conserved for a pulsed state.
            rho_0 = sg.exchange_state(
                sites,
                [sg.pulse(ss, "I(y,0)", 90) @ sg.equilibrium_state(ss)
                 for ss in sites],
                populations,
            )
            rho_t = sg.propagator(L, 0.05) @ rho_0
            for p, ss, rho_t_s in zip(
                populations,
                sites,
                sg.split_exchange_state(sites, rho_t),
            ):
                self.assertAlmostEqual(sg.measure(ss, rho_t_s, "").real, p)

    def test_exchange_bloch_mcconnell_fid(self):
        """
        Test the site-resolved transverse signal of two-site exchange against
        the exact solution of the Bloch-McConnell equations.
        """

        # Set the global parameters used in the test.
        sg.parameters.default()
        sg.parameters.magnetic_field = 1.0
        sg.parameters.temperature = 295

        # Define the sites and the exchange kinetics.
        deltas = [0.0, 2.0]
        T2 = np.array([0.5, 0.2])
        populations = sg.equilibrium_two_site_populations(1e3, 1e-3, 2e-3)

        # Test with dense and sparse backends.
        for sparse in [False, True]:
            sg.parameters.sparse_superoperator = sparse

            # Test slow, intermediate and fast exchange.
            for k_ex in [30.0, 500.0, 2e4]:
                K = sg.two_site_exchange_matrix(k_ex, populations)

                # Build the exchanging ensemble in the rotating frame at 0 ppm.
                sites = self._single_spin_sites(deltas, [2.0, 1.0], list(T2))
                Ls = [
                    sg.rotating_frame(
                        ss,
                        sg.liouvillian(sg.hamiltonian(ss), sg.relaxation(ss)),
                        ["1H"],
                        0,
                    )
                    for ss in sites
                ]
                L = sg.exchange_liouvillian(sites, Ls, K)

                # Excite both sites with a 90-degree pulse.
                rho_0 = sg.exchange_state(
                    sites,
                    [sg.pulse(ss, "I(y)", 90) @ sg.equilibrium_state(ss)
                     for ss in sites],
                    populations,
                )

                # Calculate the site-resolved transverse signals.
                def signal(rho):
                    return np.array([
                        sg.measure(ss, rho_s, "I(+)")
                        for ss, rho_s in zip(
                            sites, sg.split_exchange_state(sites, rho)
                        )
                    ])
                m_0 = signal(rho_0)

                # Construct the generator of the Bloch-McConnell equations.
                offsets = np.array([
                    sg.resonance_frequency("1H", delta=delta, unit="rad/s")
                    - sg.resonance_frequency("1H", unit="rad/s")
                    for delta in deltas
                ])
                A = np.diag(-1j*offsets - 1/T2) + K

                # Compare the signals at several times.
                for t in [0.002, 0.01, 0.05]:
                    m_t = signal(sg.propagator(L, t) @ rho_0)
                    m_ref = la.expm(A * t) @ m_0
                    error = np.abs(m_t - m_ref).max() / np.abs(m_0).max()
                    self.assertLess(error, 1e-5)

    def test_exchange_cest_steady_state(self):
        """
        Test the steady state of two-site exchange under continuous-wave
        irradiation of one site against the Bloch-McConnell equations.
        """

        # Set the global parameters used in the test.
        sg.parameters.default()
        sg.parameters.magnetic_field = 1.0
        sg.parameters.temperature = 295

        # Define the sites, the exchange kinetics and the irradiation.
        deltas = [0.0, 2.0]
        T1 = np.array([2.0, 1.0])
        T2 = np.array([0.5, 0.2])
        populations = sg.equilibrium_two_site_populations(1e3, 1e-3, 2e-3)
        K = sg.two_site_exchange_matrix(30.0, populations)
        B1 = 5e-8
        w1 = 1 / 2 * B1 * abs(sg.gamma("1H", "rad/s"))

        # Test with dense and sparse backends.
        for sparse in [False, True]:
            sg.parameters.sparse_superoperator = sparse

            # Build the irradiated Liouvillians in the frame of site B.
            sites = self._single_spin_sites(deltas, list(T1), list(T2))
            Ls = []
            for ss in sites:
                L_site = sg.liouvillian(sg.hamiltonian(ss), sg.relaxation(ss))
                L_site = sg.rotating_frame(ss, L_site, ["1H"], deltas[1])
                H_rf = sg.rotating_frame_irradiation_hamiltonian(ss, "1H", B1)
                Ls.append(L_site - 1j * H_rf)
            L = sg.exchange_liouvillian(sites, Ls, K)

            # Propagate the weighted thermal equilibrium to the steady state.
            rho_eq = sg.exchange_state(
                sites,
                [sg.equilibrium_state(ss) for ss in sites],
                populations,
            )
            rho_ss = sg.propagator(L, 60.0) @ rho_eq
            Mz = np.array([
                sg.measure(ss, rho_ss_s, "I(z)") / sg.measure(ss, rho_eq_s, "I(z)")
                for ss, rho_ss_s, rho_eq_s in zip(
                    sites,
                    sg.split_exchange_state(sites, rho_ss),
                    sg.split_exchange_state(sites, rho_eq),
                )
            ]).real

            # Solve the steady state of the Bloch-McConnell equations.
            offsets = [
                sg.resonance_frequency("1H", delta=delta, unit="rad/s")
                - sg.resonance_frequency("1H", delta=deltas[1], unit="rad/s")
                for delta in deltas
            ]
            A = np.kron(K, np.eye(3))
            b = np.zeros(6)
            for s in range(2):
                w = [w1, 0, offsets[s]]
                rotation = np.array([
                    [0, -w[2], w[1]],
                    [w[2], 0, -w[0]],
                    [-w[1], w[0], 0],
                ])
                relaxation = np.diag([1/T2[s], 1/T2[s], 1/T1[s]])
                A[3*s:3*s+3, 3*s:3*s+3] += rotation - relaxation
                b[3*s+2] = populations[s] / T1[s]
            M = la.solve(A, -b)
            Mz_ref = np.array([M[2], M[5]]) / np.array(populations)

            # Compare the longitudinal magnetisations of both sites, within
            # the accuracy of the rotating-frame transformation.
            self.assertTrue(np.allclose(Mz, Mz_ref, rtol=1e-5, atol=0))
