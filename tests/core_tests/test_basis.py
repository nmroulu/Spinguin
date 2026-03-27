"""
Tests for basis-set construction and truncation in the Spinguin core.
"""

import math
import unittest
from copy import deepcopy

import numpy as np

import spinguin as sg


class TestBasis(unittest.TestCase):
    """
    Test basis-set generation and basis-set truncation utilities.
    """

    def _build_basis(
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

    def _assert_deleted_states(
        self,
        spin_system,
        basis_original,
        is_deleted,
    ):
        """
        Check that the truncated basis contains exactly the expected states.

        Parameters
        ----------
        spin_system : SpinSystem
            Spin system whose basis set has already been truncated.
        basis_original : ndarray
            Basis states before truncation.
        is_deleted : callable
            Function that returns ``True`` for states that should have been
            removed.

        Returns
        -------
        None
            The assertions are evaluated in place.
        """

        # Test each state against the expected truncation rule.
        for operator_definition in basis_original:
            if is_deleted(operator_definition):
                with self.assertRaises(ValueError):
                    spin_system.basis.indexof(operator_definition)
            else:
                spin_system.basis.indexof(operator_definition)

    def test_make_basis_1(self):
        """
        Test basis-set construction against a hard-coded reference.
        """

        # Define the reference basis set explicitly.
        basis_reference = np.array(
            [
                [0, 0],
                [0, 1],
                [0, 2],
                [0, 3],
                [1, 0],
                [2, 0],
                [3, 0],
            ]
        )

        # Build the basis set for the test system.
        spin_system = self._build_basis(["1H", "1H"], 1)

        # Compare the generated basis set with the reference result.
        self.assertTrue(np.array_equal(spin_system.basis.basis, basis_reference))

    def test_make_basis_2(self):
        """
        Test basis-set dimensions for a larger spin system.
        """

        # Create the large test system.
        spin_system = sg.SpinSystem(["1H", "1H", "1H", "1H", "1H", "1H", "1H"])
        nspins = spin_system.nspins

        # Compare the basis dimension with the combinatorial reference value.
        for max_so in range(1, nspins):
            spin_system.basis.max_spin_order = max_so
            spin_system.basis.build()
            dimension_reference = sum(
                math.comb(nspins, k) * 3**k for k in range(max_so + 1)
            )
            self.assertEqual(spin_system.basis.dim, dimension_reference)

    def test_state_idx(self):
        """
        Test the mapping of states to their corresponding indices.
        """

        # Build the basis set for the test system.
        spin_system = self._build_basis(["14N", "1H", "1H"], 2)

        # Check state indexing for multiple supported input types.
        self.assertEqual(spin_system.basis.indexof([0, 1, 0]), 4)
        self.assertEqual(spin_system.basis.indexof((1, 0, 3)), 19)
        self.assertEqual(spin_system.basis.indexof(np.array([8, 0, 3])), 68)

        # Check that a non-existent state raises an error.
        with self.assertRaises(ValueError):
            spin_system.basis.indexof([9, 9, 9])

        # Check that an undersized state vector raises an error.
        with self.assertRaises(ValueError):
            spin_system.basis.indexof([0])

        # Check that an oversized state vector raises an error.
        with self.assertRaises(ValueError):
            spin_system.basis.indexof([0, 1, 2, 0])

    def test_truncate_basis_by_coherence(self):
        """
        Test the creation of the truncated basis using coherence order as the
        selection criterion.
        """

        # Build the full basis set for the example system.
        spin_system = self._build_basis(["1H", "1H", "1H", "1H", "14N"], 4)

        # Save the original basis set for later comparison.
        basis_original = spin_system.basis.basis.copy()

        # Create a superoperator and a state in the full basis set.
        operator_original = sg.superoperator(
            spin_system,
            "I(z,0) * I(+,1) * I(-,2)",
        )
        state_original = sg.state(
            spin_system,
            "I(+,1) * I(z,3) * I(-,4)",
        )

        # Truncate the basis and transform the operator and state with it.
        coherence_orders = [-2, 0, 1]
        operator_original_truncated, state_original_truncated = (
            spin_system.basis.truncate_by_coherence(
                coherence_orders,
                operator_original,
                state_original,
            )
        )

        # Construct the same operator and state directly in the new basis.
        operator_truncated = sg.superoperator(
            spin_system,
            "I(z,0) * I(+,1) * I(-,2)",
        )
        state_truncated = sg.state(
            spin_system,
            "I(+,1) * I(z,3) * I(-,4)",
        )

        # Verify that both construction routes give the same result.
        self.assertTrue(
            np.allclose(
                operator_original_truncated.toarray(),
                operator_truncated.toarray(),
            )
        )
        self.assertTrue(np.allclose(state_original_truncated, state_truncated))

        # Check that only the requested coherence orders remain.
        for operator_definition in basis_original:
            if sg.coherence_order(operator_definition) in coherence_orders:
                spin_system.basis.indexof(operator_definition)
            else:
                with self.assertRaises(ValueError):
                    spin_system.basis.indexof(operator_definition)

    def test_truncate_basis_by_coupling_1(self):
        """
        Test basis truncation from scalar-coupling connectivity.
        """

        # Build the full basis set for the example system.
        spin_system = self._build_basis(["1H", "1H", "1H"], 3)

        # Assign the scalar-coupling topology.
        spin_system.J_couplings = [
            [0, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
        ]

        # Save the original basis set for later comparison.
        basis_original = spin_system.basis.basis.copy()

        # Truncate the basis set according to the coupling graph.
        spin_system.basis.truncate_by_coupling()

        # Check that only the expected disconnected states are removed.
        self._assert_deleted_states(
            spin_system,
            basis_original,
            lambda operator_definition: (
                operator_definition[0] == 0
                and operator_definition[1] != 0
                and operator_definition[2] != 0
            ),
        )

    def test_truncate_basis_by_coupling_2(self):
        """
        Test basis truncation from distance-based coupling connectivity.
        """

        # Build the full basis set for the example system.
        spin_system = self._build_basis(["1H", "1H", "1H"], 3)

        # Assign Cartesian coordinates used to infer couplings.
        spin_system.xyz = [
            [0, 0, 0],
            [1, 1, 1],
            [99, 99, 99],
        ]

        # Save the original basis set for later comparison.
        basis_original = spin_system.basis.basis.copy()

        # Truncate the basis set according to the inferred coupling graph.
        spin_system.basis.truncate_by_coupling()

        # Check that only the expected disconnected states are removed.
        self._assert_deleted_states(
            spin_system,
            basis_original,
            lambda operator_definition: (
                (operator_definition[0] != 0 or operator_definition[1] != 0)
                and operator_definition[2] != 0
            ),
        )

    def test_truncate_basis_by_coupling_3(self):
        """
        Test basis truncation with combined coupling definitions.
        """

        # Build the full basis set for the example system.
        spin_system = self._build_basis(["1H", "1H", "1H"], 3)

        # Assign scalar couplings where spins 1 and 3 are coupled.
        spin_system.J_couplings = [
            [0, 0, 0],
            [0, 0, 0],
            [1, 0, 0],
        ]

        # Assign coordinates where spins 1 and 2 are coupled.
        spin_system.xyz = [
            [0, 0, 0],
            [1, 1, 1],
            [99, 99, 99],
        ]

        # Save the original basis set for later comparison.
        basis_original = spin_system.basis.basis.copy()

        # Truncate the basis set using the combined connectivity data.
        spin_system.basis.truncate_by_coupling()

        # Check that only the expected disconnected states are removed.
        self._assert_deleted_states(
            spin_system,
            basis_original,
            lambda operator_definition: (
                operator_definition[0] == 0
                and operator_definition[1] != 0
                and operator_definition[2] != 0
            ),
        )

    def test_truncate_basis_by_zte(self):
        """
        Test the basis set truncation using ZTE by comparing the generated FID
        to the exact solution.
        """

        # Set the global simulation parameters.
        sg.parameters.default()
        sg.parameters.magnetic_field = 1
        sg.parameters.temperature = 295

        # Build the basis set for the example spin system.
        spin_system = self._build_basis(
            ["1H", "1H", "1H", "1H", "1H", "14N"],
            3,
        )

        # Define the chemical shifts in ppm.
        spin_system.chemical_shifts = [8.56, 8.56, 7.47, 7.47, 7.88, 95.94]

        # Define scalar couplings in Hz.
        spin_system.J_couplings = [
            [0, 0, 0, 0, 0, 0],
            [-1.04, 0, 0, 0, 0, 0],
            [4.85, 1.05, 0, 0, 0, 0],
            [1.05, 4.85, 0.71, 0, 0, 0],
            [1.24, 1.24, 7.55, 7.55, 0, 0],
            [8.16, 8.16, 0.87, 0.87, -0.19, 0],
        ]

        # Define Cartesian coordinates of the nuclei.
        spin_system.xyz = [
            [2.0495335, 0.0000000, -1.4916842],
            [-2.0495335, 0.0000000, -1.4916842],
            [2.1458878, 0.0000000, 0.9846086],
            [-2.1458878, 0.0000000, 0.9846086],
            [0.0000000, 0.0000000, 2.2681296],
            [0.0000000, 0.0000000, -1.5987077],
        ]

        # Define the shielding tensors.
        shielding = np.zeros((6, 3, 3))
        shielding[5] = np.array([
            [-406.20, 0.00, 0.00],
            [0.00, 299.44, 0.00],
            [0.00, 0.00, -181.07],
        ])
        spin_system.shielding = shielding

        # Define the electric-field-gradient tensors.
        efg = np.zeros((6, 3, 3))
        efg[5] = np.array([
            [0.3069, 0.0000, 0.0000],
            [0.0000, 0.7969, 0.0000],
            [0.0000, 0.0000, -1.1037],
        ])
        spin_system.efg = efg

        # Define the relaxation model.
        spin_system.relaxation.thermalization = True
        spin_system.relaxation.theory = "redfield"
        spin_system.relaxation.tau_c = 50e-12

        # Define the acquisition parameters.
        dt = 1e-3
        npoints = 1000

        # Build the Hamiltonian and Redfield relaxation superoperator.
        hamiltonian = sg.hamiltonian(spin_system)
        relaxation_superoperator = sg.relaxation(spin_system)

        # Construct the Liouvillian.
        liouvillian = sg.liouvillian(hamiltonian, relaxation_superoperator)

        # Transform the Liouvillian to the rotating frame.
        liouvillian = sg.rotating_frame(
            spin_system,
            liouvillian,
            ["1H", "14N"],
            [8, 96],
        )

        # Get the thermal equilibrium state.
        state = sg.equilibrium_state(spin_system)

        # Apply a proton pulse to generate observable coherence.
        proton_indices = np.where(spin_system.isotopes == "1H")[0]
        pulse_operator = "+".join(f"I(y,{index})" for index in proton_indices)
        pulse_x = sg.pulse(spin_system, pulse_operator, 90)
        state = pulse_x @ state

        # Retain only single-quantum coherences before the ZTE step.
        liouvillian, state = spin_system.basis.truncate_by_coherence(
            [1, -1],
            liouvillian,
            state,
        )

        # Apply ZTE in a copied spin system.
        spin_system_zte = deepcopy(spin_system)
        liouvillian_zte, state_zte = spin_system_zte.basis.truncate_by_zte(
            liouvillian,
            state,
        )

        # Build the propagators for the full and truncated problems.
        propagator = sg.propagator(liouvillian, dt)
        propagator_zte = sg.propagator(liouvillian_zte, dt)

        # Construct the measurement operator.
        measurement_operator = "+".join(
            f"I(-,{index})" for index in proton_indices
        )

        # Allocate arrays for the reference and ZTE signals.
        fid = np.zeros(npoints, dtype=complex)
        fid_zte = np.zeros(npoints, dtype=complex)

        # Propagate both systems and record their free induction decays.
        for step in range(npoints):
            fid[step] = sg.measure(spin_system, state, measurement_operator)
            fid_zte[step] = sg.measure(
                spin_system_zte,
                state_zte,
                measurement_operator,
            )
            state = propagator @ state
            state_zte = propagator_zte @ state_zte

        # Verify that the ZTE truncation preserves the signal exactly.
        self.assertTrue(np.allclose(fid, fid_zte))

    def test_truncate_basis_by_indices(self):
        """
        Test basis-set truncation by explicitly retained basis indices.
        """

        # Build the full basis set for the example system.
        spin_system = self._build_basis(["1H", "1H", "1H"], 3)

        # Keep a copy of the original basis set for comparison.
        spin_system_original = deepcopy(spin_system)

        # Retain only a selected subset of basis states.
        retained_indices = [0, 7, spin_system.basis.dim - 1]
        spin_system.basis.truncate_by_indices(retained_indices)

        # Check that only the requested states remain in the basis set.
        for operator_definition in spin_system_original.basis.basis:
            index = spin_system_original.basis.indexof(operator_definition)
            if index not in retained_indices:
                with self.assertRaises(ValueError):
                    spin_system.basis.indexof(operator_definition)
            else:
                spin_system.basis.indexof(operator_definition)