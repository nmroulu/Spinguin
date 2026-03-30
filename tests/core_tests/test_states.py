"""
Tests for state construction, state measurement, and equilibrium populations.
"""

from itertools import product
import unittest

import numpy as np
import scipy.constants as const

import spinguin as sg

class TestStates(unittest.TestCase):
    """
    Test state-generation utilities and state-based observables.
    """

    def _assert_allclose(
        self,
        value,
        reference,
        rtol=1e-05,
        atol=1e-08,
    ):
        """
        Assert that two numerical arrays agree within tolerance.
        """

        # Convert sparse matrices to dense arrays when necessary.
        if hasattr(value, "toarray"):
            value = value.toarray()
        if hasattr(reference, "toarray"):
            reference = reference.toarray()

        # Compare the tested and reference arrays.
        self.assertTrue(np.allclose(value, reference, rtol=rtol, atol=atol))

    def _build_spin_system(
        self,
        isotopes,
        max_spin_order,
    ):
        """
        Create a spin system and build its basis set.
        """

        # Create the requested spin system.
        spin_system = sg.SpinSystem(isotopes)

        # Build the basis set used in the test.
        spin_system.basis.max_spin_order = max_spin_order
        spin_system.basis.build()

        return spin_system

    def _to_dense_zeeman(
        self,
        spin_system,
        state,
    ):
        """
        Convert a state to a dense Zeeman-basis density matrix.
        """

        # Convert the state to the Zeeman basis.
        state_zeeman = sg.state_to_zeeman(spin_system, state)

        # Convert sparse output to a dense array when necessary.
        if hasattr(state_zeeman, "toarray"):
            state_zeeman = state_zeeman.toarray()

        return state_zeeman

    def _build_dense_and_sparse_states(
        self,
        spin_system,
        state_function,
        *args,
    ):
        """
        Build dense and sparse variants of a state and convert both to Zeeman.
        """

        # Build the state in dense format.
        sg.parameters.sparse_state = False
        state_dense = state_function(spin_system, *args)

        # Build the state in sparse format.
        sg.parameters.sparse_state = True
        state_sparse = state_function(spin_system, *args)

        # Convert both states to dense Zeeman-basis matrices.
        state_dense = self._to_dense_zeeman(spin_system, state_dense)
        state_sparse = self._to_dense_zeeman(spin_system, state_sparse)

        return state_dense, state_sparse

    def _get_two_spin_half_operators(
        self,
    ):
        """
        Return the spin-1/2 operators used in the two-spin tests.
        """

        # Construct the spin-1/2 single-spin operators.
        identity = sg.op_E(1 / 2)
        iz_operator = sg.op_Sz(1 / 2)
        ip_operator = sg.op_Sp(1 / 2)
        im_operator = sg.op_Sm(1 / 2)

        return identity, iz_operator, ip_operator, im_operator

    def _get_operator_string(
        self,
        label,
        index,
    ):
        """
        Convert an operator label to a spherical-tensor operator string.
        """

        # Return the identity operator unchanged.
        if label == "E":
            return "E"

        # Construct the indexed operator string.
        return f"I({label}, {index})"

    def _get_operator_map(
        self,
        spin_system,
    ):
        """
        Build dense Zeeman-basis operators for all tested labels.
        """

        # Force dense operator construction for the reference matrices.
        sg.parameters.sparse_operator = False

        # Build the operator map for all tested labels.
        operators = {}
        for spin in spin_system.spins:
            operators[("E", spin)] = sg.op_E(spin)
            operators[("x", spin)] = sg.op_Sx(spin)
            operators[("y", spin)] = sg.op_Sy(spin)
            operators[("z", spin)] = sg.op_Sz(spin)
            operators[("+", spin)] = sg.op_Sp(spin)
            operators[("-", spin)] = sg.op_Sm(spin)

        return operators

    def _build_zeeman_product(
        self,
        labels,
        spins,
        operators,
    ):
        """
        Build a Kronecker product of Zeeman-basis operators.
        """

        # Initialise the Kronecker product with a scalar identity.
        zeeman_product = np.array([[1]])

        # Build the direct-product operator.
        for label, spin in zip(labels, spins):
            zeeman_product = np.kron(zeeman_product, operators[(label, spin)])

        return zeeman_product

    def test_alpha(self):
        """
        Test alpha-state construction against Zeeman-basis references.
        """

        # Reset the global settings.
        sg.parameters.default()

        # Use dense operators for the reference matrices.
        sg.parameters.sparse_operator = False

        # Build the example two-spin system.
        spin_system = self._build_spin_system(["1H", "1H"], 2)

        # Build the single-spin reference operators.
        identity, iz_operator, _, _ = self._get_two_spin_half_operators()

        # Construct the Zeeman-basis reference density matrices.
        alpha1_reference = (
            1 / 4 * np.kron(identity, identity) +
            1 / 2 * np.kron(iz_operator, identity)
        )
        alpha2_reference = (
            1 / 4 * np.kron(identity, identity) +
            1 / 2 * np.kron(identity, iz_operator)
        )

        # Build dense and sparse variants of both alpha states.
        alpha1_dense, alpha1_sparse = self._build_dense_and_sparse_states(
            spin_system,
            sg.alpha_state,
            0,
        )
        alpha2_dense, alpha2_sparse = self._build_dense_and_sparse_states(
            spin_system,
            sg.alpha_state,
            1,
        )

        # Compare the constructed states with the references.
        self._assert_allclose(alpha1_dense, alpha1_reference)
        self._assert_allclose(alpha2_dense, alpha2_reference)
        self._assert_allclose(alpha1_sparse, alpha1_reference)
        self._assert_allclose(alpha2_sparse, alpha2_reference)

    def test_beta(self):
        """
        Test beta-state construction against Zeeman-basis references.
        """

        # Reset the global settings.
        sg.parameters.default()

        # Use dense operators for the reference matrices.
        sg.parameters.sparse_operator = False

        # Build the example two-spin system.
        spin_system = self._build_spin_system(["1H", "1H"], 2)

        # Build the single-spin reference operators.
        identity, iz_operator, _, _ = self._get_two_spin_half_operators()

        # Construct the Zeeman-basis reference density matrices.
        beta1_reference = (
            1 / 4 * np.kron(identity, identity) -
            1 / 2 * np.kron(iz_operator, identity)
        )
        beta2_reference = (
            1 / 4 * np.kron(identity, identity) -
            1 / 2 * np.kron(identity, iz_operator)
        )

        # Build dense and sparse variants of both beta states.
        beta1_dense, beta1_sparse = self._build_dense_and_sparse_states(
            spin_system,
            sg.beta_state,
            0,
        )
        beta2_dense, beta2_sparse = self._build_dense_and_sparse_states(
            spin_system,
            sg.beta_state,
            1,
        )

        # Compare the constructed states with the references.
        self._assert_allclose(beta1_dense, beta1_reference)
        self._assert_allclose(beta2_dense, beta2_reference)
        self._assert_allclose(beta1_sparse, beta1_reference)
        self._assert_allclose(beta2_sparse, beta2_reference)

    def test_singlet(self):
        """
        Test singlet-state construction against a Zeeman-basis reference.
        """

        # Reset the global settings.
        sg.parameters.default()

        # Use dense operators for the reference matrices.
        sg.parameters.sparse_operator = False

        # Build the example two-spin system.
        spin_system = self._build_spin_system(["1H", "1H"], 2)

        # Build the single-spin reference operators.
        identity, iz_operator, ip_operator, im_operator = (
            self._get_two_spin_half_operators()
        )

        # Construct the Zeeman-basis reference density matrix.
        singlet_reference = (
            1 / 4 * np.kron(identity, identity) -
            np.kron(iz_operator, iz_operator) -
            1 / 2 * (
                np.kron(ip_operator, im_operator) +
                np.kron(im_operator, ip_operator)
            )
        )

        # Build dense and sparse variants of the singlet state.
        singlet_dense, singlet_sparse = self._build_dense_and_sparse_states(
            spin_system,
            sg.singlet_state,
            0,
            1,
        )

        # Compare the constructed states with the reference.
        self._assert_allclose(singlet_reference, singlet_dense)
        self._assert_allclose(singlet_reference, singlet_sparse)

    def test_triplet_zero(self):
        """
        Test triplet-zero-state construction against a Zeeman reference.
        """

        # Reset the global settings.
        sg.parameters.default()

        # Use dense operators for the reference matrices.
        sg.parameters.sparse_operator = False

        # Build the example two-spin system.
        spin_system = self._build_spin_system(["1H", "1H"], 2)

        # Build the single-spin reference operators.
        identity, iz_operator, ip_operator, im_operator = (
            self._get_two_spin_half_operators()
        )

        # Construct the Zeeman-basis reference density matrix.
        triplet_zero_reference = (
            1 / 4 * np.kron(identity, identity) -
            np.kron(iz_operator, iz_operator) +
            1 / 2 * (
                np.kron(ip_operator, im_operator) +
                np.kron(im_operator, ip_operator)
            )
        )

        # Build dense and sparse variants of the triplet-zero state.
        triplet_zero_dense, triplet_zero_sparse = (
            self._build_dense_and_sparse_states(
                spin_system,
                sg.triplet_zero_state,
                0,
                1,
            )
        )

        # Compare the constructed states with the reference.
        self._assert_allclose(triplet_zero_reference, triplet_zero_dense)
        self._assert_allclose(triplet_zero_reference, triplet_zero_sparse)

    def test_triplet_plus(self):
        """
        Test triplet-plus-state construction against a Zeeman reference.
        """

        # Reset the global settings.
        sg.parameters.default()

        # Use dense operators for the reference matrices.
        sg.parameters.sparse_operator = False

        # Build the example two-spin system.
        spin_system = self._build_spin_system(["1H", "1H"], 2)

        # Build the single-spin reference operators.
        identity, iz_operator, _, _ = self._get_two_spin_half_operators()

        # Construct the Zeeman-basis reference density matrix.
        triplet_plus_reference = (
            1 / 4 * np.kron(identity, identity) +
            1 / 2 * np.kron(identity, iz_operator) +
            1 / 2 * np.kron(iz_operator, identity) +
            np.kron(iz_operator, iz_operator)
        )

        # Build dense and sparse variants of the triplet-plus state.
        triplet_plus_dense, triplet_plus_sparse = (
            self._build_dense_and_sparse_states(
                spin_system,
                sg.triplet_plus_state,
                0,
                1,
            )
        )

        # Compare the constructed states with the reference.
        self._assert_allclose(triplet_plus_reference, triplet_plus_dense)
        self._assert_allclose(triplet_plus_reference, triplet_plus_sparse)

    def test_triplet_minus(self):
        """
        Test triplet-minus-state construction against a Zeeman reference.
        """

        # Reset the global settings.
        sg.parameters.default()

        # Use dense operators for the reference matrices.
        sg.parameters.sparse_operator = False

        # Build the example two-spin system.
        spin_system = self._build_spin_system(["1H", "1H"], 2)

        # Build the single-spin reference operators.
        identity, iz_operator, _, _ = self._get_two_spin_half_operators()

        # Construct the Zeeman-basis reference density matrix.
        triplet_minus_reference = (
            1 / 4 * np.kron(identity, identity) -
            1 / 2 * np.kron(identity, iz_operator) -
            1 / 2 * np.kron(iz_operator, identity) +
            np.kron(iz_operator, iz_operator)
        )

        # Build dense and sparse variants of the triplet-minus state.
        triplet_minus_dense, triplet_minus_sparse = (
            self._build_dense_and_sparse_states(
                spin_system,
                sg.triplet_minus_state,
                0,
                1,
            )
        )

        # Compare the constructed states with the reference.
        self._assert_allclose(triplet_minus_reference, triplet_minus_dense)
        self._assert_allclose(triplet_minus_reference, triplet_minus_sparse)

    def _test_state(
        self,
        ss: sg.SpinSystem,
        opers: dict,
        test_states: list,
        sparse_operator: bool,
        sparse_state: bool,
    ):
        """
        Helper method for `test_state()`.
        """

        # Set the state and operator sparsity.
        sg.parameters.sparse_operator = sparse_operator
        sg.parameters.sparse_state = sparse_state

        # Compare all possible three-spin state combinations.
        for labels in product(test_states, repeat=3):

            # Build the spherical-tensor operator string.
            op_string = " * ".join(
                self._get_operator_string(label, index)
                for index, label in enumerate(labels)
            )

            # Build the state in the spherical tensor basis.
            state = sg.state(ss, op_string)

            # Build the Zeeman-basis reference density matrix.
            state_reference = self._build_zeeman_product(
                labels,
                ss.spins,
                opers,
            )

            # Convert the tested state to a dense Zeeman-basis matrix.
            state = self._to_dense_zeeman(ss, state)

            # Compare the tested and reference density matrices.
            self._assert_allclose(state, state_reference)

    def test_state(self):
        """
        Test state construction from operator strings against Zeeman references.
        """

        # Reset the global settings.
        sg.parameters.default()

        # Build the example spin system.
        spin_system = self._build_spin_system(["1H", "14N", "23Na"], 3)

        # Define the operator labels included in the test.
        test_states = ["E", "x", "y", "z", "+", "-"]

        # Build the dense Zeeman-basis reference operators.
        operators = self._get_operator_map(spin_system)

        # Compare all sparsity combinations.
        self._test_state(spin_system, operators, test_states, True, True)
        self._test_state(spin_system, operators, test_states, True, False)
        self._test_state(spin_system, operators, test_states, False, True)
        self._test_state(spin_system, operators, test_states, False, False)

    def test_unit_state(self):
        """
        Test unit-state construction with and without trace normalisation.
        """

        # Reset the global settings.
        sg.parameters.default()

        # Build the example spin system.
        spin_system = self._build_spin_system(["1H", "14N", "23Na"], 3)

        # Build the unit operator in the Zeeman basis.
        operators = self._get_operator_map(spin_system)
        unit_zeeman = self._build_zeeman_product(
            ["E"] * spin_system.nspins,
            spin_system.spins,
            operators,
        )

        # Build the non-normalised unit state in dense and sparse formats.
        sg.parameters.sparse_state = False
        unit_dense = sg.unit_state(spin_system, normalized=False)
        sg.parameters.sparse_state = True
        unit_sparse = sg.unit_state(spin_system, normalized=False)

        # Convert both unit states to dense Zeeman-basis matrices.
        unit_dense = self._to_dense_zeeman(spin_system, unit_dense)
        unit_sparse = self._to_dense_zeeman(spin_system, unit_sparse)

        # Compare the non-normalised unit states with the reference.
        self._assert_allclose(unit_dense, unit_zeeman)
        self._assert_allclose(unit_sparse, unit_zeeman)

        # Build the trace-normalised unit state in dense and sparse formats.
        sg.parameters.sparse_state = False
        unit_dense = sg.unit_state(spin_system, normalized=True)
        sg.parameters.sparse_state = True
        unit_sparse = sg.unit_state(spin_system, normalized=True)

        # Convert both unit states to dense Zeeman-basis matrices.
        unit_dense = self._to_dense_zeeman(spin_system, unit_dense)
        unit_sparse = self._to_dense_zeeman(spin_system, unit_sparse)

        # Apply the same trace normalisation to the reference matrix.
        unit_zeeman = unit_zeeman / unit_zeeman.trace()

        # Compare the trace-normalised unit states with the reference.
        self._assert_allclose(unit_dense, unit_zeeman)
        self._assert_allclose(unit_sparse, unit_zeeman)

    def _test_measure(
        self,
        ss: sg.SpinSystem,
        test_states: list,
        opers: dict,
        sparse_state: bool,
    ):
        """
        Helper method for `test_measure()`.
        """

        # Set the state sparsity used in measurement tests.
        sg.parameters.sparse_state = sparse_state

        # Compare all possible state and operator combinations.
        for state_labels in product(test_states, repeat=3):

            # Build the spherical-tensor state string.
            state_string = " * ".join(
                self._get_operator_string(label, index)
                for index, label in enumerate(state_labels)
            )

            # Build the state in the spherical tensor basis.
            state = sg.state(ss, state_string)

            # Build the Zeeman-basis reference density matrix.
            state_zeeman = self._build_zeeman_product(
                state_labels,
                ss.spins,
                opers,
            )

            # Compare all possible measurement operators.
            for oper_labels in product(test_states, repeat=3):

                # Build the spherical-tensor measurement string.
                oper_string = " * ".join(
                    self._get_operator_string(label, index)
                    for index, label in enumerate(oper_labels)
                )

                # Build the Zeeman-basis measurement operator.
                oper_zeeman = self._build_zeeman_product(
                    oper_labels,
                    ss.spins,
                    opers,
                )

                # Measure the state directly in the Zeeman basis.
                result_zeeman = (state_zeeman @ oper_zeeman.conj().T).trace()

                # Measure the state with the public Spinguin API.
                result = sg.measure(ss, state, oper_string)

                # Compare the two expectation values.
                self.assertAlmostEqual(result_zeeman, result)

    def test_measure(self):
        """
        Test measurements against direct Zeeman-basis expectation values.
        """

        # Reset the global settings.
        sg.parameters.default()

        # Build the example spin system with mixed spin quantum numbers.
        spin_system = self._build_spin_system(["1H", "14N", "23Na"], 3)

        # Define the operator labels included in the test.
        test_states = ["E", "x", "y", "z", "+", "-"]

        # Build the dense Zeeman-basis reference operators.
        operators = self._get_operator_map(spin_system)

        # Compare dense and sparse state representations.
        self._test_measure(spin_system, test_states, operators, False)
        self._test_measure(spin_system, test_states, operators, True)

    def _test_equilibrium_state(
        self,
        ss: sg.SpinSystem,
        magnetic_field: float,
        temperature: float,
    ):
        """
        Helper method for testing the equilibrium state.
        """

        # Set the experimental conditions.
        sg.parameters.magnetic_field = magnetic_field
        sg.parameters.temperature = temperature

        # Construct dense and sparse equilibrium states.
        sg.parameters.sparse_state = False
        rho_equilibrium_dense = sg.equilibrium_state(ss)
        sg.parameters.sparse_state = True
        rho_equilibrium_sparse = sg.equilibrium_state(ss)

        # Compare the equilibrium magnetisation for every spin.
        for spin_index in range(ss.nspins):

            # Measure the longitudinal magnetisation from both states.
            mz_measured_dense = sg.measure(
                ss,
                rho_equilibrium_dense,
                f"I(z, {spin_index})",
            )
            mz_measured_sparse = sg.measure(
                ss,
                rho_equilibrium_sparse,
                f"I(z, {spin_index})",
            )

            # Calculate the reference equilibrium magnetisation.
            mz_reference = _thermal_magnetization(
                gamma=ss.gammas[spin_index],
                spin_quantum_number=ss.spins[spin_index],
                magnetic_field=sg.parameters.magnetic_field,
                temperature=sg.parameters.temperature,
            )

            # Compare the measured and reference magnetisations.
            self.assertAlmostEqual(mz_measured_sparse, mz_reference)
            self.assertAlmostEqual(mz_measured_dense, mz_reference)

    def test_equilibrium_state(self):
        """
        Test equilibrium magnetisation against Boltzmann-distribution values.
        """

        # Reset the global settings.
        sg.parameters.default()

        # Build the example spin system.
        spin_system = self._build_spin_system(
            ["1H", "14N", "23Na", "17O"],
            4,
        )

        # Compare the equilibrium state under two conditions.
        self._test_equilibrium_state(
            spin_system,
            magnetic_field=9.4,
            temperature=293,
        )
        self._test_equilibrium_state(
            spin_system,
            magnetic_field=100,
            temperature=1,
        )


def _thermal_magnetization(
    gamma: float,
    spin_quantum_number: float,
    magnetic_field: float,
    temperature: float,
) -> float:
    """
    Calculate the thermal-equilibrium magnetisation for one spin.

    Parameters
    ----------
    gamma : float
        Gyromagnetic ratio of the nucleus in rad/s/T.
    spin_quantum_number : float
        Spin quantum number.
    magnetic_field : float
        Magnetic field in Tesla.
    temperature : float
        Temperature in Kelvin.

    Returns
    -------
    magnetization : float
        Magnetization at thermal equilibrium.
    """
    # Get the available spin magnetic quantum numbers.
    magnetic_quantum_numbers = np.arange(
        -spin_quantum_number,
        spin_quantum_number + 1,
    )

    # Calculate the Boltzmann populations of the Zeeman levels.
    populations = {}
    for magnetic_quantum_number in magnetic_quantum_numbers:

        # Calculate the population numerator for the current level.
        numerator = np.exp(
            magnetic_quantum_number * const.hbar * gamma * magnetic_field /
            (const.k * temperature)
        )

        # Calculate the partition-function denominator.
        denominator = sum(
            np.exp(
                other_quantum_number * const.hbar * gamma * magnetic_field /
                (const.k * temperature)
            )
            for other_quantum_number in magnetic_quantum_numbers
        )
        populations[magnetic_quantum_number] = numerator / denominator

    # Calculate the equilibrium magnetisation.
    magnetization = sum(
        magnetic_quantum_number * populations[magnetic_quantum_number]
        for magnetic_quantum_number in magnetic_quantum_numbers
    )

    return magnetization