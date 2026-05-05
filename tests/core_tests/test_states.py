"""
Tests for state construction, state measurement, and equilibrium populations.
"""

import unittest

import numpy as np
import scipy.constants as const

import spinguin as sg
from ._helpers import build_spin_system

class TestStates(unittest.TestCase):
    """
    Test state-generation utilities and state-based observables.
    """

    def test_alpha(self):
        """
        Test alpha-state construction against Zeeman-basis references.
        """

        # Reset the global settings.
        sg.parameters.default()

        # Use dense operators for the reference matrices.
        sg.parameters.sparse_operator = False

        # Build the example two-spin system.
        ss = build_spin_system(["1H", "1H"], 2)

        # Build the single-spin reference operators.
        E = sg.op_E(1/2)
        Iz = sg.op_Sz(1/2)

        # Construct the Zeeman-basis reference density matrices.
        alpha1_ref = 1/4 * np.kron(E, E) + 1/2 * np.kron(Iz, E)
        alpha2_ref = 1/4 * np.kron(E, E) + 1/2 * np.kron(E, Iz)

        # Test with dense and sparse backends
        for sparse in (False, True):
            sg.parameters.sparse_state = sparse

            # Create alpha states using the inbuilt function
            alpha1 = sg.alpha_state(ss, 0)
            alpha2 = sg.alpha_state(ss, 1)

            # Convert the states to density matrices
            alpha1 = sg.state_to_zeeman(ss, alpha1)
            alpha2 = sg.state_to_zeeman(ss, alpha2)

            # Compare the constructed states with the references.
            self.assertTrue(np.allclose(alpha1, alpha1_ref))
            self.assertTrue(np.allclose(alpha2, alpha2_ref))

    def test_beta(self):
        """
        Test beta-state construction against Zeeman-basis references.
        """
        # Reset the global settings.
        sg.parameters.default()

        # Use dense operators for the reference matrices.
        sg.parameters.sparse_operator = False

        # Build the example two-spin system.
        ss = build_spin_system(["1H", "1H"], 2)

        # Build the single-spin reference operators.
        E = sg.op_E(1/2)
        Iz = sg.op_Sz(1/2)

        # Construct the Zeeman-basis reference density matrices.
        beta1_ref = 1/4 * np.kron(E, E) - 1/2 * np.kron(Iz, E)
        beta2_ref = 1/4 * np.kron(E, E) - 1/2 * np.kron(E, Iz)

        # Test with dense and sparse backends
        for sparse in (False, True):
            sg.parameters.sparse_state = sparse

            # Create beta states using the inbuilt function
            beta1 = sg.beta_state(ss, 0)
            beta2 = sg.beta_state(ss, 1)

            # Convert the states to density matrices
            beta1 = sg.state_to_zeeman(ss, beta1)
            beta2 = sg.state_to_zeeman(ss, beta2)

            # Compare the constructed states with the references.
            self.assertTrue(np.allclose(beta1, beta1_ref))
            self.assertTrue(np.allclose(beta2, beta2_ref))

    def test_singlet(self):
        """
        Test singlet-state construction against a Zeeman-basis reference.
        """

        # Reset the global settings.
        sg.parameters.default()

        # Use dense operators for the reference matrices.
        sg.parameters.sparse_operator = False

        # Build the example two-spin system.
        ss = build_spin_system(["1H", "1H"], 2)

        # Build the single-spin reference operators.
        E = sg.op_E(1/2)
        Iz = sg.op_Sz(1/2)
        Ip = sg.op_Sp(1/2)
        Im = sg.op_Sm(1/2)

        # Construct the Zeeman-basis reference density matrix.
        singlet_ref = \
            1/4 * np.kron(E, E) - np.kron(Iz, Iz) - \
            1/2 * (np.kron(Ip, Im) + np.kron(Im, Ip))
        
        # Test with dense and sparse backends
        for sparse in (False, True):
            sg.parameters.sparse_state = sparse

            # Create the singlet state using the inbuilt function
            singlet = sg.singlet_state(ss, 0, 1)

            # Convert the state to a density matrix
            singlet = sg.state_to_zeeman(ss, singlet)

            # Compare the constructed state with the reference.
            self.assertTrue(np.allclose(singlet, singlet_ref))

    def test_triplet_zero(self):
        """
        Test triplet-zero-state construction against a Zeeman reference.
        """

        # Reset the global settings.
        sg.parameters.default()

        # Use dense operators for the reference matrices.
        sg.parameters.sparse_operator = False

        # Build the example two-spin system.
        ss = build_spin_system(["1H", "1H"], 2)

        # Build the single-spin reference operators.
        E = sg.op_E(1/2)
        Iz = sg.op_Sz(1/2)
        Ip = sg.op_Sp(1/2)
        Im = sg.op_Sm(1/2)

        # Construct the Zeeman-basis reference density matrix.
        triplet_zero_ref = \
            1/4 * np.kron(E, E) - np.kron(Iz, Iz) + \
            1/2 * (np.kron(Ip, Im) + np.kron(Im, Ip))
        
        # Test with dense and sparse backends
        for sparse in (False, True):
            sg.parameters.sparse_state = sparse

            # Create the triplet-zero state using the inbuilt function
            triplet_zero = sg.triplet_zero_state(ss, 0, 1)

            # Convert the state to a density matrix
            triplet_zero = sg.state_to_zeeman(ss, triplet_zero)

            # Compare the constructed state with the reference.
            self.assertTrue(np.allclose(triplet_zero, triplet_zero_ref))

    def test_triplet_plus(self):
        """
        Test triplet-plus-state construction against a Zeeman reference.
        """

        # Reset the global settings.
        sg.parameters.default()

        # Use dense operators for the reference matrices.
        sg.parameters.sparse_operator = False

        # Build the example two-spin system.
        ss = build_spin_system(["1H", "1H"], 2)

        # Build the single-spin reference operators.
        E = sg.op_E(1/2)
        Iz = sg.op_Sz(1/2)

        # Create density matrix for the triplet-plus state for reference
        triplet_plus_ref = \
            1/4 * np.kron(E, E) + 1/2 * np.kron(E, Iz) + \
            1/2 * np.kron(Iz, E) + np.kron(Iz, Iz)
        
        # Test with dense and sparse backends
        for sparse in (False, True):
            sg.parameters.sparse_state = sparse

            # Create the triplet-plus state using the inbuilt function
            triplet_plus = sg.triplet_plus_state(ss, 0, 1)

            # Convert the state to a density matrix
            triplet_plus = sg.state_to_zeeman(ss, triplet_plus)

            # Compare the constructed state with the reference.
            self.assertTrue(np.allclose(triplet_plus, triplet_plus_ref))

    def test_triplet_minus(self):
        """
        Test triplet-minus-state construction against a Zeeman reference.
        """

        # Reset the global settings.
        sg.parameters.default()

        # Use dense operators for the reference matrices.
        sg.parameters.sparse_operator = False

        # Build the example two-spin system.
        ss = build_spin_system(["1H", "1H"], 2)

        # Build the single-spin reference operators.
        E = sg.op_E(1/2)
        Iz = sg.op_Sz(1/2)

        # Create the triplet-minus density matrix for reference
        triplet_minus_ref = \
            1/4 * np.kron(E, E) - 1/2 * np.kron(E, Iz) - \
            1/2 * np.kron(Iz, E) + np.kron(Iz, Iz)
        
        # Test with dense and sparse backends
        for sparse in (False, True):
            sg.parameters.sparse_state = sparse

            # Create the triplet-minus state using the inbuilt function
            triplet_minus = sg.triplet_minus_state(ss, 0, 1)

            # Convert the state to a density matrix
            triplet_minus = sg.state_to_zeeman(ss, triplet_minus)

            # Compare the constructed state with the reference.
            self.assertTrue(np.allclose(triplet_minus, triplet_minus_ref))

    def _test_state(
        self,
        ss: sg.SpinSystem,
        test_opers: list[dict[str, np.ndarray]],
        sparse_operator: bool,
        sparse_state: bool
    ) -> None:
        """
        Helper method for test_state().
        """
        # Set the state and operator sparsity.
        sg.parameters.sparse_operator = sparse_operator
        sg.parameters.sparse_state = sparse_state

        # Test the creation of states with all possible combinations
        for op1_k, op1_v in test_opers[0].items():
            for op2_k, op2_v in test_opers[1].items():

                # Create the state vector using inbuilt function
                op_string = f"{op1_k} * {op2_k}"
                state = sg.state(ss, op_string)

                # Create the reference density matrix
                state_ref = np.kron(op1_v, op2_v)

                # Convert the state to density matrix
                state = sg.state_to_zeeman(ss, state)

                # Convert the state to dense if necessary
                if sparse_operator:
                    state = state.toarray()

                # Compare to the reference
                self.assertTrue(np.allclose(state, state_ref))


    def test_state(self):
        """
        Test state construction from operator strings against Zeeman references.
        """

        # Reset to defaults
        sg.parameters.default()

        # Build the example spin system.
        ss = build_spin_system(["1H", "14N"], 2)

        # Create a set of operators to test
        sg.parameters.sparse_operator = False
        test_opers = []
        for i in range(ss.nspins):
            ops = {}

            # Unit operator
            ops["E"] = sg.op_E(ss.spins[i])

            # Cartesian operators
            ops[f"I(x, {i})"] = sg.op_Sx(ss.spins[i])
            ops[f"I(y, {i})"] = sg.op_Sy(ss.spins[i])
            ops[f"I(z, {i})"] = sg.op_Sz(ss.spins[i])

            # Ladder operators
            ops[f"I(+, {i})"] = sg.op_Sp(ss.spins[i])
            ops[f"I(-, {i})"] = sg.op_Sm(ss.spins[i])

            # Spherical tensor operators
            for l in range(int(2*ss.spins[i] + 1)):
                for q in range(-l, l + 1):
                    ops[f"T({l}, {q}, {i})"] = sg.op_T(ss.spins[i], l, q)
                    
            test_opers.append(ops)

        # Use the helper method to test all possible state combinations
        self._test_state(ss, test_opers, True, True)
        self._test_state(ss, test_opers, True, False)
        self._test_state(ss, test_opers, False, True)
        self._test_state(ss, test_opers, False, False)

    def test_unit_state(self):
        """
        Test unit-state construction with and without trace normalisation.
        """

        # Reset to default parameters
        sg.parameters.default()

        # Build the example spin system.
        ss = build_spin_system(["1H", "14N", "23Na"], 3)

        # Build the unit operator in the Zeeman basis.
        sg.parameters.sparse_operator = False
        unit_ref = np.array([[1]])
        for spin in ss.spins:
            unit_ref = np.kron(unit_ref, sg.op_E(spin))
        unit_ref_norm = unit_ref / unit_ref.trace()
        

        # Test with dense and sparse backends
        for sparse in (False, True):
            sg.parameters.sparse_state = sparse

            # Create the unit state using inbuilt function.
            unit = sg.unit_state(ss, normalized=False)
            unit_norm = sg.unit_state(ss, normalized=True)

            # Convert to density matrix
            unit = sg.state_to_zeeman(ss, unit)
            unit_norm = sg.state_to_zeeman(ss, unit_norm)
        
            # Compare the unit states with the reference.
            self.assertTrue(np.allclose(unit, unit_ref))
            self.assertTrue(np.allclose(unit_norm, unit_ref_norm))


    def _test_measure(
        self,
        ss: sg.SpinSystem,
        test_opers: list[dict[str, np.ndarray]],
        sparse_state: bool
    ) -> None:
        """
        Helper method for test_measure().
        """
        # Change the state sparsity
        sg.parameters.sparse_state = sparse_state

        # Collect all state combinations
        cases = []
        for op1_k, op1_v in test_opers[0].items():
            for op2_k, op2_v in test_opers[1].items():

                # Create the operator string and the reference operator
                op_string = f"{op1_k} * {op2_k}"
                op_ref = np.kron(op1_v, op2_v)

                cases.append((op_string, op_ref))

        # Create and measure each state
        for (state_string, state_oper) in cases:
            state = sg.state(ss, state_string)

            for (meas_string, meas_oper) in cases:

                # Obtain the reference measurement
                result_ref = (state_oper @ meas_oper.conj().T).trace()

                # Measure using the inbuilt function
                result = sg.measure(ss, state, meas_string)

                # Compare with the reference
                self.assertAlmostEqual(result_ref, result)

    def test_measure(self):
        """
        Test measurements against direct Zeeman-basis expectation values.
        """

        # Reset to defaults
        sg.parameters.default()

        # Build the example spin system with mixed spin quantum numbers.
        ss = build_spin_system(["1H", "14N"], 2)

        # Create a set of operators to test
        sg.parameters.sparse_operator = False
        test_opers = []
        for i in range(ss.nspins):
            ops = {}

            # Unit operator
            ops["E"] = sg.op_E(ss.spins[i])

            # Cartesian operators
            ops[f"I(x, {i})"] = sg.op_Sx(ss.spins[i])
            ops[f"I(y, {i})"] = sg.op_Sy(ss.spins[i])
            ops[f"I(z, {i})"] = sg.op_Sz(ss.spins[i])

            # Ladder operators
            ops[f"I(+, {i})"] = sg.op_Sp(ss.spins[i])
            ops[f"I(-, {i})"] = sg.op_Sm(ss.spins[i])

            # Spherical tensor operators
            for l in range(int(2*ss.spins[i] + 1)):
                for q in range(-l, l + 1):
                    ops[f"T({l}, {q}, {i})"] = sg.op_T(ss.spins[i], l, q)
                    
            test_opers.append(ops)
        
        # Test with the helper method using dense and sparse states
        self._test_measure(ss, test_opers, False)
        self._test_measure(ss, test_opers, True)

    def _test_equilibrium_state(
        self,
        ss: sg.SpinSystem,
        magnetic_field: float,
        temperature: float
    ):
        """
        Helper method for testing the equilibrium state.
        """
        # Set the experimental conditions
        sg.parameters.magnetic_field = magnetic_field
        sg.parameters.temperature = temperature
        
        # Construct the equilibrium state
        sg.parameters.sparse_state = False
        rho_eq_dense = sg.equilibrium_state(ss)
        sg.parameters.sparse_state = True
        rho_eq_sparse = sg.equilibrium_state(ss)

        # Test the thermal equilibrium for each spin
        for i in range(ss.nspins):

            # Measure the z-magnetization
            Mz_measured_dense = sg.measure(ss, rho_eq_dense, f"I(z, {i})")
            Mz_measured_sparse = sg.measure(ss, rho_eq_sparse, f"I(z, {i})")

            # Calculate a reference value for the magnetisation in equilibrium
            Mz_ref = _thermal_magnetization(
                gamma = ss.gammas[i],
                S = ss.spins[i],
                B = sg.parameters.magnetic_field,
                T = sg.parameters.temperature
            )

            # Compare
            self.assertAlmostEqual(Mz_measured_sparse, Mz_ref)
            self.assertAlmostEqual(Mz_measured_dense, Mz_ref)

    def test_equilibrium_state(self):
        """
        Compare the expectation value of Iz operator for the equilibrium
        state against the reference value calculated from Boltzmann
        distribution.
        """
        # Reset parameters
        sg.parameters.default()

        # Create an example spin system
        ss = sg.SpinSystem(["1H", "14N", "23Na", "17O"])

        # Create a basis set
        ss.basis.max_spin_order = 4
        ss.basis.build()

        # Test the equilibrium state using helper method
        self._test_equilibrium_state(ss, magnetic_field=9.4, temperature=293)
        self._test_equilibrium_state(ss, magnetic_field=100, temperature=1)

def _thermal_magnetization(gamma: float, S: float, B: float, T: float) -> float:
    """
    Calculates the magnetization at thermal equilibrium. Helper function for
    testing.
    
    Parameters
    ----------
    gamma : float
        Gyromagnetic ratio of the nucleus in rad/s/T.
    S : float
        Spin quantum number.
    B : float
        Magnetic field in Tesla.
    T : float
        Temperature in Kelvin.

    Returns
    -------
    magnetization : float
        Magnetization at thermal equilibrium.
    """
    # Get the possible spin magnetic quantum numbers (from largest to smallest)
    m = np.arange(-S, S + 1)

    # Get the populations of the states
    populations = {}
    for m_i in m:
        # Population according to the Boltzmann distribution
        numerator = np.exp(m_i * const.hbar * gamma * B / (const.k * T))
        denominator = sum(
            np.exp(m_j * const.hbar * gamma * B / (const.k * T)) for m_j in m)
        populations[m_i] = numerator / denominator

    # Calculate the polarization
    magnetization = sum(m_i * populations[m_i] for m_i in m)
    
    return magnetization