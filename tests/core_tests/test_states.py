import unittest
import numpy as np
import scipy.constants as const
import spinguin as sg

class TestStates(unittest.TestCase):

    def test_alpha(self):
        """
        Test creating the alpha state vector in the spherical tensor
        basis, converting that to Zeeman basis and compare with reference
        result.
        """
        # Reset to defaults
        sg.parameters.default()

        # Use dense format for operators
        sg.parameters.sparse_operator = False

        # Create an example spin system
        ss = sg.SpinSystem(["1H", "1H"])

        # Create a basis set
        ss.basis.max_spin_order = 2
        ss.basis.build()

        # Create Hilbert-space spin operators
        E = sg.op_E(1/2)
        Iz = sg.op_Sz(1/2)

        # Create density matrices for alpha states for reference
        alpha1_ref = 1/4 * np.kron(E, E) + 1/2 * np.kron(Iz, E)
        alpha2_ref = 1/4 * np.kron(E, E) + 1/2 * np.kron(E, Iz)

        # Create alpha states using the inbuilt function (dense format)
        sg.parameters.sparse_state = False
        alpha1_dense = sg.alpha_state(ss, 0)
        alpha2_dense = sg.alpha_state(ss, 1)

        # Create alpha states using the inbuilt function (sparse format)
        sg.parameters.sparse_state = True
        alpha1_sparse = sg.alpha_state(ss, 0)
        alpha2_sparse = sg.alpha_state(ss, 1)

        # Convert the states to density matrices (to dense format)
        alpha1_dense = sg.state_to_zeeman(ss, alpha1_dense)
        alpha2_dense = sg.state_to_zeeman(ss, alpha2_dense)
        alpha1_sparse = sg.state_to_zeeman(ss, alpha1_sparse)
        alpha2_sparse = sg.state_to_zeeman(ss, alpha2_sparse)

        # Compare
        self.assertTrue(np.allclose(alpha1_dense, alpha1_ref))
        self.assertTrue(np.allclose(alpha2_dense, alpha2_ref))
        self.assertTrue(np.allclose(alpha1_sparse, alpha1_ref))
        self.assertTrue(np.allclose(alpha2_sparse, alpha2_ref))

    def test_beta(self):
        """
        Test creating the beta state vector in the spherical tensor
        basis, converting that to Zeeman basis and compare with reference
        result.
        """
        # Reset to defaults
        sg.parameters.default()

        # Use dense format for operators
        sg.parameters.sparse_operator = False

        # Create an example spin system
        ss = sg.SpinSystem(["1H", "1H"])

        # Create a basis set
        ss.basis.max_spin_order = 2
        ss.basis.build()

        # Create Hilbert-space spin operators
        E = sg.op_E(1/2)
        Iz = sg.op_Sz(1/2)

        # Create Zeeman beta states
        beta1_ref = 1/4 * np.kron(E, E) - 1/2 * np.kron(Iz, E)
        beta2_ref = 1/4 * np.kron(E, E) - 1/2 * np.kron(E, Iz)

        # Create beta states using the inbuilt funtion (dense format)
        sg.parameters.sparse_state = False
        beta1_dense = sg.beta_state(ss, 0)
        beta2_dense = sg.beta_state(ss, 1)

        # Create beta states using the inbuilt funtion (sparse format)
        sg.parameters.sparse_state = True
        beta1_sparse = sg.beta_state(ss, 0)
        beta2_sparse = sg.beta_state(ss, 1)

        # Convert the states to density matrices (to dense format)
        beta1_dense = sg.state_to_zeeman(ss, beta1_dense)
        beta2_dense = sg.state_to_zeeman(ss, beta2_dense)
        beta1_sparse = sg.state_to_zeeman(ss, beta1_sparse)
        beta2_sparse = sg.state_to_zeeman(ss, beta2_sparse)

        # Compare
        self.assertTrue(np.allclose(beta1_dense, beta1_ref))
        self.assertTrue(np.allclose(beta2_dense, beta2_ref))
        self.assertTrue(np.allclose(beta1_sparse, beta1_ref))
        self.assertTrue(np.allclose(beta2_sparse, beta2_ref))

    def test_singlet(self):
        """
        Test creating the singlet state vector in the spherical tensor
        basis, converting that to Zeeman basis and compare with reference
        result.
        """
        # Reset to defaults
        sg.parameters.default()

        # Use dense format for operators
        sg.parameters.sparse_operator = False

        # Create an example spin system
        ss = sg.SpinSystem(["1H", "1H"])

        # Create a basis set
        ss.basis.max_spin_order = 2
        ss.basis.build()

        # Create Hilbert-space spin operators
        E = sg.op_E(1/2)
        Iz = sg.op_Sz(1/2)
        Ip = sg.op_Sp(1/2)
        Im = sg.op_Sm(1/2)

        # Create the singlet state density matrix for reference
        singlet_ref = 1/4 * np.kron(E, E) - np.kron(Iz, Iz) - \
                      1/2 * (np.kron(Ip, Im) + np.kron(Im, Ip))
        
        # Create the singlet state using the inbuilt function (dense format)
        sg.parameters.sparse_state = False
        singlet_dense = sg.singlet_state(ss, 0, 1)

        # Create the singlet state using the inbuilt function (sparse format)
        sg.parameters.sparse_state = True
        singlet_sparse = sg.singlet_state(ss, 0, 1)

        # Convert the states to density matrices
        singlet_dense = sg.state_to_zeeman(ss, singlet_dense)
        singlet_sparse = sg.state_to_zeeman(ss, singlet_sparse)

        # Compare
        self.assertTrue(np.allclose(singlet_ref, singlet_dense))
        self.assertTrue(np.allclose(singlet_ref, singlet_sparse))

    def test_triplet_zero(self):
        """
        Test creating the triplet zero state vector in the spherical tensor
        basis, converting that to Zeeman basis and compare with reference
        result.
        """
        # Reset to defaults
        sg.parameters.default()

        # Use dense format for operators
        sg.parameters.sparse_operator = False

        # Create an example spin system
        ss = sg.SpinSystem(["1H", "1H"])

        # Create a basis set
        ss.basis.max_spin_order = 2
        ss.basis.build()

        # Create Hilbert-space spin operators
        E = sg.op_E(1/2)
        Iz = sg.op_Sz(1/2)
        Ip = sg.op_Sp(1/2)
        Im = sg.op_Sm(1/2)

        # Create the triplet-zero density matrix for reference
        triplet_zero_ref = 1/4 * np.kron(E, E) - np.kron(Iz, Iz) + \
                           1/2 * (np.kron(Ip, Im) + np.kron(Im, Ip))
        
        # Create the triplet-zero state using the inbuilt function (dense)
        sg.parameters.sparse_state = False
        triplet_zero_dense = sg.triplet_zero_state(ss, 0, 1)

        # Create the triplet-zero state using the inbuilt function (sparse)
        sg.parameters.sparse_state = True
        triplet_zero_sparse = sg.triplet_zero_state(ss, 0, 1)

        # Convert the states to density matrices
        triplet_zero_dense = sg.state_to_zeeman(ss, triplet_zero_dense)
        triplet_zero_sparse = sg.state_to_zeeman(ss, triplet_zero_sparse)

        # Compare
        self.assertTrue(np.allclose(triplet_zero_ref, triplet_zero_dense))
        self.assertTrue(np.allclose(triplet_zero_ref, triplet_zero_sparse))

    def test_triplet_plus(self):
        """
        Test creating the triplet plus state vector in the spherical tensor
        basis, converting that to Zeeman basis and compare with reference
        result.
        """
        # Reset to defaults
        sg.parameters.default()

        # Use dense format for operators
        sg.parameters.sparse_operator = False

        # Create an example spin system
        ss = sg.SpinSystem(["1H", "1H"])

        # Create a basis set
        ss.basis.max_spin_order = 2
        ss.basis.build()

        # Create Hilbert-space spin operators
        E = sg.op_E(1/2)
        Iz = sg.op_Sz(1/2)

        # Create density matrix for the triplet-plus state for reference
        triplet_plus_ref = 1/4 * np.kron(E, E) + 1/2 * np.kron(E, Iz) + \
                           1/2 * np.kron(Iz, E) + np.kron(Iz, Iz)
        
        # Create the triplet-plus state using the inbuilt function (dense)
        sg.parameters.sparse_state = False
        triplet_plus_dense = sg.triplet_plus_state(ss, 0, 1)

        # Create the triplet-plus state using the inbuilt function (sparse)
        sg.parameters.sparse_state = True
        triplet_plus_sparse = sg.triplet_plus_state(ss, 0, 1)

        # Convert the states to density matrices
        triplet_plus_dense = sg.state_to_zeeman(ss, triplet_plus_dense)
        triplet_plus_sparse = sg.state_to_zeeman(ss, triplet_plus_sparse)

        # Compare
        self.assertTrue(np.allclose(triplet_plus_ref, triplet_plus_dense))
        self.assertTrue(np.allclose(triplet_plus_ref, triplet_plus_sparse))

    def test_triplet_minus(self):
        """
        Test creating the triplet minus state vector in the spherical tensor
        basis, converting that to Zeeman basis and compare with reference
        result.
        """
        # Reset to defaults
        sg.parameters.default()

        # Use dense format for operators
        sg.parameters.sparse_operator = False

        # Create an example spin system
        ss = sg.SpinSystem(["1H", "1H"])

        # Create a basis set
        ss.basis.max_spin_order = 2
        ss.basis.build()

        # Create Hilbert-space spin operators
        E = sg.op_E(1/2)
        Iz = sg.op_Sz(1/2)

        # Create the triplet-minus density matrix for reference
        triplet_minus_ref = 1/4 * np.kron(E, E) - 1/2 * np.kron(E, Iz) - \
                            1/2 * np.kron(Iz, E) + np.kron(Iz, Iz)
        
        # Create the triplet-minus state using the inbuilt function (dense)
        sg.parameters.sparse_state = False
        triplet_minus_dense = sg.triplet_minus_state(ss, 0, 1)

        # Create the triplet-minus state using the inbuilt function (sparse)
        sg.parameters.sparse_state = True
        triplet_minus_sparse = sg.triplet_minus_state(ss, 0, 1)

        # Convert to density matrices
        triplet_minus_dense = sg.state_to_zeeman(ss, triplet_minus_dense)
        triplet_minus_sparse = sg.state_to_zeeman(ss, triplet_minus_sparse)

        # Compare
        self.assertTrue(np.allclose(triplet_minus_ref, triplet_minus_dense))
        self.assertTrue(np.allclose(triplet_minus_ref, triplet_minus_sparse))

    def _test_state(
        self,
        ss: sg.SpinSystem,
        opers: dict,
        test_states: list,
        sparse_operator: bool,
        sparse_state: bool
    ):
        """
        Helper method for test_state().
        """
        # Set the sparsity
        sg.parameters.sparse_operator = sparse_operator
        sg.parameters.sparse_state = sparse_state

        # Test the creation of states with all possible combinations
        for i in test_states:
            if i == "E":
                op_i = "E"
            else:
                op_i = f"I({i}, 0)"

            for j in test_states:
                if j == "E":
                    op_j = "E"
                else:
                    op_j = f"I({j}, 1)"

                for k in test_states:
                    if k == "E":
                        op_k = "E"
                    else:
                        op_k = f"I({k}, 2)"

                    # Create the state vector in the spherical tensor basis
                    op_string = f"{op_i} * {op_j} * {op_k}"
                    state = sg.state(ss, op_string)

                    # Create the reference density matrix
                    state_ref = np.kron(
                        opers[(i, 1/2)], np.kron(opers[(j, 1)], opers[(k, 3/2)])
                    )

                    # Convert the state to density matrix
                    state = sg.state_to_zeeman(ss, state)

                    # Convert the state to dense if necessary
                    if sparse_operator:
                        state = state.toarray()

                    # Compare to the reference
                    self.assertTrue(np.allclose(state, state_ref))


    def test_state(self):
        """
        A test that creates various states for a spin system using the
        operator string. These states are converted to Zeeman eigenbasis
        and compared with reference.
        """
        # Reset to defaults
        sg.parameters.default()

        # Create an example spin system
        ss = sg.SpinSystem(["1H", "14N", "23Na"])

        # Build the basis set
        ss.basis.max_spin_order = 3
        ss.basis.build()

        # States to test
        test_states = ['E', 'x', 'y', 'z', '+', '-']

        # Get the Zeeman eigenbasis operators (as dense matrices)
        sg.parameters.sparse_operator = False
        opers = {}
        for spin in ss.spins:
            opers[('E', spin)] = sg.op_E(spin)
            opers[('x', spin)] = sg.op_Sx(spin)
            opers[('y', spin)] = sg.op_Sy(spin)
            opers[('z', spin)] = sg.op_Sz(spin)
            opers[('+', spin)] = sg.op_Sp(spin)
            opers[('-', spin)] = sg.op_Sm(spin)

        # Use the helper method to test all possible state combinations
        self._test_state(ss, opers, test_states, True, True)
        self._test_state(ss, opers, test_states, True, False)
        self._test_state(ss, opers, test_states, False, True)
        self._test_state(ss, opers, test_states, False, False)

    def test_unit_state(self):
        """
        A test that creates the unit state in the spherical tensor basis
        with the two normalization conventions and compares them to the
        expected reference values.
        """
        # Reset to default parameters
        sg.parameters.default()

        # Create an example spin system
        ss = sg.SpinSystem(["1H", "14N", "23Na"])

        # Build a basis set
        ss.basis.max_spin_order = 3
        ss.basis.build()

        # Create the unit state in the Zeeman eigenbasis
        sg.parameters.sparse_operator = False
        unit_zeeman = np.array([[1]])
        for spin in ss.spins:
            unit_zeeman = np.kron(unit_zeeman, sg.op_E(spin))

        # Create the non-normalized unit state in the spherical tensor basis
        sg.parameters.sparse_state = False
        unit_dense = sg.unit_state(ss, normalized=False)
        sg.parameters.sparse_state = True
        unit_sparse = sg.unit_state(ss, normalized=False)

        # Convert to density matrices
        unit_dense = sg.state_to_zeeman(ss, unit_dense)
        unit_sparse = sg.state_to_zeeman(ss, unit_sparse)
        
        # Compare
        self.assertTrue(np.allclose(unit_dense, unit_zeeman))
        self.assertTrue(np.allclose(unit_sparse, unit_zeeman))

        # Create the trace-normalized unit state in the spherical tensor basis
        sg.parameters.sparse_state = False
        unit_dense = sg.unit_state(ss, normalized=True)
        sg.parameters.sparse_state = True
        unit_sparse = sg.unit_state(ss, normalized=True)

        # Convert to density matrices
        unit_dense = sg.state_to_zeeman(ss, unit_dense)
        unit_sparse = sg.state_to_zeeman(ss, unit_sparse)

        # Apply trace normalization to the unit state in the Zeeman eigenbasis
        unit_zeeman = unit_zeeman / unit_zeeman.trace()

        # Compare
        self.assertTrue(np.allclose(unit_dense, unit_zeeman))
        self.assertTrue(np.allclose(unit_sparse, unit_zeeman))

    def _test_measure(
        self,
        ss: sg.SpinSystem,
        test_states: list,
        opers: dict,
        sparse_state: bool
    ):
        """
        Helper method for test_measure().
        """
        # Change the state sparsity
        sg.parameters.sparse_state = sparse_state

        # Try all possible state combinations
        for i in test_states:
            if i == "E":
                op_i = "E"
            else:
                op_i = f"I({i}, 0)"

            for j in test_states:
                if j == "E":
                    op_j = "E"
                else:
                    op_j = f"I({j}, 1)"

                for k in test_states:
                    if k == "E":
                        op_k = "E"
                    else:
                        op_k = f"I({k}, 2)"

                    # Create the state vector in the spherical tensor basis
                    op_string = f"{op_i} * {op_j} * {op_k}"
                    state = sg.state(ss, op_string)

                    # Create the density matrices in the Zeeman eigenbasis
                    op1, op2, op3 = \
                        opers[(i, 1/2)], opers[(j, 1)], opers[(k, 3/2)]
                    state_zeeman = np.kron(op1, np.kron(op2, op3))

                    # Measure each state combination
                    for l in test_states:
                        if l == 'E':
                            op_l = 'E'
                        else:
                            op_l = f"I({l}, 0)"

                        for m in test_states:
                            if m == 'E':
                                op_m = 'E'
                            else:
                                op_m = f"I({m}, 1)"

                            for n in test_states:
                                if n == 'E':
                                    op_n = 'E'
                                else:
                                    op_n = f"I({n}, 2)"

                                # Measure using the Zeeman eigenbasis
                                op1, op2, op3 = opers[(l, 1/2)], \
                                                opers[(m, 1)], \
                                                opers[(n, 3/2)]
                                oper_zeeman = np.kron(op1, np.kron(op2, op3))
                                result_zeeman = (
                                    state_zeeman @ oper_zeeman.conj().T
                                ).trace()

                                # Measure using the inbuilt function
                                op_string = f"{op_l} * {op_m} * {op_n}"
                                result = sg.measure(ss, state, op_string)

                                # Compare
                                self.assertAlmostEqual(result_zeeman, result)

    def test_measure(self):
        """
        Different states are created for a spin system using both
        the Zeeman eigenbasis and spherical tensor basis, and the
        expectation values for varying operators are compared.
        """
        # Reset to defaults
        sg.parameters.default()

        # Create an example spin system with different spin quantum numbers
        ss = sg.SpinSystem(["1H", "14N", "23Na"])
        ss.basis.max_spin_order = 3
        ss.basis.build()

        # States to test
        test_states = ['E', 'x', 'y', 'z', '+', '-']

        # Get the Zeeman eigenbasis operators in dense format
        sg.parameters.sparse_operator = False
        opers = {}
        for spin in ss.spins:
            opers[('E', spin)] = sg.op_E(spin)
            opers[('x', spin)] = sg.op_Sx(spin)
            opers[('y', spin)] = sg.op_Sy(spin)
            opers[('z', spin)] = sg.op_Sz(spin)
            opers[('+', spin)] = sg.op_Sp(spin)
            opers[('-', spin)] = sg.op_Sm(spin)
        
        # Test with the helper method using dense and sparse states
        self._test_measure(ss, test_states, opers, False)
        self._test_measure(ss, test_states, opers, True)

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