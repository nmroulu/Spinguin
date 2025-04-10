import unittest
import numpy as np
from spinguin._spin_system import SpinSystem
from spinguin import _operators, _states
import scipy.constants as const

class TestStates(unittest.TestCase):

    def test_alpha(self):
        # Create an example spin system
        isotopes = np.array(['1H', '1H'])
        spin_system = SpinSystem(isotopes)

        # Create Zeeman operators
        E = _operators.op_E(1/2)
        Iz = _operators.op_Sz(1/2)

        # Create Zeeman alpha states
        alpha1_zeeman = 1/4 * np.kron(E, E) + 1/2 * np.kron(Iz, E)
        alpha2_zeeman = 1/4 * np.kron(E, E) + 1/2 * np.kron(E, Iz)

        # Create alpha states in the spherical tensor basis
        alpha1_sparse = _states.alpha(spin_system, 0, sparse=True)
        alpha2_sparse = _states.alpha(spin_system, 1, sparse=True)
        alpha1_dense = _states.alpha(spin_system, 0, sparse=False)
        alpha2_dense = _states.alpha(spin_system, 1, sparse=False)

        # Compare
        self.assertTrue(np.allclose(_states.rho_to_zeeman(spin_system, alpha1_sparse), alpha1_zeeman))
        self.assertTrue(np.allclose(_states.rho_to_zeeman(spin_system, alpha2_sparse), alpha2_zeeman))
        self.assertTrue(np.allclose(_states.rho_to_zeeman(spin_system, alpha1_dense), alpha1_zeeman))
        self.assertTrue(np.allclose(_states.rho_to_zeeman(spin_system, alpha2_dense), alpha2_zeeman))

    def test_beta(self):
        # Create an example spin system
        isotopes = np.array(['1H', '1H'])
        spin_system = SpinSystem(isotopes)

        # Create Zeeman operators
        E = _operators.op_E(1/2)
        Iz = _operators.op_Sz(1/2)

        # Create Zeeman beta states
        beta1_zeeman = 1/4 * np.kron(E, E) - 1/2 * np.kron(Iz, E)
        beta2_zeeman = 1/4 * np.kron(E, E) - 1/2 * np.kron(E, Iz)

        # Create beta states in the spherical tensor basis
        beta1_sparse = _states.beta(spin_system, 0, sparse=True)
        beta2_sparse = _states.beta(spin_system, 1, sparse=True)
        beta1_dense = _states.beta(spin_system, 0, sparse=False)
        beta2_dense = _states.beta(spin_system, 1, sparse=False)

        # Compare
        self.assertTrue(np.allclose(_states.rho_to_zeeman(spin_system, beta1_sparse), beta1_zeeman))
        self.assertTrue(np.allclose(_states.rho_to_zeeman(spin_system, beta2_sparse), beta2_zeeman))
        self.assertTrue(np.allclose(_states.rho_to_zeeman(spin_system, beta1_dense), beta1_zeeman))
        self.assertTrue(np.allclose(_states.rho_to_zeeman(spin_system, beta2_dense), beta2_zeeman))

    def test_singlet(self):
        # Create an example spin system
        isotopes = np.array(['1H', '1H'])
        spin_system = SpinSystem(isotopes)

        # Create Zeeman operators
        E = _operators.op_E(1/2)
        Iz = _operators.op_Sz(1/2)
        Ip = _operators.op_Sp(1/2)
        Im = _operators.op_Sm(1/2)

        # Create the singlet state and compare
        singlet_zeeman = 1/4 * np.kron(E, E) - np.kron(Iz, Iz) - 1/2 * (np.kron(Ip, Im) + np.kron(Im, Ip))
        singlet_sparse = _states.singlet(spin_system, 0, 1, sparse=True)
        singlet_dense = _states.singlet(spin_system, 0, 1, sparse=False)
        self.assertTrue(np.allclose(singlet_zeeman, _states.rho_to_zeeman(spin_system, singlet_sparse)))
        self.assertTrue(np.allclose(singlet_zeeman, _states.rho_to_zeeman(spin_system, singlet_dense)))

    def test_triplet_zero(self):
        # Create an example spin system
        isotopes = np.array(['1H', '1H'])
        spin_system = SpinSystem(isotopes)

        # Create Zeeman operators
        E = _operators.op_E(1/2)
        Iz = _operators.op_Sz(1/2)
        Ip = _operators.op_Sp(1/2)
        Im = _operators.op_Sm(1/2)

        # Create the triplet-zero state and compare
        triplet_zero_zeeman = 1/4 * np.kron(E, E) - np.kron(Iz, Iz) + 1/2 * (np.kron(Ip, Im) + np.kron(Im, Ip))
        triplet_zero_sparse = _states.triplet_zero(spin_system, 0, 1, sparse=True)
        triplet_zero_dense = _states.triplet_zero(spin_system, 0, 1, sparse=False)
        self.assertTrue(np.allclose(triplet_zero_zeeman, _states.rho_to_zeeman(spin_system, triplet_zero_sparse)))
        self.assertTrue(np.allclose(triplet_zero_zeeman, _states.rho_to_zeeman(spin_system, triplet_zero_dense)))

    def test_triplet_plus(self):
        # Create an example spin system
        isotopes = np.array(['1H', '1H'])
        spin_system = SpinSystem(isotopes)

        # Create Zeeman operators
        E = _operators.op_E(1/2)
        Iz = _operators.op_Sz(1/2)

        # Create the triplet-plus state and compare
        triplet_plus_zeeman = 1/4 * np.kron(E, E) + 1/2 * np.kron(E, Iz) + 1/2 * np.kron(Iz, E) + np.kron(Iz, Iz)
        triplet_plus_sparse = _states.triplet_plus(spin_system, 0, 1, sparse=True)
        triplet_plus_dense = _states.triplet_plus(spin_system, 0, 1, sparse=False)
        self.assertTrue(np.allclose(triplet_plus_zeeman, _states.rho_to_zeeman(spin_system, triplet_plus_sparse)))
        self.assertTrue(np.allclose(triplet_plus_zeeman, _states.rho_to_zeeman(spin_system, triplet_plus_dense)))

    def test_triplet_minus(self):
        # Create an example spin system
        isotopes = np.array(['1H', '1H'])
        spin_system = SpinSystem(isotopes)

        # Create Zeeman operators
        E = _operators.op_E(1/2)
        Iz = _operators.op_Sz(1/2)

        # Create the triplet-minus state and compare
        triplet_minus_zeeman = 1/4 * np.kron(E, E) - 1/2 * np.kron(E, Iz) - 1/2 * np.kron(Iz, E) + np.kron(Iz, Iz)
        triplet_minus_sparse = _states.triplet_minus(spin_system, 0, 1, sparse=True)
        triplet_minus_dense = _states.triplet_minus(spin_system, 0, 1, sparse=False)
        self.assertTrue(np.allclose(triplet_minus_zeeman, _states.rho_to_zeeman(spin_system, triplet_minus_sparse)))
        self.assertTrue(np.allclose(triplet_minus_zeeman, _states.rho_to_zeeman(spin_system, triplet_minus_dense)))

    def test_state_and_rho_to_zeeman(self):
        # Create an example spin system with different spin quantum numbers
        isotopes = np.array(['1H', '14N', '23Na'])
        spin_system = SpinSystem(isotopes)
        spins = spin_system.spins

        # States to test
        test_states = ['E', 'I_x', 'I_y', 'I_z', 'I_+', 'I_-']

        # Get the Zeeman eigenbasis operators
        opers = {}
        for spin in spins:
            opers[('E', spin)] = _operators.op_E(spin)
            opers[('I_x', spin)] = _operators.op_Sx(spin)
            opers[('I_y', spin)] = _operators.op_Sy(spin)
            opers[('I_z', spin)] = _operators.op_Sz(spin)
            opers[('I_+', spin)] = _operators.op_Sp(spin)
            opers[('I_-', spin)] = _operators.op_Sm(spin)

        # Try all possible state combinations
        for i in test_states:
            for j in test_states:
                for k in test_states:
                    # Create the state vector in the spherical tensor basis
                    state_sparse = _states.state(spin_system, (i, j, k), (0, 1, 2), sparse=True)
                    state_dense = _states.state(spin_system, (i, j, k), (0, 1, 2), sparse=False)

                    # Create the density matrix in the Zeeman eigenbasis
                    state_zeeman = np.kron(opers[(i, 1/2)], np.kron(opers[(j, 1)], opers[(k, 3/2)]))

                    # Convert state vectors to Zeeman and compare
                    self.assertTrue(np.allclose(_states.rho_to_zeeman(spin_system, state_sparse), state_zeeman))
                    self.assertTrue(np.allclose(_states.rho_to_zeeman(spin_system, state_dense), state_zeeman))

    def test_unit_state(self):
        # Create an example spin system with different spin quantum numbers
        isotopes = np.array(['1H', '14N', '23Na'])
        spin_system = SpinSystem(isotopes)
        spins = spin_system.spins

        # Create the unit state in the Zeeman eigenbasis
        unit_zeeman = 1
        for spin in spins:
            unit_zeeman = np.kron(unit_zeeman, _operators.op_E(spin))

        # Create the non-normalized unit state in the spherical tensor basis
        unit_sparse = _states.unit_state(spin_system, sparse=True, normalized=False)
        unit_dense = _states.unit_state(spin_system, sparse=False, normalized=False)

        # Compare
        self.assertTrue(np.allclose(_states.rho_to_zeeman(spin_system, unit_sparse), unit_zeeman))
        self.assertTrue(np.allclose(_states.rho_to_zeeman(spin_system, unit_dense), unit_zeeman))

        # Create the trace-normalized unit state in the spherical tensor basis
        unit_sparse = _states.unit_state(spin_system, sparse=True, normalized=True)
        unit_dense = _states.unit_state(spin_system, sparse=False, normalized=True)

        # Apply trace normalization to the unit state in the Zeeman eigenbasis
        unit_zeeman = unit_zeeman / unit_zeeman.trace()

        # Compare
        self.assertTrue(np.allclose(_states.rho_to_zeeman(spin_system, unit_sparse), unit_zeeman))
        self.assertTrue(np.allclose(_states.rho_to_zeeman(spin_system, unit_dense), unit_zeeman))

    def test_measure(self):
        # Create an example spin system
        isotopes = np.array(['1H', '14N', '23Na'])
        spin_system = SpinSystem(isotopes)
        spins = spin_system.spins

        # States to test
        test_states = ['E', 'I_x', 'I_y', 'I_z', 'I_+', 'I_-']

        # Get the Zeeman eigenbasis operators
        opers = {}
        for spin in spins:
            opers[('E', spin)] = _operators.op_E(spin)
            opers[('I_x', spin)] = _operators.op_Sx(spin)
            opers[('I_y', spin)] = _operators.op_Sy(spin)
            opers[('I_z', spin)] = _operators.op_Sz(spin)
            opers[('I_+', spin)] = _operators.op_Sp(spin)
            opers[('I_-', spin)] = _operators.op_Sm(spin)

        # Try all possible state combinations
        for i in test_states:
            for j in test_states:
                for k in test_states:
                    # Create the state vector
                    state_sparse = _states.state(spin_system, (i, j, k), (0, 1, 2), sparse=True)
                    state_dense = _states.state(spin_system, (i, j, k), (0, 1, 2), sparse=False)

                    # Create the density matrices in the Zeeman eigenbasis
                    op1, op2, op3 = opers[(i, 1/2)], opers[(j, 1)], opers[(k, 3/2)]
                    state_zeeman = np.kron(op1, np.kron(op2, op3))

                    # Measure each state combination
                    for l in test_states:
                        for m in test_states:
                            for n in test_states:
                                # Measure using the Zeeman eigenbasis
                                op1, op2, op3 = opers[(l, 1/2)], opers[(m, 1)], opers[(n, 3/2)]
                                oper_zeeman = np.kron(op1, np.kron(op2, op3))
                                result_zeeman = (state_zeeman @ oper_zeeman.conj().T).trace()

                                # Measure using the inbuilt function
                                result_sparse = _states.measure(spin_system, state_sparse, (l, m, n), (0, 1, 2))
                                result_dense = _states.measure(spin_system, state_dense, (l, m, n), (0, 1, 2))

                                # Compare
                                self.assertAlmostEqual(result_zeeman, result_sparse)
                                self.assertAlmostEqual(result_zeeman, result_dense)

    def test_rho_thermal_equilibrium(self):
        # Create an example spin system with different spin quantum numbers
        isotopes = np.array(['1H', '14N', '23Na', '17O'])
        spin_system = SpinSystem(isotopes)
        gammas = spin_system.gammas
        spins = spin_system.spins
        size = spin_system.size

        # Conditions
        field = 14.1
        temperature = 273

        # Make the thermal equilibrium state
        rho_eq_sparse = _states.rho_thermal_equilibrium(spin_system, temperature, field, sparse=True)
        rho_eq_dense = _states.rho_thermal_equilibrium(spin_system, temperature, field, sparse=False)

        # Test the thermal equilibrium for each spin
        for i in range(size):
            # Measure the magnetization
            Mz_measured_sparse = _states.measure(spin_system, rho_eq_sparse, 'I_z', i)
            Mz_measured_dense = _states.measure(spin_system, rho_eq_dense, 'I_z', i)

            # Calculate the thermal magnetization directly using the Boltzmann distribution
            Mz_calculated = thermal_magnetization(gammas[i], spins[i], field, temperature)

            # Compare
            self.assertAlmostEqual(Mz_measured_sparse, Mz_calculated)
            self.assertAlmostEqual(Mz_measured_dense, Mz_calculated)

        # Test that the code is future-proof (extreme conditions)
        field = 1000
        temperature = 1

        # Make the thermal equilibrium state
        rho_eq_sparse = _states.rho_thermal_equilibrium(spin_system, temperature, field, sparse=True)
        rho_eq_dense = _states.rho_thermal_equilibrium(spin_system, temperature, field, sparse=False)

        # Test the thermal equilibrium for each spin
        for i in range(size):
            # Measure the magnetization
            Mz_measured_sparse = _states.measure(spin_system, rho_eq_sparse, 'I_z', i)
            Mz_measured_dense = _states.measure(spin_system, rho_eq_dense, 'I_z', i)

            # Calculate the thermal magnetization directly using the Boltzmann distribution
            Mz_calculated = thermal_magnetization(gammas[i], spins[i], field, temperature)

            # Compare
            self.assertAlmostEqual(Mz_measured_sparse, Mz_calculated)
            self.assertAlmostEqual(Mz_measured_dense, Mz_calculated)

def thermal_magnetization(gamma: float, S: float, B: float, T: float) -> float:
    """
    Calculates the magnetization at thermal equilibrium. Helper function for testing.
    
    Parameters
    ----------
    gamma : float
        Gyromagnetic ratio of the nucleus in Hz/T.
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
        denominator = sum(np.exp(m_j * const.hbar * gamma * B / (const.k * T)) for m_j in m)
        populations[m_i] = numerator / denominator

    # Calculate the polarization
    magnetization = sum(m_i * populations[m_i] for m_i in m)
    
    return magnetization
