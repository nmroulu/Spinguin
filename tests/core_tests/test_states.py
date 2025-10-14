import unittest
import numpy as np
import scipy.constants as const
from spinguin._core.operators import op_E, op_Sx, op_Sy, op_Sz, op_Sp, op_Sm
from spinguin._core.states import \
    alpha_state, beta_state, state_to_zeeman, singlet_state, \
    triplet_zero_state, triplet_plus_state, triplet_minus_state, \
    state_from_string, unit_state, measure, equilibrium_state
from spinguin._core.basis import make_basis
from spinguin._core.nmr_isotopes import ISOTOPES
from spinguin._core.hamiltonian import sop_H_Z

class TestStates(unittest.TestCase):

    def test_alpha(self):
        """
        Test creating the alpha state vector in the spherical tensor
        basis, converting that to Zeeman basis and compare with reference
        result.
        """

        # Create an example spin system
        spins = np.array([1/2, 1/2])
        max_spin_order = 2
        basis = make_basis(spins, max_spin_order)

        # Create Zeeman operators
        E = op_E(1/2, sparse=False)
        Iz = op_Sz(1/2, sparse=False)

        # Create Zeeman alpha states for reference
        alpha1_zeeman = 1/4 * np.kron(E, E) + 1/2 * np.kron(Iz, E)
        alpha2_zeeman = 1/4 * np.kron(E, E) + 1/2 * np.kron(E, Iz)

        # Create alpha states in the spherical tensor basis
        alpha1_sparse = alpha_state(basis, spins, 0, sparse=True)
        alpha2_sparse = alpha_state(basis, spins, 1, sparse=True)
        alpha1_dense = alpha_state(basis, spins, 0, sparse=False)
        alpha2_dense = alpha_state(basis, spins, 1, sparse=False)

        # Compare
        self.assertTrue(np.allclose(
            state_to_zeeman(basis, spins, alpha1_sparse, sparse=False),
            alpha1_zeeman))
        self.assertTrue(np.allclose(
            state_to_zeeman(basis, spins, alpha2_sparse, sparse=False),
            alpha2_zeeman))
        self.assertTrue(np.allclose(
            state_to_zeeman(basis, spins, alpha1_dense, sparse=False),
            alpha1_zeeman))
        self.assertTrue(np.allclose(
            state_to_zeeman(basis, spins, alpha2_dense, sparse=False),
            alpha2_zeeman))

    def test_beta(self):
        """
        Test creating the beta state vector in the spherical tensor
        basis, converting that to Zeeman basis and compare with reference
        result.
        """

        # Create an example spin system
        spins = np.array([1/2, 1/2])
        max_spin_order = 2
        basis = make_basis(spins, max_spin_order)

        # Create Zeeman operators
        E = op_E(1/2, sparse=False)
        Iz = op_Sz(1/2, sparse=False)

        # Create Zeeman beta states
        beta1_zeeman = 1/4 * np.kron(E, E) - 1/2 * np.kron(Iz, E)
        beta2_zeeman = 1/4 * np.kron(E, E) - 1/2 * np.kron(E, Iz)

        # Create beta states in the spherical tensor basis
        beta1_sparse = beta_state(basis, spins, 0, sparse=True)
        beta2_sparse = beta_state(basis, spins, 1, sparse=True)
        beta1_dense = beta_state(basis, spins, 0, sparse=False)
        beta2_dense = beta_state(basis, spins, 1, sparse=False)

        # Compare
        self.assertTrue(np.allclose(
            state_to_zeeman(basis, spins, beta1_sparse, sparse=False),
            beta1_zeeman))
        self.assertTrue(np.allclose(
            state_to_zeeman(basis, spins, beta2_sparse, sparse=False),
            beta2_zeeman))
        self.assertTrue(np.allclose(
            state_to_zeeman(basis, spins, beta1_dense, sparse=False),
            beta1_zeeman))
        self.assertTrue(np.allclose(
            state_to_zeeman(basis, spins, beta2_dense, sparse=False),
            beta2_zeeman))

    def test_singlet(self):
        """
        Test creating the singlet state vector in the spherical tensor
        basis, converting that to Zeeman basis and compare with reference
        result.
        """

        # Create an example spin system
        spins = np.array([1/2, 1/2])
        max_spin_order = 2
        basis = make_basis(spins, max_spin_order)

        # Create Zeeman operators
        E = op_E(1/2, sparse=False)
        Iz = op_Sz(1/2, sparse=False)
        Ip = op_Sp(1/2, sparse=False)
        Im = op_Sm(1/2, sparse=False)

        # Create the singlet state in both bases and compare
        singlet_zeeman = 1/4 * np.kron(E, E) - np.kron(Iz, Iz) - \
                         1/2 * (np.kron(Ip, Im) + np.kron(Im, Ip))
        singlet_sparse = singlet_state(basis, spins, 0, 1, sparse=True)
        singlet_dense = singlet_state(basis, spins, 0, 1, sparse=False)
        self.assertTrue(np.allclose(
            singlet_zeeman,
            state_to_zeeman(basis, spins, singlet_sparse, sparse=False)))
        self.assertTrue(np.allclose(
            singlet_zeeman,
            state_to_zeeman(basis, spins, singlet_dense, sparse=False)))

    def test_triplet_zero(self):
        """
        Test creating the triplet zero state vector in the spherical tensor
        basis, converting that to Zeeman basis and compare with reference
        result.
        """

        # Create an example spin system
        spins = np.array([1/2, 1/2])
        max_spin_order = 2
        basis = make_basis(spins, max_spin_order)

        # Create Zeeman operators
        E = op_E(1/2, sparse=False)
        Iz = op_Sz(1/2, sparse=False)
        Ip = op_Sp(1/2, sparse=False)
        Im = op_Sm(1/2, sparse=False)

        # Create the triplet-zero state and compare
        triplet_zero_zeeman = 1/4 * np.kron(E, E) - np.kron(Iz, Iz) + \
                              1/2 * (np.kron(Ip, Im) + np.kron(Im, Ip))
        triplet_zero_sparse = triplet_zero_state(basis, spins, 0, 1,
                                                 sparse=True)
        triplet_zero_dense = triplet_zero_state(basis, spins, 0, 1,
                                                sparse=False)
        self.assertTrue(np.allclose(
            triplet_zero_zeeman,
            state_to_zeeman(basis, spins, triplet_zero_sparse, sparse=False)))
        self.assertTrue(np.allclose(
            triplet_zero_zeeman,
            state_to_zeeman(basis, spins, triplet_zero_dense, sparse=False)))

    def test_triplet_plus(self):
        """
        Test creating the triplet plus state vector in the spherical tensor
        basis, converting that to Zeeman basis and compare with reference
        result.
        """

        # Create an example spin system
        spins = np.array([1/2, 1/2])
        max_spin_order = 2
        basis = make_basis(spins, max_spin_order)

        # Create Zeeman operators
        E = op_E(1/2, sparse=False)
        Iz = op_Sz(1/2, sparse=False)

        # Create the triplet-plus state and compare
        triplet_plus_zeeman = 1/4 * np.kron(E, E) + 1/2 * np.kron(E, Iz) + \
                              1/2 * np.kron(Iz, E) + np.kron(Iz, Iz)
        triplet_plus_sparse = triplet_plus_state(basis, spins, 0, 1,
                                                 sparse=True)
        triplet_plus_dense = triplet_plus_state(basis, spins, 0, 1,
                                                sparse=False)
        self.assertTrue(np.allclose(
            triplet_plus_zeeman,
            state_to_zeeman(basis, spins, triplet_plus_sparse, sparse=False)))
        self.assertTrue(np.allclose(
            triplet_plus_zeeman,
            state_to_zeeman(basis, spins, triplet_plus_dense, sparse=False)))

    def test_triplet_minus(self):
        """
        Test creating the triplet minus state vector in the spherical tensor
        basis, converting that to Zeeman basis and compare with reference
        result.
        """

        # Create an example spin system
        spins = np.array([1/2, 1/2])
        max_spin_order = 2
        basis = make_basis(spins, max_spin_order)

        # Create Zeeman operators
        E = op_E(1/2, sparse=False)
        Iz = op_Sz(1/2, sparse=False)

        # Create the triplet-minus state and compare
        triplet_minus_zeeman = 1/4 * np.kron(E, E) - 1/2 * np.kron(E, Iz) - \
                               1/2 * np.kron(Iz, E) + np.kron(Iz, Iz)
        triplet_minus_sparse = triplet_minus_state(basis, spins, 0, 1,
                                                   sparse=True)
        triplet_minus_dense = triplet_minus_state(basis, spins, 0, 1,
                                                  sparse=False)
        self.assertTrue(np.allclose(
            triplet_minus_zeeman,
            state_to_zeeman(basis, spins, triplet_minus_sparse, sparse=False)))
        self.assertTrue(np.allclose(
            triplet_minus_zeeman,
            state_to_zeeman(basis, spins, triplet_minus_dense, sparse=False)))

    def test_state_from_string_and_rho_to_zeeman(self):
        """
        A test that creates various states for a spin system using the
        operator string. These states are converted to Zeeman eigenbasis
        and compared with reference.
        """

        # Create an example spin system with different spin quantum numbers
        spins = np.array([1/2, 1, 3/2])
        max_spin_order = 3
        basis = make_basis(spins, max_spin_order)

        # States to test
        test_states = ['E', 'x', 'y', 'z', '+', '-']

        # Get the Zeeman eigenbasis operators
        opers = {}
        for spin in spins:
            opers[('E', spin)] = op_E(spin, sparse=False)
            opers[('x', spin)] = op_Sx(spin, sparse=False)
            opers[('y', spin)] = op_Sy(spin, sparse=False)
            opers[('z', spin)] = op_Sz(spin, sparse=False)
            opers[('+', spin)] = op_Sp(spin, sparse=False)
            opers[('-', spin)] = op_Sm(spin, sparse=False)

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
                    state_sparse = state_from_string(basis, spins, op_string,
                                                     sparse=True)
                    state_dense = state_from_string(basis, spins, op_string,
                                                    sparse=False)

                    # Create the density matrix in the Zeeman eigenbasis
                    state_zeeman = np.kron(opers[(i, 1/2)],
                                           np.kron(opers[(j, 1)],
                                                   opers[(k, 3/2)]))

                    # Test the conversion using all possible sparsity combinations
                    state_sparse_to_zeeman_dense = state_to_zeeman(
                        basis, spins, state_sparse, sparse=False)
                    state_sparse_to_zeeman_sparse = state_to_zeeman(
                        basis, spins, state_sparse, sparse=True).toarray()
                    state_dense_to_zeeman_dense = state_to_zeeman(
                        basis, spins, state_dense, sparse=False)
                    state_dense_to_zeeman_sparse = state_to_zeeman(
                        basis, spins, state_dense, sparse=True).toarray()

                    # Compare to the Zeeman reference
                    self.assertTrue(np.allclose(state_sparse_to_zeeman_dense,
                                                state_zeeman))
                    self.assertTrue(np.allclose(state_sparse_to_zeeman_sparse,
                                                state_zeeman))
                    self.assertTrue(np.allclose(state_dense_to_zeeman_dense,
                                                state_zeeman))
                    self.assertTrue(np.allclose(state_dense_to_zeeman_sparse,
                                                state_zeeman))

    def test_unit_state(self):
        """
        A test that creates the unit state in the spherical tensor basis
        with the two normalization conventions and compares them to the
        expected reference values.
        """

        # Create an example spin system with different spin quantum numbers
        spins = np.array([1/2, 1, 3/2])
        max_spin_order = 3
        basis = make_basis(spins, max_spin_order)

        # Create the unit state in the Zeeman eigenbasis
        unit_zeeman = np.array([[1]])
        for spin in spins:
            unit_zeeman = np.kron(unit_zeeman, op_E(spin, sparse=False))

        # Create the non-normalized unit state in the spherical tensor basis
        unit_sparse = unit_state(basis, spins, sparse=True, normalized=False)
        unit_dense = unit_state(basis, spins, sparse=False, normalized=False)

        # Compare
        self.assertTrue(np.allclose(
            state_to_zeeman(basis, spins, unit_sparse, sparse=False),
            unit_zeeman))
        self.assertTrue(np.allclose(
            state_to_zeeman(basis, spins, unit_dense, sparse=False),
            unit_zeeman))

        # Create the trace-normalized unit state in the spherical tensor basis
        unit_sparse = unit_state(basis, spins, sparse=True, normalized=True)
        unit_dense = unit_state(basis, spins, sparse=False, normalized=True)

        # Apply trace normalization to the unit state in the Zeeman eigenbasis
        unit_zeeman = unit_zeeman / unit_zeeman.trace()

        # Compare
        self.assertTrue(np.allclose(
            state_to_zeeman(basis, spins, unit_sparse, sparse=False),
            unit_zeeman))
        self.assertTrue(np.allclose(
            state_to_zeeman(basis, spins, unit_dense, sparse=False),
            unit_zeeman))

    def test_measure(self):
        """
        Different states are created for a spin system using both
        the Zeeman eigenbasis and spherical tensor basis, and the
        expectation values for varying operators are compared.
        """

        # Create an example spin system with different spin quantum numbers
        spins = np.array([1/2, 1, 3/2])
        max_spin_order = 3
        basis = make_basis(spins, max_spin_order)

        # States to test
        test_states = ['E', 'x', 'y', 'z', '+', '-']

        # Get the Zeeman eigenbasis operators
        opers = {}
        for spin in spins:
            opers[('E', spin)] = op_E(spin, sparse=False)
            opers[('x', spin)] = op_Sx(spin, sparse=False)
            opers[('y', spin)] = op_Sy(spin, sparse=False)
            opers[('z', spin)] = op_Sz(spin, sparse=False)
            opers[('+', spin)] = op_Sp(spin, sparse=False)
            opers[('-', spin)] = op_Sm(spin, sparse=False)

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
                    state_sparse = state_from_string(basis, spins, op_string,
                                                     sparse=True)
                    state_dense = state_from_string(basis, spins, op_string,
                                                    sparse=False)

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
                                    state_zeeman @ oper_zeeman.conj().T).trace()

                                # Measure using the inbuilt function
                                op_string = f"{op_l} * {op_m} * {op_n}"
                                result_sparse = measure(
                                    basis, spins, state_sparse, op_string)
                                result_dense = measure(
                                    basis, spins, state_dense, op_string)

                                # Compare
                                self.assertAlmostEqual(result_zeeman,
                                                       result_sparse)
                                self.assertAlmostEqual(result_zeeman,
                                                       result_dense)

    def test_equilibrium_state(self):
        """
        Compare the expectation value of Iz operator for the equilibrium
        state against the reference value calculated from Boltzmann
        distribution.
        """

        # Create an example spin system with different spin quantum numbers
        spins = np.array([1/2, 1, 3/2, 5/2])
        nspins = spins.shape[0]
        max_spin_order = nspins
        basis = make_basis(spins, max_spin_order)

        # Obtain the gyromagnetic ratios
        y_1H = 2*np.pi * ISOTOPES['1H'][1] * 1e6
        y_14N = 2*np.pi * ISOTOPES['14N'][1] * 1e6
        y_23Na = 2*np.pi * ISOTOPES['23Na'][1] * 1e6
        y_17O = 2*np.pi * ISOTOPES['17O'][1] * 1e6
        gammas = np.array([y_1H, y_14N, y_23Na, y_17O])

        # Conditions
        B = 14.1
        T = 273
        
        # Construct the left Hamiltonian superoperator (only Zeeman)
        H_left = sop_H_Z(basis, gammas, spins, B, side="left", sparse=True)

        # Make the thermal equilibrium state
        rho_eq_sparse = equilibrium_state(basis, spins, H_left, T, sparse=True,
                                          zero_value=1e-18)
        rho_eq_dense = equilibrium_state(basis, spins, H_left, T, sparse=False,
                                         zero_value=1e-18)

        # Test the thermal equilibrium for each spin
        for i in range(nspins):

            # Measure the z-magnetization
            Mz_measured_sparse = measure(
                basis, spins, rho_eq_sparse, f"I(z, {i})")
            Mz_measured_dense = measure(
                basis, spins, rho_eq_dense, f"I(z, {i})")

            # Calculate the thermal magnetization directly using the Boltzmann
            # distribution
            Mz_calculated = thermal_magnetization(gammas[i], spins[i], B, T)

            # Compare
            self.assertAlmostEqual(Mz_measured_sparse, Mz_calculated)
            self.assertAlmostEqual(Mz_measured_dense, Mz_calculated)

        # Test that the code is future-proof (extreme conditions)
        B = 1000
        T = 1

        # Construct the left Hamiltonian superoperator (only Zeeman)
        H_left = sop_H_Z(basis, gammas, spins, B, side="left", sparse=True)

        # Make the thermal equilibrium state
        rho_eq_sparse = equilibrium_state(basis, spins, H_left, T, sparse=True,
                                          zero_value=1e-18)
        rho_eq_dense = equilibrium_state(basis, spins, H_left, T, sparse=False,
                                         zero_value=1e-18)

        # Test the thermal equilibrium for each spin
        for i in range(nspins):

            # Measure the magnetization
            Mz_measured_sparse = measure(
                basis, spins, rho_eq_sparse, f"I(z, {i})")
            Mz_measured_dense = measure(
                basis, spins, rho_eq_dense, f"I(z, {i})")

            # Calculate the thermal magnetization directly using the Boltzmann
            # distribution
            Mz_calculated = thermal_magnetization(gammas[i], spins[i], B, T)

            # Compare
            self.assertAlmostEqual(Mz_measured_sparse, Mz_calculated)
            self.assertAlmostEqual(Mz_measured_dense, Mz_calculated)

def thermal_magnetization(gamma: float, S: float, B: float, T: float) -> float:
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