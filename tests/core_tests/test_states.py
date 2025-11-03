import unittest
import numpy as np
import scipy.constants as const
import spinguin as sg

class TestStates(unittest.TestCase):

    def test_alpha_state(self):
        """
        Test creating the alpha state vector in the spherical tensor basis by
        converting that to density matrix and comparing to a reference result.
        """
        # Create an example spin system
        spin_system = sg.SpinSystem(["1H", "1H"])
        spin_system.basis.max_spin_order = 2
        spin_system.basis.build()

        # Create Hilbert-space spin operators
        sg.config.sparse_operator = False
        E = sg.op_E(1/2)
        Iz = sg.op_Sz(1/2)

        # Create the reference states
        alpha1_ref = 1/4 * np.kron(E, E) + 1/2 * np.kron(Iz, E)
        alpha2_ref = 1/4 * np.kron(E, E) + 1/2 * np.kron(E, Iz)

        # Create alpha states in the spherical tensor basis (sparse)
        sg.config.sparse_state = True
        alpha1_sparse = sg.alpha_state(spin_system, 0)
        alpha2_sparse = sg.alpha_state(spin_system, 1)
        self.assertTrue(np.allclose(
            sg.state_to_zeeman(spin_system, alpha1_sparse),
            alpha1_ref)
        )
        self.assertTrue(np.allclose(
            sg.state_to_zeeman(spin_system, alpha2_sparse),
            alpha2_ref)
        )

        # Create alpha states in the spherical tensor basis (dense)
        sg.config.sparse_state = False
        alpha1_dense = sg.alpha_state(spin_system, 0)
        alpha2_dense = sg.alpha_state(spin_system, 1)
        self.assertTrue(np.allclose(
            sg.state_to_zeeman(spin_system, alpha1_dense),
            alpha1_ref)
        )
        self.assertTrue(np.allclose(
            sg.state_to_zeeman(spin_system, alpha2_dense),
            alpha2_ref)
        )

    def test_beta_state(self):
        """
        Test creating the beta state vector in the spherical tensor basis by
        converting that to density matrix and comparing to a reference result.
        """
        # Create an example spin system
        spin_system = sg.SpinSystem(["1H", "1H"])
        spin_system.basis.max_spin_order = 2
        spin_system.basis.build()

        # Create Hilbert-space spin operators
        sg.config.sparse_operator = False
        E = sg.op_E(1/2)
        Iz = sg.op_Sz(1/2)

        # Create the reference states
        beta1_ref = 1/4 * np.kron(E, E) - 1/2 * np.kron(Iz, E)
        beta2_ref = 1/4 * np.kron(E, E) - 1/2 * np.kron(E, Iz)

        # Create beta states in the spherical tensor basis (sparse)
        sg.config.sparse_state = True
        beta1_sparse = sg.beta_state(spin_system, 0)
        beta2_sparse = sg.beta_state(spin_system, 1)
        self.assertTrue(np.allclose(
            sg.state_to_zeeman(spin_system, beta1_sparse),
            beta1_ref)
        )
        self.assertTrue(np.allclose(
            sg.state_to_zeeman(spin_system, beta2_sparse),
            beta2_ref)
        )

        # Create beta states in the spherical tensor basis (dense)
        sg.config.sparse_state = False
        beta1_dense = sg.beta_state(spin_system, 0)
        beta2_dense = sg.beta_state(spin_system, 1)
        self.assertTrue(np.allclose(
            sg.state_to_zeeman(spin_system, beta1_dense),
            beta1_ref)
        )
        self.assertTrue(np.allclose(
            sg.state_to_zeeman(spin_system, beta2_dense),
            beta2_ref)
        )

    def test_equilibrium_state(self):
        """
        Compare the expectation value of Iz operator for the equilibrium state
        against the reference value calculated from Boltzmann distribution.
        """
        # Create an example spin system with different spin quantum numbers
        spin_system = sg.SpinSystem(["1H", "14N", "23Na", "17O"])
        spin_system.basis.max_spin_order = 4
        spin_system.basis.build()

        # Set the conditions
        sg.parameters.magnetic_field = 14.1
        sg.parameters.temperature = 273

        # Make the thermal equilibrium state
        sg.config.sparse_state = True
        rho_eq_sparse = sg.equilibrium_state(spin_system)
        sg.config.sparse_state = False
        rho_eq_dense = sg.equilibrium_state(spin_system)

        # Test the thermal equilibrium for each spin
        for i in range(spin_system.nspins):

            # Measure the z-magnetization
            Mz_sparse = sg.measure(spin_system, rho_eq_sparse, f"I(z, {i})")
            Mz_dense = sg.measure(spin_system, rho_eq_dense, f"I(z, {i})")

            # Calculate the thermal magnetization directly using the Boltzmann
            # distribution
            Mz_calculated = _thermal_magnetization(
                gamma = spin_system.gammas[i],
                S = spin_system.spins[i],
                B = sg.parameters.magnetic_field,
                T = sg.parameters.temperature
            )

            # Compare
            self.assertAlmostEqual(Mz_sparse, Mz_calculated)
            self.assertAlmostEqual(Mz_dense, Mz_calculated)

        # Test extreme conditions
        sg.parameters.magnetic_field = 1000
        sg.parameters.temperature = 1

        # Make the thermal equilibrium state
        sg.config.sparse_state = True
        rho_eq_sparse = sg.equilibrium_state(spin_system)
        sg.config.sparse_state = False
        rho_eq_dense = sg.equilibrium_state(spin_system)

        # Test the thermal equilibrium for each spin
        for i in range(spin_system.nspins):

            # Measure the z-magnetization
            Mz_sparse = sg.measure(spin_system, rho_eq_sparse, f"I(z, {i})")
            Mz_dense = sg.measure(spin_system, rho_eq_dense, f"I(z, {i})")

            # Calculate the thermal magnetization directly using the Boltzmann
            # distribution
            Mz_calculated = _thermal_magnetization(
                gamma = spin_system.gammas[i],
                S = spin_system.spins[i],
                B = sg.parameters.magnetic_field,
                T = sg.parameters.temperature
            )

            # Compare
            self.assertAlmostEqual(Mz_sparse, Mz_calculated)
            self.assertAlmostEqual(Mz_dense, Mz_calculated)

    def test_measure(self):
        """
        Different states are created for a spin system in both the Hilbert
        space and the Liouville space (spherical tensor basis), and the
        expectation values for varying operators are compared.
        """
        # Create an example spin system with different spin quantum numbers
        spin_system = sg.SpinSystem(["1H", "14N", "23Na"])
        spin_system.basis.max_spin_order = 3
        spin_system.basis.build()

        # States to test
        test_states = ['E', 'x', 'y', 'z', '+', '-']

        # Get the Zeeman eigenbasis operators
        sg.config.sparse_operator = False
        opers = {}
        for spin in spin_system.spins:
            opers[('E', spin)] = sg.op_E(spin)
            opers[('x', spin)] = sg.op_Sx(spin)
            opers[('y', spin)] = sg.op_Sy(spin)
            opers[('z', spin)] = sg.op_Sz(spin)
            opers[('+', spin)] = sg.op_Sp(spin)
            opers[('-', spin)] = sg.op_Sm(spin)

        # Test measuring using the sparse backend
        sg.config.sparse_state = True
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
                    state = sg.state(spin_system, op_string)

                    # Create the density matrices in the Zeeman eigenbasis
                    op1, op2, op3 = \
                        opers[(i, 1/2)], opers[(j, 1)], opers[(k, 3/2)]
                    state_hilb = np.kron(op1, np.kron(op2, op3))

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
                                op1 = opers[(l, 1/2)]
                                op2 = opers[(m, 1)]
                                op3 = opers[(n, 3/2)]
                                oper_hilb = np.kron(op1, np.kron(op2, op3))
                                result_hilb = (
                                    state_hilb @ oper_hilb.conj().T
                                ).trace()

                                # Measure using the inbuilt function
                                op_string = f"{op_l} * {op_m} * {op_n}"
                                result_liouv = sg.measure(
                                    spin_system, state, op_string
                                )

                                # Compare
                                self.assertAlmostEqual(
                                    result_hilb,
                                    result_liouv
                                )

        # Perform the same test using the dense backend
        sg.config.sparse_state = False
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
                    state = sg.state(spin_system, op_string)

                    # Create the density matrices in the Zeeman eigenbasis
                    op1, op2, op3 = \
                        opers[(i, 1/2)], opers[(j, 1)], opers[(k, 3/2)]
                    state_hilb = np.kron(op1, np.kron(op2, op3))

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
                                op1 = opers[(l, 1/2)]
                                op2 = opers[(m, 1)]
                                op3 = opers[(n, 3/2)]
                                oper_hilb = np.kron(op1, np.kron(op2, op3))
                                result_hilb = (
                                    state_hilb @ oper_hilb.conj().T
                                ).trace()

                                # Measure using the inbuilt function
                                op_string = f"{op_l} * {op_m} * {op_n}"
                                result_liouv = sg.measure(
                                    spin_system, state, op_string
                                )

                                # Compare
                                self.assertAlmostEqual(
                                    result_hilb,
                                    result_liouv
                                )

    def test_singlet_state(self):
        """
        Test creating the singlet state vector in the spherical tensor basis by
        converting that to density matrix and comparing to a reference result.
        """
        # Create an example spin system
        spin_system = sg.SpinSystem(["1H", "1H"])
        spin_system.basis.max_spin_order = 2
        spin_system.basis.build()

        # Create Hilbert-space spin operators
        sg.config.sparse_operator = False
        E = sg.op_E(1/2)
        Iz = sg.op_Sz(1/2)
        Ip = sg.op_Sp(1/2)
        Im = sg.op_Sm(1/2)

        # Create the singlet state in both bases and compare
        singlet_zeeman = 1/4 * np.kron(E, E) - np.kron(Iz, Iz) - \
                         1/2 * (np.kron(Ip, Im) + np.kron(Im, Ip))
        sg.config.sparse_state = True
        singlet_sparse = sg.singlet_state(spin_system, 0, 1)
        sg.config.sparse_state = False
        singlet_dense = sg.singlet_state(spin_system, 0, 1)
        self.assertTrue(np.allclose(
            singlet_zeeman,
            sg.state_to_zeeman(spin_system, singlet_sparse))
        )
        self.assertTrue(np.allclose(
            singlet_zeeman,
            sg.state_to_zeeman(spin_system, singlet_dense))
        )
        
    def test_state(self):
        """
        A test that creates various states for a spin system using the
        operator string. These states are converted to Zeeman eigenbasis
        and compared with reference.
        """
        # Create an example spin system with different spin quantum numbers
        spin_system = sg.SpinSystem(["1H", "14N", "23Na"])
        spin_system.basis.max_spin_order = 3
        spin_system.basis.build()

        # States to test
        test_states = ['E', 'x', 'y', 'z', '+', '-']

        # Get the Zeeman eigenbasis operators
        sg.config.sparse_operator = False
        opers = {}
        for spin in spin_system.spins:
            opers[('E', spin)] = sg.op_E(spin)
            opers[('x', spin)] = sg.op_Sx(spin)
            opers[('y', spin)] = sg.op_Sy(spin)
            opers[('z', spin)] = sg.op_Sz(spin)
            opers[('+', spin)] = sg.op_Sp(spin)
            opers[('-', spin)] = sg.op_Sm(spin)

        # Test all possible state combinations (sparse backend)
        sg.config.sparse_state = True
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
                    state_liouv = sg.state(spin_system, op_string)

                    # Create the density matrix in the Zeeman eigenbasis
                    state_hilb = np.kron(
                        opers[(i, 1/2)],
                        np.kron(opers[(j, 1)], opers[(k, 3/2)])
                    )

                    # Compare
                    self.assertTrue(np.allclose(
                        sg.state_to_zeeman(spin_system, state_liouv),
                        state_hilb
                    ))

        # Test all possible state combinations (dense backend)
        sg.config.sparse_state = False
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
                    state_liouv = sg.state(spin_system, op_string)

                    # Create the density matrix in the Zeeman eigenbasis
                    state_hilb = np.kron(
                        opers[(i, 1/2)],
                        np.kron(opers[(j, 1)], opers[(k, 3/2)])
                    )

                    # Compare
                    self.assertTrue(np.allclose(
                        sg.state_to_zeeman(spin_system, state_liouv),
                        state_hilb
                    ))
        
    def test_triplet_minus(self):
        """
        Test creating the triplet minus state vector in the spherical tensor
        basis by converting that to density matrix and comparing to a reference
        result.
        """
        # Create an example spin system
        spin_system = sg.SpinSystem(["1H", "1H"])
        spin_system.basis.max_spin_order = 2
        spin_system.basis.build()

        # Create Zeeman operators
        sg.config.sparse_operator = False
        E = sg.op_E(1/2)
        Iz = sg.op_Sz(1/2)

        # Create the triplet-minus state and compare
        triplet_minus_zeeman = 1/4 * np.kron(E, E) - 1/2 * np.kron(E, Iz) - \
                               1/2 * np.kron(Iz, E) + np.kron(Iz, Iz)
        sg.config.sparse_state = True
        triplet_minus_sparse = sg.triplet_minus_state(spin_system, 0, 1)
        sg.config.sparse_state = False
        triplet_minus_dense = sg.triplet_minus_state(spin_system, 0, 1)
        self.assertTrue(np.allclose(
            triplet_minus_zeeman,
            sg.state_to_zeeman(spin_system, triplet_minus_sparse)
        ))
        self.assertTrue(np.allclose(
            triplet_minus_zeeman,
            sg.state_to_zeeman(spin_system, triplet_minus_dense)
        ))

    def test_triplet_plus(self):
        """
        Test creating the triplet plus state vector in the spherical tensor
        basis by converting that to density matrix and comparing to a reference
        result.
        """
        # Create an example spin system
        spin_system = sg.SpinSystem(["1H", "1H"])
        spin_system.basis.max_spin_order = 2
        spin_system.basis.build()

        # Create Zeeman operators
        sg.config.sparse_operator = False
        E = sg.op_E(1/2)
        Iz = sg.op_Sz(1/2)

        # Create the triplet-plus state and compare
        triplet_plus_zeeman = 1/4 * np.kron(E, E) + 1/2 * np.kron(E, Iz) + \
                              1/2 * np.kron(Iz, E) + np.kron(Iz, Iz)
        sg.config.sparse_state = True
        triplet_plus_sparse = sg.triplet_plus_state(spin_system, 0, 1)
        sg.config.sparse_state = False
        triplet_plus_dense = sg.triplet_plus_state(spin_system, 0, 1)
        self.assertTrue(np.allclose(
            triplet_plus_zeeman,
            sg.state_to_zeeman(spin_system, triplet_plus_sparse)
        ))
        self.assertTrue(np.allclose(
            triplet_plus_zeeman,
            sg.state_to_zeeman(spin_system, triplet_plus_dense)
        ))
        
    def test_triplet_zero(self):
        """
        Test creating the triplet zero state vector in the spherical tensor
        basis by converting that to density matrix and comparing to a reference
        result.
        """
        # Create an example spin system
        spin_system = sg.SpinSystem(["1H", "1H"])
        spin_system.basis.max_spin_order = 2
        spin_system.basis.build()

        # Create Zeeman operators
        sg.config.sparse_operator = False
        E = sg.op_E(1/2)
        Iz = sg.op_Sz(1/2)
        Ip = sg.op_Sp(1/2)
        Im = sg.op_Sm(1/2)

        # Create the triplet-zero state and compare
        triplet_zero_zeeman = 1/4 * np.kron(E, E) - np.kron(Iz, Iz) + \
                              1/2 * (np.kron(Ip, Im) + np.kron(Im, Ip))
        sg.config.sparse_state = True
        triplet_zero_sparse = sg.triplet_zero_state(spin_system, 0, 1)
        sg.config.sparse_state = False
        triplet_zero_dense = sg.triplet_zero_state(spin_system, 0, 1)
        self.assertTrue(np.allclose(
            triplet_zero_zeeman,
            sg.state_to_zeeman(spin_system, triplet_zero_sparse)
        ))
        self.assertTrue(np.allclose(
            triplet_zero_zeeman,
            sg.state_to_zeeman(spin_system, triplet_zero_dense)
        ))

    def test_unit_state(self):
        """
        A test that creates the unit state in the spherical tensor basis
        with the two normalization conventions and compares them to the
        expected reference values.
        """

        # Create an example spin system with different spin quantum numbers
        spin_system = sg.SpinSystem(["1H", "14N", "23Na"])
        spin_system.basis.max_spin_order = 3
        spin_system.basis.build()

        # Create the unit state in the Zeeman eigenbasis
        sg.config.sparse_operator = False
        unit_zeeman = np.array([[1]])
        for spin in spin_system.spins:
            unit_zeeman = np.kron(unit_zeeman, sg.op_E(spin))

        # Create the non-normalized unit state in the spherical tensor basis
        sg.config.sparse_state = True
        unit_sparse = sg.unit_state(spin_system, normalized=False)
        sg.config.sparse_state = False
        unit_dense = sg.unit_state(spin_system, normalized=False)

        # Compare
        self.assertTrue(np.allclose(
            sg.state_to_zeeman(spin_system, unit_sparse),
            unit_zeeman
        ))
        self.assertTrue(np.allclose(
            sg.state_to_zeeman(spin_system, unit_dense),
            unit_zeeman
        ))

        # Create the trace-normalized unit state in the spherical tensor basis
        sg.config.sparse_state = True
        unit_sparse = sg.unit_state(spin_system, normalized=True)
        sg.config.sparse_state = False
        unit_dense = sg.unit_state(spin_system, normalized=True)

        # Apply trace normalization to the unit state in the Zeeman eigenbasis
        unit_zeeman = unit_zeeman / unit_zeeman.trace()

        # Compare
        self.assertTrue(np.allclose(
            sg.state_to_zeeman(spin_system, unit_sparse),
            unit_zeeman
        ))
        self.assertTrue(np.allclose(
            sg.state_to_zeeman(spin_system, unit_dense),
            unit_zeeman
        ))

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