import unittest
import numpy as np
import os
from spinguin.system.spin_system import SpinSystem

class TestSpinSystem(unittest.TestCase):

    def test_assign_isotopes(self):
        """
        A test for creating a SpinSystem instance and assigning isotopes using
        different input types.
        """

        # Initialize our SpinSystem object
        spin_system = SpinSystem()

        # Assigning isotopes should work with any array like object
        isotopes = np.array(['1H', '19F', '14N'])
        spin_system.isotopes = isotopes
        self.assertTrue((spin_system.isotopes == isotopes).all())

        spin_system.isotopes = list(isotopes)
        self.assertTrue((spin_system.isotopes == isotopes).all())

        spin_system.isotopes = tuple(isotopes)
        self.assertTrue((spin_system.isotopes == isotopes).all())

        # Assigning isotopes should work from text file
        spin_system.isotopes = os.path.join(os.path.dirname(__file__),
                                            'test_data',
                                            'isotopes.txt')
        self.assertTrue((spin_system.isotopes == isotopes).all())

        # Assigning isotopes with incorrect file path should result in error
        with self.assertRaises(FileNotFoundError):
            spin_system.isotopes = "not_a_real_file.txt"

        # Assigning an array with incorrect dimensions should result in error
        isotopes = np.array([['1H', '19F', '14N']])
        with self.assertRaises(ValueError):
            spin_system.isotopes = isotopes

        # Assigning an isotope that is not defined should result in error
        isotopes = np.array(['4H'])
        with self.assertRaises(ValueError):
            spin_system.isotopes = isotopes

    def test_assign_chemical_shifts(self):
        """
        A test for creating a SpinSystem instance and assigning chemical shifts
        using different input types.
        """

        # Initialize our SpinSystem object
        spin_system = SpinSystem()

        # Trying to configure chemical shifts before isotopes results in error
        chemical_shifts = np.array([8.00, -127.5, 40.50])
        with self.assertRaises(ValueError):
            spin_system.chemical_shifts = chemical_shifts

        # Configure isotopes before configuring chemical shifts
        spin_system.isotopes = np.array(['1H', '19F', '14N'])

        # Assigning chemical shifts should work with any array like object
        spin_system.chemical_shifts = chemical_shifts
        self.assertTrue((spin_system.chemical_shifts == chemical_shifts).all())

        spin_system.chemical_shifts = list(chemical_shifts)
        self.assertTrue((spin_system.chemical_shifts == chemical_shifts).all())

        spin_system.chemical_shifts = tuple(chemical_shifts)
        self.assertTrue((spin_system.chemical_shifts == chemical_shifts).all())

        # Assigning chemical_shifts should work from text file
        spin_system.chemical_shifts = os.path.join(os.path.dirname(__file__),
                                                   'test_data',
                                                   'chemical_shifts.txt')
        self.assertTrue((spin_system.chemical_shifts == chemical_shifts).all())

        # Assigning chemical_shifts with incorrect file path should result in error
        with self.assertRaises(FileNotFoundError):
            spin_system.chemical_shifts = "not_a_real_file.txt"

        # Assigning an array with incorrect dimensions should result in error
        chemical_shifts = np.array([[8.00, -127.5, 40.50]])
        with self.assertRaises(ValueError):
            spin_system.chemical_shifts = chemical_shifts

    def test_assign_J_couplings(self):
        """
        A test for creating a SpinSystem instance and assigning J-couplings
        using different input types.
        """

        # Initialize our SpinSystem object
        spin_system = SpinSystem()

        # Trying to configure J-couplings before isotopes results in error
        J_couplings = np.array([
            [0,    0,    0],
            [1.05, 0,    0],
            [0.50, 9.17, 0]
        ])
        with self.assertRaises(ValueError):
            spin_system.J_couplings = J_couplings

        # Configure isotopes before configuring J-couplings
        spin_system.isotopes = np.array(['1H', '19F', '14N'])

        # Assigning J-couplings should work with any array like object
        spin_system.J_couplings = J_couplings
        self.assertTrue((spin_system.J_couplings == J_couplings).all())

        spin_system.J_couplings = list(J_couplings)
        self.assertTrue((spin_system.J_couplings == J_couplings).all())

        spin_system.J_couplings = tuple(J_couplings)
        self.assertTrue((spin_system.J_couplings == J_couplings).all())

        # Assigning J-couplings should work from text file
        spin_system.J_couplings = os.path.join(os.path.dirname(__file__),
                                               'test_data',
                                               'J_couplings.txt')
        self.assertTrue((spin_system.J_couplings == J_couplings).all())

        # Assigning J-couplings with incorrect file path should result in error
        with self.assertRaises(FileNotFoundError):
            spin_system.J_couplings = "not_a_real_file.txt"

        # Assigning an array with incorrect dimensions should result in error
        J_couplings = np.array([[
            [0,    0,    0],
            [1.05, 0,    0],
            [0.50, 9.17, 0]
        ]])
        with self.assertRaises(ValueError):
            spin_system.J_couplings = J_couplings

    def test_assign_xyz(self):
        """
        A test for creating a SpinSystem instance and assigning XYZ coordinates
        using different input types.
        """

        # Initialize our SpinSystem object
        spin_system = SpinSystem()

        # Trying to configure XYZ before isotopes results in error
        xyz = np.array([
            [1.0527, 2.2566, 0.9925],
            [0.0014, 1.5578, 2.1146],
            [1.3456, 0.3678, 1.4251]
        ])
        with self.assertRaises(ValueError):
            spin_system.xyz = xyz

        # Configure isotopes before configuring XYZ
        spin_system.isotopes = np.array(['1H', '19F', '14N'])

        # Assigning XYZ should work with any array like object
        spin_system.xyz = xyz
        self.assertTrue((spin_system.xyz == xyz).all())

        spin_system.xyz = list(xyz)
        self.assertTrue((spin_system.xyz == xyz).all())

        spin_system.xyz = tuple(xyz)
        self.assertTrue((spin_system.xyz == xyz).all())

        # Assigning XYZ should work from text file
        spin_system.xyz = os.path.join(os.path.dirname(__file__),
                                       'test_data',
                                       'xyz.txt')
        self.assertTrue((spin_system.xyz == xyz).all())

        # Assigning XYZ with incorrect file path should result in error
        with self.assertRaises(FileNotFoundError):
            spin_system.xyz = "not_a_real_file.txt"

        # Assigning an array with incorrect dimensions should result in error
        xyz = np.array([[
            [1.0527, 2.2566, 0.9925],
            [0.0014, 1.5578, 2.1146],
            [1.3456, 0.3678, 1.4251]
        ]])
        with self.assertRaises(ValueError):
            spin_system.xyz = xyz

    def test_assign_shielding(self):
        """
        A test for creating a SpinSystem instance and assigning shielding
        tensors using different input types.
        """

        # Initialize our SpinSystem object
        spin_system = SpinSystem()

        # Trying to configure shielding tensors before isotopes results in error
        shielding = np.array([
            [[0, 0, 0],
             [0, 0, 0],
             [0, 0, 0]],
            [[101.6, -75.2, 11.1],
             [30.5,   10.1, 87.4],
             [99.7,  -21.1, 11.2]],
            [[171.9, -58.6, 91.1],
             [37.5,   10.7, 86.9],
             [109.7, -91.1, 81.8]]
        ])
        with self.assertRaises(ValueError):
            spin_system.shielding = shielding

        # Configure isotopes before configuring shielding tensors
        spin_system.isotopes = np.array(['1H', '19F', '14N'])

        # Assigning shielding tensors should work with any array like object
        spin_system.shielding = shielding
        self.assertTrue((spin_system.shielding == shielding).all())

        spin_system.shielding = list(shielding)
        self.assertTrue((spin_system.shielding == shielding).all())

        spin_system.shielding = tuple(shielding)
        self.assertTrue((spin_system.shielding == shielding).all())

        # Assigning shielding tensors should work from text file
        spin_system.shielding = os.path.join(os.path.dirname(__file__),
                                             'test_data',
                                             'shielding.txt')
        self.assertTrue((spin_system.shielding == shielding).all())

        # Assigning shielding tensors with incorrect file path should result in error
        with self.assertRaises(FileNotFoundError):
            spin_system.shielding = "not_a_real_file.txt"

        # Assigning an array with incorrect dimensions should result in error
        shielding = np.array([[
            [[0, 0, 0],
             [0, 0, 0],
             [0, 0, 0]],
            [[101.6, -75.2, 11.1],
             [30.5,   10.1, 87.4],
             [99.7,  -21.1, 11.2]],
            [[171.9, -58.6, 91.1],
             [37.5,   10.7, 86.9],
             [109.7, -91.1, 81.8]]
        ]])
        with self.assertRaises(ValueError):
            spin_system.shielding = shielding

    def test_assign_efg(self):
        """
        A test for creating a SpinSystem instance and assigning efg tensors
        using different input types.
        """

        # Initialize our SpinSystem object
        spin_system = SpinSystem()

        # Trying to configure efg tensors before isotopes results in error
        efg = np.array([
            [[0, 0, 0],
             [0, 0, 0],
             [0, 0, 0]],
            [[ 0.31, 0.00, 0.01],
             [-0.20, 0.04, 0.87],
             [ 0.11, 0.16, 0.65]],
            [[0.34, 0.67, 0.23],
             [0.38, 0.65, 0.26],
             [0.29, 0.82, 0.06]]
        ])
        with self.assertRaises(ValueError):
            spin_system.efg = efg

        # Configure isotopes before configuring efg tensors
        spin_system.isotopes = np.array(['1H', '19F', '14N'])

        # Assigning efg tensors should work with any array like object
        spin_system.efg = efg
        self.assertTrue((spin_system.efg == efg).all())

        spin_system.efg = list(efg)
        self.assertTrue((spin_system.efg == efg).all())

        spin_system.efg = tuple(efg)
        self.assertTrue((spin_system.efg == efg).all())

        # Assigning efg tensors should work from text file
        spin_system.efg = os.path.join(os.path.dirname(__file__),
                                       'test_data',
                                       'efg.txt')
        self.assertTrue((spin_system.efg == efg).all())

        # Assigning efg tensors with incorrect file path should result in error
        with self.assertRaises(FileNotFoundError):
            spin_system.efg = "not_a_real_file.txt"

        # Assigning an array with incorrect dimensions should result in error
        efg = np.array([[
            [[0, 0, 0],
             [0, 0, 0],
             [0, 0, 0]],
            [[ 0.31, 0.00, 0.01],
             [-0.20, 0.04, 0.87],
             [ 0.11, 0.16, 0.65]],
            [[0.34, 0.67, 0.23],
             [0.38, 0.65, 0.26],
             [0.29, 0.82, 0.06]]
        ]])
        with self.assertRaises(ValueError):
            spin_system.efg = efg

    def test_set_maximum_spin_order(self):
        """
        A test for creating a SpinSystem instance and setting the maximum spin
        order.
        """

        # Initialize our SpinSystem object
        spin_system = SpinSystem()

        # Trying to set the maximum spin order before isotopes results in error
        with self.assertRaises(ValueError):
            spin_system.max_spin_order = 3

        # Configure isotopes before setting the maximum spin order
        spin_system.isotopes = np.array(['1H', '19F', '14N'])

        # Maximum spin orders within 1 and the number of spins should be fine
        for max_spin_order in range(1, spin_system.nspins+1):
            spin_system.max_spin_order = max_spin_order

        # Setting the maximum spin order to less than 1 causes an error
        with self.assertRaises(ValueError):
            spin_system.max_spin_order = 0

        # Setting the maximum spin order above number of spins causes an error
        with self.assertRaises(ValueError):
            spin_system.max_spin_order = spin_system.nspins + 1

    def test_build_basis(self):
        """
        A test for creating a SpinSystem instance and building a basis set.
        """

        # Initialize our SpinSystem object
        spin_system = SpinSystem()

        # Trying to build a basis before setting the isotopes results in an error
        with self.assertRaises(ValueError):
            spin_system.build_basis()

        # Configure isotopes before building the basis
        spin_system.isotopes = np.array(['1H', '19F', '14N'])

        # Building a basis without specifying the maximum spin order gives a warning
        # and sets the maximum spin order to the number of spins
        with self.assertWarns(Warning):
            spin_system.build_basis()
        self.assertEqual(spin_system.max_spin_order, spin_system.nspins)
        self.assertTrue(isinstance(spin_system.basis, np.ndarray))

    def test_set_magnetic_field(self):
        """
        A test for creating a SpinSystem instance and setting the magnetic
        field.
        """

        # Initialize our SpinSystem object
        spin_system = SpinSystem()

        # Set the magnetic field
        magnetic_field = 1
        spin_system.magnetic_field = magnetic_field
        self.assertEqual(spin_system.magnetic_field, magnetic_field)

    def test_set_temperature(self):
        """
        A test for creating a SpinSystem instance and setting the temperature.
        """

        # Initialize our SpinSystem object
        spin_system = SpinSystem()

        # Set the temperature
        temperature = 293
        spin_system.temperature = temperature
        self.assertEqual(spin_system.temperature, temperature)

    def test_set_relaxation_theory(self):
        """
        A test for creating a SpinSystem instance and setting the relaxation
        theory.
        """

        # Initialize our SpinSystem object
        spin_system = SpinSystem()

        # Set the relaxation theory
        relaxation_theory = "redfield"
        spin_system.relaxation_theory = relaxation_theory
        self.assertEqual(spin_system.relaxation_theory, relaxation_theory)

    def test_set_thermalization(self):
        """
        A test for adjusting the thermalization setting.
        """

        # Initialize our SpinSystem object
        spin_system = SpinSystem()

        # Set the thermalization setting
        thermalization = True
        spin_system.thermalization = thermalization
        self.assertEqual(spin_system.thermalization, thermalization)

    def test_set_tau_c(self):
        """
        A test for creating a SpinSystem instance and setting the correlation
        time.
        """

        # Initialize our SpinSystem object
        spin_system = SpinSystem()

        # Set the correlation time
        tau_c = 50e-12
        spin_system.tau_c = tau_c
        self.assertEqual(spin_system.tau_c, tau_c)

    def test_set_sr2k(self):
        """
        A test for creating a SpinSystem instance and setting the scalar
        relaxation of the second kind.
        """

        # Initialize our SpinSystem object
        spin_system = SpinSystem()

        # Set the sr2k
        sr2k = True
        spin_system.sr2k = sr2k
        self.assertEqual(spin_system.sr2k, sr2k)

    def test_set_dynamic_frequency_shift(self):
        """
        A test for creating a SpinSystem instance and setting the dynamic
        frequency shift.
        """

        # Initialize our SpinSystem object
        spin_system = SpinSystem()

        # Set the dynamic frequency shift
        dynamic_frequency_shift = True
        spin_system.dynamic_frequency_shift = dynamic_frequency_shift
        self.assertEqual(spin_system.dynamic_frequency_shift, dynamic_frequency_shift)

    def test_set_antisymmetric_relaxation(self):
        """
        A test for creating a SpinSystem instance and setting the antisymmetric
        relaxation.
        """

        # Initialize our SpinSystem object
        spin_system = SpinSystem()

        # Set the antisymmetric relaxation
        antisymmetric_relaxation = True
        spin_system.antisymmetric_relaxation = antisymmetric_relaxation
        self.assertEqual(spin_system.antisymmetric_relaxation, antisymmetric_relaxation)

    def test_set_relative_error(self):
        """
        A test for creating a SpinSystem instance and setting the relative error.
        """

        # Initialize our SpinSystem object
        spin_system = SpinSystem()

        # Set the relative error
        relative_error = 1e-12
        spin_system.relative_error = relative_error
        self.assertEqual(spin_system.relative_error, relative_error)

    def test_set_T1(self):
        """
        A test for setting the longitudinal relaxation time constants.
        """
        
        # Initialize our SpinSystem object
        spin_system = SpinSystem()

        # Trying to configure T1 times before isotopes raises an error
        T1 = [1, 2, 3]
        with self.assertRaises(ValueError):
            spin_system.T1 = T1

        # After configuring isotopes, should work
        spin_system.isotopes = ['1H', '1H', '1H']
        spin_system.T1 = T1
        self.assertTrue((spin_system.T1 == T1).all())

        # Assigning T1 should work from text file
        spin_system.T1 = os.path.join(os.path.dirname(__file__),
                                            'test_data',
                                            'T1.txt')
        self.assertTrue((spin_system.T1 == T1).all())

    def test_set_T2(self):
        """
        A test for setting the transverse relaxation time constants.
        """
        
        # Initialize our SpinSystem object
        spin_system = SpinSystem()

        # Trying to configure T2 times before isotopes raises an error
        T2 = [1, 2, 3]
        with self.assertRaises(ValueError):
            spin_system.T2 = T2

        # After configuring isotopes, should work
        spin_system.isotopes = ['1H', '1H', '1H']
        spin_system.T2 = T2
        self.assertTrue((spin_system.T2 == T2).all())

        # Assigning T2 should work from text file
        spin_system.T2 = os.path.join(os.path.dirname(__file__),
                                            'test_data',
                                            'T2.txt')
        self.assertTrue((spin_system.T2 == T2).all())

    def test_set_sparse_operator(self):
        """
        A test for creating a SpinSystem instance and setting the sparsity
        setting for operator.
        """

        # Initialize our SpinSystem object
        spin_system = SpinSystem()

        # Set the sparsity for operator
        sparse_operator = False
        spin_system.sparse_operator = sparse_operator
        self.assertEqual(spin_system.sparse_operator, sparse_operator)

    def test_set_sparse_superoperator(self):
        """
        A test for creating a SpinSystem instance and setting the sparsity
        setting for superoperator.
        """

        # Initialize our SpinSystem object
        spin_system = SpinSystem()

        # Set the sparsity for superoperator
        sparse_superoperator = False
        spin_system.sparse_superoperator = sparse_superoperator
        self.assertEqual(spin_system.sparse_superoperator, sparse_superoperator)

    def test_set_sparse_hamiltonian(self):
        """
        A test for creating a SpinSystem instance and setting the sparsity
        setting for Hamiltonian.
        """

        # Initialize our SpinSystem object
        spin_system = SpinSystem()

        # Set the sparsity for Hamiltonian
        sparse_hamiltonian = False
        spin_system.sparse_hamiltonian = sparse_hamiltonian
        self.assertEqual(spin_system.sparse_hamiltonian, sparse_hamiltonian)

    def test_set_sparse_relaxation(self):
        """
        A test for creating a SpinSystem instance and setting the sparsity
        setting for relaxation superoperator.
        """

        # Initialize our SpinSystem object
        spin_system = SpinSystem()

        # Set the sparsity for relaxation superoperator
        sparse_relaxation = False
        spin_system.sparse_relaxation = sparse_relaxation
        self.assertEqual(spin_system.sparse_relaxation, sparse_relaxation)

    def test_set_propagator_density(self):
        """
        A test for creating a SpinSystem instance and setting the threshold
        for propagator density.
        """

        # Initialize our SpinSystem object
        spin_system = SpinSystem()

        # Set the density threshold for propagator
        propagator_density = 1.0
        spin_system.propagator_density = propagator_density
        self.assertEqual(spin_system.propagator_density, propagator_density)

    def test_set_sparse_state(self):
        """
        A test for creating a SpinSystem instance and setting the sparsity
        setting for state vectors.
        """

        # Initialize our SpinSystem object
        spin_system = SpinSystem()

        # Set the sparsity for state vectors
        sparse_state = True
        spin_system.sparse_state = sparse_state
        self.assertEqual(spin_system.sparse_state, sparse_state)

    def test_set_zero_hamiltonian(self):
        """
        A test for creating a SpinSystem instance and setting the zero-value
        threshold for Hamiltonian.
        """

        # Initialize our SpinSystem object
        spin_system = SpinSystem()

        # Set the zero-value threshold for Hamiltonian
        zero_hamiltonian = 1e-6
        spin_system.zero_hamiltonian = zero_hamiltonian
        self.assertEqual(spin_system.zero_hamiltonian, zero_hamiltonian)

    def test_set_zero_aux(self):
        """
        A test for creating a SpinSystem instance and setting the zero-value
        threshold for auxiliary matrix method.
        """

        # Initialize our SpinSystem object
        spin_system = SpinSystem()

        # Set the zero-value threshold for auxiliary matrix method
        zero_aux = 1e-6
        spin_system.zero_aux = zero_aux
        self.assertEqual(spin_system.zero_aux, zero_aux)

    def test_set_zero_relaxation(self):
        """
        A test for creating a SpinSystem instance and setting the zero-value
        threshold for relaxation superoperator.
        """

        # Initialize our SpinSystem object
        spin_system = SpinSystem()

        # Set the zero-value threshold for relaxation superoperator
        zero_relaxation = 1e-6
        spin_system.zero_relaxation = zero_relaxation
        self.assertEqual(spin_system.zero_relaxation, zero_relaxation)

    def test_set_zero_interaction(self):
        """
        A test for creating a SpinSystem instance and setting the zero-value
        threshold for interaction tensors.
        """

        # Initialize our SpinSystem object
        spin_system = SpinSystem()

        # Set the zero-value threshold for interaction tensors
        zero_interaction = 1e-6
        spin_system.zero_interaction = zero_interaction
        self.assertEqual(spin_system.zero_interaction, zero_interaction)

    def test_set_zero_propagator(self):
        """
        A test for creating a SpinSystem instance and setting the zero-value
        threshold for propagator.
        """

        # Initialize our SpinSystem object
        spin_system = SpinSystem()

        # Set the zero-value threshold for propagator
        zero_propagator = 1e-6
        spin_system.zero_propagator = zero_propagator
        self.assertEqual(spin_system.zero_propagator, zero_propagator)

    def test_set_zero_pulse(self):
        """
        A test for creating a SpinSystem instance and setting the zero-value
        threshold for pulse.
        """

        # Initialize our SpinSystem object
        spin_system = SpinSystem()

        # Set the zero-value threshold for pulse
        zero_pulse = 1e-6
        spin_system.zero_pulse = zero_pulse
        self.assertEqual(spin_system.zero_pulse, zero_pulse)

    def test_set_zero_thermalization(self):
        """
        A test for creating a SpinSystem instance and setting the zero-value
        threshold for thermalization.
        """

        # Initialize our SpinSystem object
        spin_system = SpinSystem()

        # Set the zero-value threshold for thermalization
        zero_thermalization = 1e-6
        spin_system.zero_thermalization = zero_thermalization
        self.assertEqual(spin_system.zero_thermalization, zero_thermalization)

    def test_set_zero_equilibrium(self):
        """
        A test for creating a SpinSystem instance and setting the zero-value
        threshold for equilibrium state.
        """

        # Initialize our SpinSystem object
        spin_system = SpinSystem()

        # Set the zero-value threshold for equilibrium state
        zero_equilibrium = 1e-6
        spin_system.zero_equilibrium = zero_equilibrium
        self.assertEqual(spin_system.zero_equilibrium, zero_equilibrium)

    def test_create_operator(self):
        """
        A test for creating Hilbert-space operators for the SpinSystem instance.
        """

        # Initialize our SpinSystem object
        spin_system = SpinSystem()

        # Operator to make
        operator = "I(x,0) * I(y,1)"

        # Trying to make an operator before defining the isotopes causes error
        with self.assertRaises(ValueError):
            spin_system.operator(operator)
        
        # Creating an operator is successful after setting isotopes
        spin_system.isotopes = ['1H', '1H']
        spin_system.operator(operator)

    def test_create_superoperator(self):
        """
        A test for creating Liouville-space superoperators for the SpinSystem
        instance.
        """

        # Initialize our SpinSystem object
        spin_system = SpinSystem()

        # Superoperator to make
        operator = "I(x,0) * I(y,1)"

        # Trying to make a superoperator before defining the isotopes causes
        # error
        with self.assertRaises(ValueError):
            spin_system.superoperator(operator)
        
        # Trying to make a superoperator before defining basis causes error
        spin_system.isotopes = ['1H', '1H']
        with self.assertRaises(ValueError):
            spin_system.superoperator(operator)

        # When basis is built, constructing superoperator should work
        spin_system.max_spin_order = spin_system.nspins
        spin_system.build_basis()
        spin_system.superoperator(operator)

    def test_create_hamiltonian(self):
        """
        A test for creating Hamiltonians for the SpinSystem instance.
        """

        # Initialize our SpinSystem object
        spin_system = SpinSystem()

        # Trying to make Hamiltonian before setting isotopes causes error
        with self.assertRaises(ValueError):
            spin_system.hamiltonian(interactions="all")
        spin_system.isotopes = ["1H", "1H", "1H"]

        # Trying to make Hamiltonian before building basis causes error
        with self.assertRaises(ValueError):
            spin_system.hamiltonian(interactions="all")
        spin_system.max_spin_order = spin_system.nspins
        spin_system.build_basis()

        # Trying to make Zeeman, chemical shift, or "all" Hamiltonian before
        # setting field causes error
        with self.assertRaises(ValueError):
            spin_system.hamiltonian(interactions="zeeman")
        with self.assertRaises(ValueError):
            spin_system.hamiltonian(interactions="chemical_shift")
        with self.assertRaises(ValueError):
            spin_system.hamiltonian(interactions="all")
        spin_system.magnetic_field = 1

        # Trying to make chemical shift or "all" Hamiltonian before setting
        # chemical shifts causes error
        with self.assertRaises(ValueError):
            spin_system.hamiltonian(interactions="chemical_shift")
        with self.assertRaises(ValueError):
            spin_system.hamiltonian(interactions="all")
        spin_system.chemical_shifts = [6.00, 7.00, 8.00]

        # Trying to make J-coupling or "all" Hamiltonian before settings
        # J-couplings causes error
        with self.assertRaises(ValueError):
            spin_system.hamiltonian(interactions="J_coupling")
        with self.assertRaises(ValueError):
            spin_system.hamiltonian(interactions="all")
        spin_system.J_couplings = [
            [0, 0, 0],
            [1, 0, 0],
            [2, 1, 0]
        ]

        # After all parameters are set, no errors should be raised
        spin_system.hamiltonian(interactions="all")
        spin_system.hamiltonian(interactions="zeeman")
        spin_system.hamiltonian(interactions="chemical_shift")
        spin_system.hamiltonian(interactions="J_coupling")

