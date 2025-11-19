import unittest
import numpy as np
import os
import spinguin as sg

class TestSpinSystem(unittest.TestCase):

    def test_assign_isotopes(self):
        """
        A test for creating a SpinSystem instance and assigning isotopes using
        different input types.
        """
        # Isotopes
        isotopes = np.array(['1H', '19F', '14N'])

        # Initialising spin system should work with any array like object
        spin_system = sg.SpinSystem(isotopes)
        self.assertTrue((spin_system.isotopes == isotopes).all())
        spin_system = sg.SpinSystem(list(isotopes))
        self.assertTrue((spin_system.isotopes == isotopes).all())
        spin_system = sg.SpinSystem(tuple(isotopes))
        self.assertTrue((spin_system.isotopes == isotopes).all())

        # Initialising spin system should work from text file
        isotopes_txt = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'test_data',
            'isotopes.txt')
        spin_system = sg.SpinSystem(isotopes_txt)
        self.assertTrue((spin_system.isotopes == isotopes).all())

        # Initialising spin system with incorrect file path results in error
        isotopes_txt = "not_a_real_file.txt"
        with self.assertRaises(FileNotFoundError):
            spin_system = sg.SpinSystem(isotopes_txt)

        # Assigning an array with incorrect dimensions should result in error
        isotopes = np.array([['1H', '19F', '14N']])
        with self.assertRaises(ValueError):
            spin_system = sg.SpinSystem(isotopes)

        # Assigning an isotope that is not defined should result in error
        isotopes = np.array(['4H'])
        with self.assertRaises(ValueError):
            spin_system = sg.SpinSystem(isotopes)

    def test_assign_chemical_shifts(self):
        """
        A test for creating a SpinSystem instance and assigning chemical shifts
        using different input types.
        """

        # Initialize our SpinSystem object
        spin_system = sg.SpinSystem(["1H", "19F", "14N"])

        # Trying to configure chemical shifts for a spin system of incorrect
        # size results in an error
        chemical_shifts = np.array([8.00, -127.5])
        with self.assertRaises(ValueError):
            spin_system.chemical_shifts = chemical_shifts

        # Assigning chemical shifts should work with any array like object
        chemical_shifts = np.array([8.00, -127.5, 40.50])
        spin_system.chemical_shifts = chemical_shifts
        self.assertTrue((spin_system.chemical_shifts == chemical_shifts).all())

        spin_system.chemical_shifts = list(chemical_shifts)
        self.assertTrue((spin_system.chemical_shifts == chemical_shifts).all())

        spin_system.chemical_shifts = tuple(chemical_shifts)
        self.assertTrue((spin_system.chemical_shifts == chemical_shifts).all())

        # Assigning chemical_shifts should work from text file
        spin_system.chemical_shifts = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'test_data',
            'chemical_shifts.txt')
        self.assertTrue((spin_system.chemical_shifts == chemical_shifts).all())

        # Assigning chemical_shifts with incorrect file path should result in
        # error
        with self.assertRaises(FileNotFoundError):
            spin_system.chemical_shifts = "not_a_real_file.txt"

    def test_assign_J_couplings(self):
        """
        A test for creating a SpinSystem instance and assigning J-couplings
        using different input types.
        """

        # Initialize our SpinSystem object
        spin_system = sg.SpinSystem(["1H", "19F", "14N"])

        # Trying to configure J-couplings for a spin system of incorrect size
        # should result in an error
        J_couplings = np.array([
            [0,    0],
            [1.05, 0]
        ])
        with self.assertRaises(ValueError):
            spin_system.J_couplings = J_couplings


        # Assigning J-couplings should work with any array like object
        J_couplings = np.array([
            [0,    0,    0],
            [1.05, 0,    0],
            [0.50, 9.17, 0]
        ])
        spin_system.J_couplings = J_couplings
        self.assertTrue((spin_system.J_couplings == J_couplings).all())

        spin_system.J_couplings = list(J_couplings)
        self.assertTrue((spin_system.J_couplings == J_couplings).all())

        spin_system.J_couplings = tuple(J_couplings)
        self.assertTrue((spin_system.J_couplings == J_couplings).all())

        # Assigning J-couplings should work from text file
        spin_system.J_couplings = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'test_data',
            'J_couplings.txt')
        self.assertTrue((spin_system.J_couplings == J_couplings).all())

        # Assigning J-couplings with incorrect file path should result in error
        with self.assertRaises(FileNotFoundError):
            spin_system.J_couplings = "not_a_real_file.txt"

    def test_assign_xyz(self):
        """
        A test for creating a SpinSystem instance and assigning XYZ coordinates
        using different input types.
        """

        # Initialize our SpinSystem object
        spin_system = sg.SpinSystem(["1H", "19F", "14N"])

        # Trying to configure XYZ for a spin system of incorrect size should
        # result in an error
        xyz = np.array([
            [1.0527, 2.2566, 0.9925],
            [0.0014, 1.5578, 2.1146]
        ])
        with self.assertRaises(ValueError):
            spin_system.xyz = xyz

        # Assigning XYZ should work with any array like object
        xyz = np.array([
            [1.0527, 2.2566, 0.9925],
            [0.0014, 1.5578, 2.1146],
            [1.3456, 0.3678, 1.4251]
        ])
        spin_system.xyz = xyz
        self.assertTrue((spin_system.xyz == xyz).all())

        spin_system.xyz = list(xyz)
        self.assertTrue((spin_system.xyz == xyz).all())

        spin_system.xyz = tuple(xyz)
        self.assertTrue((spin_system.xyz == xyz).all())

        # Assigning XYZ should work from text file
        spin_system.xyz = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
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
        spin_system = sg.SpinSystem(["1H", "19F", "14N"])

        # Trying to configure shielding tensors for a spin system of incorrect
        # size should result in an error
        shielding = np.array([
            [[0, 0, 0],
             [0, 0, 0],
             [0, 0, 0]],
            [[101.6, -75.2, 11.1],
             [30.5,   10.1, 87.4],
             [99.7,  -21.1, 11.2]]
        ])
        with self.assertRaises(ValueError):
            spin_system.shielding = shielding

        # Assigning shielding tensors should work with any array like object
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
        spin_system.shielding = shielding
        self.assertTrue((spin_system.shielding == shielding).all())

        spin_system.shielding = list(shielding)
        self.assertTrue((spin_system.shielding == shielding).all())

        spin_system.shielding = tuple(shielding)
        self.assertTrue((spin_system.shielding == shielding).all())

        # Assigning shielding tensors should work from text file
        spin_system.shielding = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'test_data',
            'shielding.txt')
        self.assertTrue((spin_system.shielding == shielding).all())

        # Assigning shielding tensors with incorrect file path should result in
        # error
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
        spin_system = sg.SpinSystem(["1H", "19F", "14N"])

        # Trying to configure efg tensors for a spin system of incorrect size
        # should result in an error
        efg = np.array([
            [[0, 0, 0],
             [0, 0, 0],
             [0, 0, 0]],
            [[ 0.31, 0.00, 0.01],
             [-0.20, 0.04, 0.87],
             [ 0.11, 0.16, 0.65]]
        ])
        with self.assertRaises(ValueError):
            spin_system.efg = efg

        # Assigning efg tensors should work with any array like object
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
        spin_system.efg = efg
        self.assertTrue((spin_system.efg == efg).all())

        spin_system.efg = list(efg)
        self.assertTrue((spin_system.efg == efg).all())

        spin_system.efg = tuple(efg)
        self.assertTrue((spin_system.efg == efg).all())

        # Assigning efg tensors should work from text file
        spin_system.efg = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
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
        order for the basis.
        """

        # Initialize our SpinSystem object
        spin_system = sg.SpinSystem(['1H', '19F', '14N'])

        # Maximum spin orders within 1 and the number of spins should be fine
        for max_spin_order in range(1, spin_system.nspins+1):
            spin_system.basis.max_spin_order = max_spin_order

        # Setting the maximum spin order to less than 1 causes an error
        with self.assertRaises(ValueError):
            spin_system.basis.max_spin_order = 0

        # Setting the maximum spin order above number of spins causes an error
        with self.assertRaises(ValueError):
            spin_system.basis.max_spin_order = spin_system.nspins + 1

    def test_build_basis(self):
        """
        A test for creating a SpinSystem instance and building a basis set.
        """

        # Initialize our SpinSystem object
        spin_system = sg.SpinSystem(['1H', '19F', '14N'])

        # Building a basis without specifying the maximum spin order gives a
        # warning and sets the maximum spin order to the number of spins
        with self.assertWarns(Warning):
            spin_system.basis.build()
        self.assertEqual(spin_system.basis.max_spin_order, spin_system.nspins)
        self.assertTrue(isinstance(spin_system.basis.basis, np.ndarray))

    def test_set_relaxation_theory(self):
        """
        A test for creating a SpinSystem instance and setting the relaxation
        theory.
        """

        # Initialize our SpinSystem object
        spin_system = sg.SpinSystem(['1H', '19F', '14N'])

        # Set the relaxation theory
        relaxation_theory = "redfield"
        spin_system.relaxation.theory = relaxation_theory
        self.assertEqual(spin_system.relaxation.theory, relaxation_theory)

        relaxation_theory = "phenomenological"
        spin_system.relaxation.theory = relaxation_theory
        self.assertEqual(spin_system.relaxation.theory, relaxation_theory)

        # Setting an unknown theory should result in an error
        with self.assertRaises(ValueError):
            spin_system.relaxation.theory = "unknown_theory"

    def test_set_thermalization(self):
        """
        A test for adjusting the thermalization setting.
        """

        # Initialize our SpinSystem object
        spin_system = sg.SpinSystem(['1H', '19F', '14N'])

        # Set the thermalization setting
        thermalization = True
        spin_system.relaxation.thermalization = thermalization
        self.assertEqual(spin_system.relaxation.thermalization, thermalization)

    def test_set_tau_c(self):
        """
        A test for creating a SpinSystem instance and setting the correlation
        time.
        """

        # Initialize our SpinSystem object
        spin_system = sg.SpinSystem(['1H', '19F', '14N'])

        # Set the correlation time
        tau_c = 50e-12
        spin_system.relaxation.tau_c = tau_c
        self.assertEqual(spin_system.relaxation.tau_c, tau_c)

    def test_set_sr2k(self):
        """
        A test for creating a SpinSystem instance and setting the scalar
        relaxation of the second kind.
        """

        # Initialize our SpinSystem object
        spin_system = sg.SpinSystem(['1H', '19F', '14N'])

        # Set the sr2k
        sr2k = True
        spin_system.relaxation.sr2k = sr2k
        self.assertEqual(spin_system.relaxation.sr2k, sr2k)

    def test_set_dynamic_frequency_shift(self):
        """
        A test for creating a SpinSystem instance and setting the dynamic
        frequency shift.
        """

        # Initialize our SpinSystem object
        spin_system = sg.SpinSystem(['1H', '19F', '14N'])

        # Set the dynamic frequency shift
        dynamic_frequency_shift = True
        spin_system.relaxation.dynamic_frequency_shift = dynamic_frequency_shift
        self.assertEqual(spin_system.relaxation.dynamic_frequency_shift,
                         dynamic_frequency_shift)

    def test_set_antisymmetric_relaxation(self):
        """
        A test for creating a SpinSystem instance and setting the antisymmetric
        relaxation.
        """

        # Initialize our SpinSystem object
        spin_system = sg.SpinSystem(['1H', '19F', '14N'])

        # Set the antisymmetric relaxation
        antisymmetric_relaxation = True
        spin_system.relaxation.antisymmetric = antisymmetric_relaxation
        self.assertEqual(spin_system.relaxation.antisymmetric,
                         antisymmetric_relaxation)

    def test_set_relative_error(self):
        """
        A test for creating a SpinSystem instance and setting the relative error.
        """

        # Initialize our SpinSystem object
        spin_system = sg.SpinSystem(['1H', '19F', '14N'])

        # Set the relative error
        relative_error = 1e-12
        spin_system.relaxation.relative_error = relative_error
        self.assertEqual(spin_system.relaxation.relative_error, relative_error)

    def test_set_T1(self):
        """
        A test for setting the longitudinal relaxation time constants.
        """
        
        # Initialize our SpinSystem object
        spin_system = sg.SpinSystem(['1H', '19F', '14N'])

        # Trying to configure T1 times for a spin system of different size
        # results in an error
        T1 = [1, 2]
        with self.assertRaises(ValueError):
            spin_system.relaxation.T1 = T1

        # Test configuring T1
        T1 = [1, 2, 3]
        spin_system.relaxation.T1 = T1
        self.assertTrue((spin_system.relaxation.T1 == T1).all())

        # Assigning T1 should work from text file
        spin_system.relaxation.T1 = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'test_data',
            'T1.txt')
        self.assertTrue((spin_system.relaxation.T1 == T1).all())

    def test_set_T2(self):
        """
        A test for setting the transverse relaxation time constants.
        """
        
        # Initialize our SpinSystem object
        spin_system = sg.SpinSystem(['1H', '19F', '14N'])

        # Trying to configure T2 times for a spin system of different size
        # results in an error
        T2 = [1, 2]
        with self.assertRaises(ValueError):
            spin_system.relaxation.T2 = T2

        # Test configuring T2
        T2 = [1, 2, 3]
        spin_system.relaxation.T2 = T2
        self.assertTrue((spin_system.relaxation.T2 == T2).all())

        # Assigning T2 should work from text file
        spin_system.relaxation.T2 = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'test_data',
            'T2.txt')
        self.assertTrue((spin_system.relaxation.T2 == T2).all())

    def test_subsystem(self):
        """
        Test creating the subsystem from a spin system.
        """
        # Reset parameters to defaults
        sg.parameters.default()

        # Create an example spin system
        ss = sg.SpinSystem(["1H", "14N", "19F"])
        
        # Try that the subsystem works with nothing else assigned to the system
        sub = ss.subsystem([0, 1])
        self.assertTrue(np.array_equal(np.array(["1H", "14N"]), sub.isotopes))

        # Test that incorrect input leads to error
        with self.assertRaises(ValueError):
            ss.subsystem([0, 1, 1])
        with self.assertRaises(ValueError):
            ss.subsystem([1, 3])

        # Assign the spin system properties
        ss.chemical_shifts = [0, 1, 2]
        ss.J_couplings = [
            [0, 0, 0],
            [1, 0, 0],
            [2, 3, 0]
        ]
        ss.xyz = [
            [0, 0, 0],
            [1, 1, 1],
            [2, 2, 2]
        ]
        ss.shielding = [
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0]
            ],
            [
                [1, 1, 1],
                [1, 1, 1],
                [1, 1, 1]
            ],
            [
                [2, 2, 2],
                [2, 2, 2],
                [2, 2, 2]
            ]
        ]
        ss.efg = [
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0]
            ],
            [
                [1, 1, 1],
                [1, 1, 1],
                [1, 1, 1]
            ],
            [
                [2, 2, 2],
                [2, 2, 2],
                [2, 2, 2]
            ]
        ]

        # Create a subsystem with all spins
        sub = ss.subsystem([0, 1, 2])

        # Everything should remain the same
        self.assertTrue(np.array_equal(ss.isotopes, sub.isotopes))
        self.assertTrue(np.array_equal(ss.chemical_shifts, sub.chemical_shifts))
        self.assertTrue(np.array_equal(ss.J_couplings, sub.J_couplings))
        self.assertTrue(np.array_equal(ss.xyz, sub.xyz))
        self.assertTrue(np.array_equal(ss.shielding, sub.shielding))
        self.assertTrue(np.array_equal(ss.efg, sub.efg))

        # Create a subsystem with one spin
        sub = ss.subsystem([1])

        # Check that the properties were copied correctly
        self.assertTrue(np.array_equal(np.array(["14N"]), sub.isotopes))
        self.assertTrue(np.array_equal(np.array([1]), sub.chemical_shifts))
        self.assertTrue(np.array_equal(np.array([[0]]), sub.J_couplings))
        self.assertTrue(np.array_equal(np.array([[1, 1, 1]]), sub.xyz))
        self.assertTrue(np.array_equal(
            np.array([[
                [1, 1, 1],
                [1, 1, 1],
                [1, 1, 1]
            ]]),
            sub.shielding
        ))
        self.assertTrue(np.array_equal(
            np.array([[
                [1, 1, 1],
                [1, 1, 1],
                [1, 1, 1]
            ]]),
            sub.efg
        ))

        # Create a subsystem with two spins
        sub = ss.subsystem([0, 2])

        # Check that the properties were copied correctly
        self.assertTrue(np.array_equal(np.array(["1H", "19F"]), sub.isotopes))
        self.assertTrue(np.array_equal(np.array([0, 2]), sub.chemical_shifts))
        self.assertTrue(np.array_equal(
            np.array([
                [0, 0],
                [2, 0]
            ]),
            sub.J_couplings
        ))
        self.assertTrue(np.array_equal(
            np.array([
                [0, 0, 0],
                [2, 2, 2]
            ]),
            sub.xyz
        ))
        self.assertTrue(np.array_equal(
            np.array([
                [
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0]
                ],
                [
                    [2, 2, 2],
                    [2, 2, 2],
                    [2, 2, 2]
                ]
            ]),
            sub.shielding
        ))
        self.assertTrue(np.array_equal(
            np.array([
                [
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0]
                ],
                [
                    [2, 2, 2],
                    [2, 2, 2],
                    [2, 2, 2]
                ]
            ]),
            sub.efg
        ))