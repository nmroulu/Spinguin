"""
Tests for `SpinSystem` property assignment, relaxation settings, and
subsystem construction.
"""

import os
import unittest

import numpy as np

import spinguin as sg
from ._helpers import build_spin_system, test_data_path

class TestSpinSystem(unittest.TestCase):
    """
    Test `SpinSystem` input handling, configuration, and subsystem copying.
    """

    def test_assign_isotopes(self):
        """
        Test `SpinSystem` construction from isotope arrays and files.
        """

        # Define the reference isotope labels.
        isotopes = np.array(['1H', '19F', '14N'])

        # Construct the spin system from several array-like inputs.
        for value in (isotopes, list(isotopes), tuple(isotopes)):
            spin_system = sg.SpinSystem(value)
            self.assertTrue(np.array_equal(spin_system.isotopes, isotopes))

        # Construct the spin system from a text file.
        spin_system = sg.SpinSystem(test_data_path("isotopes.txt"))
        self.assertTrue(np.array_equal(spin_system.isotopes, isotopes))

        # Verify that an incorrect file path raises an error.
        with self.assertRaises(FileNotFoundError):
            spin_system = sg.SpinSystem("not_a_real_file.txt")

        # Verify that incorrectly shaped isotope data raise an error.
        with self.assertRaises(ValueError):
            spin_system = sg.SpinSystem([['1H', '19F', '14N']])

        # Verify that unknown isotopes raise an error.
        with self.assertRaises(ValueError):
            spin_system = sg.SpinSystem(["4H"])

    def test_assign_chemical_shifts(self):
        """
        Test chemical-shift assignment from array-like objects and files.
        """

        # Create the reference spin system.
        spin_system = sg.SpinSystem(["1H", "19F", "14N"])

        # Verify that an incorrect number of chemical shifts raises an error.
        with self.assertRaises(ValueError):
            spin_system.chemical_shifts = [8.00, -127.5]

        # Verify assignment from several array-like input types.
        delta = np.array([8.00, -127.5, 40.50])
        for value in (delta, list(delta), tuple(delta)):
            spin_system.chemical_shifts = value
            self.assertTrue(np.array_equal(spin_system.chemical_shifts, delta))

        # Verify assignment from a text file.
        spin_system.chemical_shifts = test_data_path("chemical_shifts.txt")
        self.assertTrue(np.array_equal(spin_system.chemical_shifts, delta))

        # Verify that an incorrect file path raises an error.
        with self.assertRaises(FileNotFoundError):
            spin_system.chemical_shifts = "not_a_real_file.txt"

    def test_assign_J_couplings(self):
        """
        A test for creating a SpinSystem instance and assigning J-couplings
        using different input types.
        """

        # Create the reference spin system.
        spin_system = sg.SpinSystem(["1H", "19F", "14N"])

        # Verify that an incorrectly sized coupling matrix raises an error.
        J_couplings = np.array([
            [0,    0],
            [1.05, 0]
        ])
        with self.assertRaises(ValueError):
            spin_system.J_couplings = J_couplings

        # Verify assignment from several array-like input types.
        J_couplings = np.array([
            [0,    0,    0],
            [1.05, 0,    0],
            [0.50, 9.17, 0]
        ])
        for value in (J_couplings, list(J_couplings), tuple(J_couplings)):
            spin_system.J_couplings = value
            self.assertTrue(
                np.array_equal(spin_system.J_couplings, J_couplings)
            )

        # Verify assignment from a text file.
        spin_system.J_couplings = test_data_path("J_couplings.txt")
        self.assertTrue(np.array_equal(spin_system.J_couplings, J_couplings))

        # Verify that an incorrect file path raises an error.
        with self.assertRaises(FileNotFoundError):
            spin_system.J_couplings = "not_a_real_file.txt"

    def test_assign_xyz(self):
        """
        Test Cartesian-coordinate assignment from array-like objects and files.
        """

        # Create the reference spin system.
        spin_system = sg.SpinSystem(["1H", "19F", "14N"])

        # Verify that an incorrect coordinate array size raises an error.
        xyz = np.array([
            [1.0527, 2.2566, 0.9925],
            [0.0014, 1.5578, 2.1146]
        ])
        with self.assertRaises(ValueError):
            spin_system.xyz = xyz

        # Verify assignment from several array-like input types.
        xyz = np.array([
            [1.0527, 2.2566, 0.9925],
            [0.0014, 1.5578, 2.1146],
            [1.3456, 0.3678, 1.4251]
        ])
        for value in (xyz, list(xyz), tuple(xyz)):
            spin_system.xyz = value
            self.assertTrue(np.array_equal(spin_system.xyz, xyz))

        # Verify assignment from a text file.
        spin_system.xyz = test_data_path("xyz.txt")
        self.assertTrue(np.array_equal(spin_system.xyz, xyz))

        # Verify that an incorrect file path raises an error.
        with self.assertRaises(FileNotFoundError):
            spin_system.xyz = "not_a_real_file.txt"

        # Verify that incorrectly shaped coordinate data raise an error.
        xyz = np.array([[
            [1.0527, 2.2566, 0.9925],
            [0.0014, 1.5578, 2.1146],
            [1.3456, 0.3678, 1.4251]
        ]])
        with self.assertRaises(ValueError):
            spin_system.xyz = xyz

    def test_assign_shielding(self):
        """
        Test shielding-tensor assignment from array-like objects and files.
        """

        # Create the reference spin system.
        spin_system = sg.SpinSystem(["1H", "19F", "14N"])

        # Verify that an incorrect number of shielding tensors raises an error.
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

        # Verify assignment from several array-like input types.
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
        for value in (shielding, list(shielding), tuple(shielding)):
            spin_system.shielding = value
            self.assertTrue(np.array_equal(spin_system.shielding, shielding))

        # Verify assignment from a text file.
        spin_system.shielding = test_data_path("shielding.txt")
        self.assertTrue(np.array_equal(spin_system.shielding, shielding))

        # Verify that an incorrect file path raises an error.
        with self.assertRaises(FileNotFoundError):
            spin_system.shielding = "not_a_real_file.txt"

        # Verify that incorrectly shaped shielding data raise an error.
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
        Test EFG-tensor assignment from array-like objects and files.
        """

        # Create the reference spin system.
        spin_system = sg.SpinSystem(["1H", "19F", "14N"])

        # Verify that an incorrect number of EFG tensors raises an error.
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

        # Verify assignment from several array-like input types.
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
        for value in (efg, list(efg), tuple(efg)):
            spin_system.efg = value
            self.assertTrue(np.array_equal(spin_system.efg, efg))

        # Verify assignment from a text file.
        spin_system.efg = test_data_path("efg.txt")
        self.assertTrue(np.array_equal(spin_system.efg, efg))

        # Verify that an incorrect file path raises an error.
        with self.assertRaises(FileNotFoundError):
            spin_system.efg = "not_a_real_file.txt"

        # Verify that incorrectly shaped EFG data raise an error.
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
        Test setting the maximum spin order for the basis.
        """

        # Create the reference spin system.
        spin_system = sg.SpinSystem(['1H', '19F', '14N'])

        # Verify that valid spin orders can be assigned.
        for max_spin_order in range(1, spin_system.nspins + 1):
            spin_system.basis.max_spin_order = max_spin_order
            self.assertEqual(spin_system.basis.max_spin_order, max_spin_order)

        # Verify that spin orders below one raise an error.
        with self.assertRaises(ValueError):
            spin_system.basis.max_spin_order = 0

        # Verify that spin orders above the system size raise an error.
        with self.assertRaises(ValueError):
            spin_system.basis.max_spin_order = spin_system.nspins + 1

    def test_build_basis(self):
        """
        Test basis construction for a new spin system.
        """

        # Create the reference spin system.
        spin_system = sg.SpinSystem(['1H', '19F', '14N'])

        # Verify that the default basis build emits a warning.
        with self.assertWarns(Warning):
            spin_system.basis.build()

        # Verify that the basis settings were updated correctly.
        self.assertEqual(spin_system.basis.max_spin_order, spin_system.nspins)
        self.assertTrue(isinstance(spin_system.basis.basis, np.ndarray))

    def test_set_relaxation_theory(self):
        """
        Test assigning the relaxation-theory name.
        """

        # Create the reference spin system.
        spin_system = sg.SpinSystem(['1H', '19F', '14N'])

        # Verify that known theory labels are accepted.
        for theory in ("redfield", "phenomenological"):
            spin_system.relaxation.theory = theory
            self.assertEqual(spin_system.relaxation.theory, theory)

        # Verify that an unknown theory label raises an error.
        with self.assertRaises(ValueError):
            spin_system.relaxation.theory = "unknown_theory"

    def test_set_thermalization(self):
        """
        Test assigning the thermalization setting.
        """

        # Create the reference spin system.
        spin_system = sg.SpinSystem(['1H', '19F', '14N'])

        # Assign and verify the thermalization flag.
        for value in (False, True):
            spin_system.relaxation.thermalization = value
            self.assertEqual(spin_system.relaxation.thermalization, value)

    def test_set_tau_c(self):
        """
        Test assigning the rotational correlation time.
        """

        # Create the reference spin system.
        spin_system = sg.SpinSystem(['1H', '19F', '14N'])

        # Set and verify correlation time for isotropic rotational diffusion.
        tau_c = 50e-12
        spin_system.relaxation.tau_c = tau_c
        self.assertEqual(spin_system.relaxation.tau_c, tau_c)

        # Set and verify correlation time for anisotropic rotational diffusion.
        tau_c = np.array([10e-12, 20e-12, 30e-12])
        for value in (tau_c, list(tau_c), tuple(tau_c)):
            spin_system.relaxation.tau_c = value
            self.assertTrue(
                np.array_equal(spin_system.relaxation.tau_c, tau_c)
            )

        # Verify that unsupported input raises an error
        with self.assertRaises(ValueError):
            spin_system.relaxation.tau_c = [10e-12, 20e-12]

    def test_set_sr2k(self):
        """
        Test assigning the SR2K setting.
        """

        # Create the reference spin system.
        spin_system = sg.SpinSystem(['1H', '19F', '14N'])

        # Assign and verify the SR2K flag.
        for sr2k in (False, True):
            spin_system.relaxation.sr2k = sr2k
            self.assertEqual(spin_system.relaxation.sr2k, sr2k)

    def test_set_dynamic_frequency_shift(self):
        """
        Test assigning the dynamic-frequency-shift setting.
        """

        # Create the reference spin system.
        spin_system = sg.SpinSystem(['1H', '19F', '14N'])

        # Assign and verify the dynamic frequency shift flag.
        for dfs in (False, True):
            spin_system.relaxation.dynamic_frequency_shift = dfs
            self.assertEqual(
                spin_system.relaxation.dynamic_frequency_shift,
                dfs
            )

    def test_set_antisymmetric_relaxation(self):
        """
        Test assigning the antisymmetric-relaxation setting.
        """

        # Create the reference spin system.
        spin_system = sg.SpinSystem(['1H', '19F', '14N'])

        # Assign and verify the antisymmetric relaxation flag.
        for anti in (False, True):
            spin_system.relaxation.antisymmetric = anti
            self.assertEqual(spin_system.relaxation.antisymmetric, anti)

    def test_set_relative_error(self):
        """
        Test assigning the relaxation relative-error threshold.
        """

        # Create the reference spin system.
        spin_system = sg.SpinSystem(['1H', '19F', '14N'])

        # Assign and verify the relative error.
        relative_error = 1e-12
        spin_system.relaxation.relative_error = relative_error
        self.assertEqual(spin_system.relaxation.relative_error, relative_error)

    def test_set_R1(self):
        """
        Test assigning longitudinal relaxation rates.
        """

        # Create the reference spin system.
        ss = sg.SpinSystem(['1H', '14N'])

        # Verify that an incorrect number of rates raises an error.
        with self.assertRaises(ValueError):
            ss.relaxation.R1 = [1, 0.5, 2]

        # Assign and verify the relaxation rates and derived T1 values.
        R1 = [1, 0.5]
        ss.relaxation.R1 = R1
        self.assertTrue(np.array_equal(ss.relaxation.R1, R1))
        self.assertTrue(np.array_equal(ss.relaxation.T1, 1/np.array(R1)))

        # Verify assignment from a text file.
        ss.relaxation.R1 = test_data_path("R1.txt")
        self.assertTrue(np.array_equal(ss.relaxation.R1, R1))

    def test_set_R2(self):
        """
        Test assigning transverse relaxation rates.
        """

        # Create the reference spin system.
        ss = sg.SpinSystem(['1H', '14N'])

        # Verify that an incorrect number of rates raises an error.
        with self.assertRaises(ValueError):
            ss.relaxation.R2 = [1, 0.5, 2]

        # Assign and verify the relaxation rates and derived T2 values.
        R2 = [1, 0.5]
        ss.relaxation.R2 = R2
        self.assertTrue(np.array_equal(ss.relaxation.R2, R2))
        self.assertTrue(np.array_equal(ss.relaxation.T2, 1/np.array(R2)))

        # Verify assignment from a text file.
        ss.relaxation.R2 = test_data_path("R2.txt")
        self.assertTrue((ss.relaxation.R2 == R2).all())

    def test_set_T1(self):
        """
        Test assigning longitudinal relaxation time constants.
        """
        
        # Create the reference spin system.
        spin_system = sg.SpinSystem(['1H', '19F', '14N'])

        # Verify that an incorrect number of T1 values raises an error.
        with self.assertRaises(ValueError):
            spin_system.relaxation.T1 = [1, 2]

        # Assign and verify the T1 values.
        T1 = [1, 2, 3]
        spin_system.relaxation.T1 = T1
        self.assertTrue(np.array_equal(spin_system.relaxation.T1, T1))

        # Verify assignment from a text file.
        spin_system.relaxation.T1 = test_data_path("T1.txt")
        self.assertTrue(np.array_equal(spin_system.relaxation.T1, T1))

    def test_set_T2(self):
        """
        Test assigning transverse relaxation time constants.
        """
        
        # Create the reference spin system.
        spin_system = sg.SpinSystem(['1H', '19F', '14N'])

        # Verify that an incorrect number of T2 values raises an error.
        with self.assertRaises(ValueError):
            spin_system.relaxation.T2 = [1, 2]

        # Assign and verify the T2 values.
        T2 = [1, 2, 3]
        spin_system.relaxation.T2 = T2
        self.assertTrue(np.array_equal(spin_system.relaxation.T2, T2))

        # Verify assignment from a text file.
        spin_system.relaxation.T2 = test_data_path("T2.txt")
        self.assertTrue(np.array_equal(spin_system.relaxation.T2, T2))

    def test_subsystem(self):
        """
        Test creating subsystems with copied properties.
        """
        # Reset parameters to defaults.
        sg.parameters.default()

        # Create the example spin system
        ss = sg.SpinSystem(["1H", "14N", "19F"])
        
        # Verify that isotopes are copied even without other assigned data.
        sub = ss.subsystem([0, 1])
        self.assertTrue(np.array_equal(np.array(["1H", "14N"]), sub.isotopes))

        # Verify that invalid subsystem indices raise errors.
        with self.assertRaises(ValueError):
            ss.subsystem([0, 1, 1])
        with self.assertRaises(ValueError):
            ss.subsystem([1, 3])

        # Create the fully populated reference spin system.
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

        # Verify that a full subsystem reproduces all properties.
        sub = ss.subsystem([0, 1, 2])
        self.assertTrue(np.array_equal(ss.isotopes, sub.isotopes))
        self.assertTrue(np.array_equal(ss.chemical_shifts, sub.chemical_shifts))
        self.assertTrue(np.array_equal(ss.J_couplings, sub.J_couplings))
        self.assertTrue(np.array_equal(ss.xyz, sub.xyz))
        self.assertTrue(np.array_equal(ss.shielding, sub.shielding))
        self.assertTrue(np.array_equal(ss.efg, sub.efg))

        # Verify that a one-spin subsystem copies all properties correctly.
        sub = ss.subsystem([1])
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

        # Verify that a two-spin subsystem copies all properties correctly.
        sub = ss.subsystem([0, 2])
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