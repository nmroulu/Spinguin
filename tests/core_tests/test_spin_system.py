import os
import unittest

import numpy as np

import spinguin as sg


"""
Tests for `SpinSystem` property assignment, relaxation settings, and
subsystem construction.
"""


class TestSpinSystem(unittest.TestCase):
    """
    Test `SpinSystem` input handling, configuration, and subsystem copying.
    """

    def _get_test_data_path(
        self,
        filename,
    ):
        """
        Return the absolute path to a file in the shared test-data directory.

        Parameters
        ----------
        filename : str
            Name of the requested test-data file.

        Returns
        -------
        str
            Absolute path to the requested file.
        """

        # Locate the shared test-data directory.
        test_data_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "test_data",
        )

        return os.path.join(test_data_dir, filename)

    def _build_populated_spin_system(
        self,
    ):
        """
        Create a small spin system with all tested properties assigned.

        Returns
        -------
        SpinSystem
            Spin system with shifts, couplings, coordinates, shielding, and EFG
            tensors assigned.
        """

        # Create the reference spin system used in subsystem tests.
        spin_system = sg.SpinSystem(["1H", "14N", "19F"])

        # Assign the spin-system properties.
        spin_system.chemical_shifts = [0, 1, 2]
        spin_system.J_couplings = [
            [0, 0, 0],
            [1, 0, 0],
            [2, 3, 0],
        ]
        spin_system.xyz = [
            [0, 0, 0],
            [1, 1, 1],
            [2, 2, 2],
        ]
        spin_system.shielding = [
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
            ],
            [
                [1, 1, 1],
                [1, 1, 1],
                [1, 1, 1],
            ],
            [
                [2, 2, 2],
                [2, 2, 2],
                [2, 2, 2],
            ],
        ]
        spin_system.efg = [
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
            ],
            [
                [1, 1, 1],
                [1, 1, 1],
                [1, 1, 1],
            ],
            [
                [2, 2, 2],
                [2, 2, 2],
                [2, 2, 2],
            ],
        ]

        return spin_system

    def _assert_array_equal(
        self,
        value,
        reference,
    ):
        """
        Assert that two arrays are exactly equal.

        Parameters
        ----------
        value : array_like
            Tested array.
        reference : array_like
            Reference array.

        Returns
        -------
        None
            The assertion is evaluated in place.
        """

        # Compare the tested and reference arrays elementwise.
        self.assertTrue(np.array_equal(value, reference))

    def _assert_attribute_accepts_array_like(
        self,
        spin_system,
        attribute_name,
        reference,
    ):
        """
        Assert that an attribute accepts NumPy, list, and tuple inputs.

        Parameters
        ----------
        spin_system : SpinSystem
            Spin system whose attribute is assigned.
        attribute_name : str
            Name of the tested attribute.
        reference : array_like
            Reference value used for all assignments.

        Returns
        -------
        None
            The assertions are evaluated in place.
        """

        # Assign the attribute from several array-like input types.
        for value in (reference, list(reference), tuple(reference)):
            setattr(spin_system, attribute_name, value)
            self._assert_array_equal(getattr(spin_system, attribute_name), reference)

    def _set_and_assert_relaxation_attribute(
        self,
        attribute_name,
        value,
    ):
        """
        Assign and verify a scalar relaxation attribute.

        Parameters
        ----------
        attribute_name : str
            Name of the relaxation attribute.
        value : object
            Value assigned to the relaxation attribute.

        Returns
        -------
        None
            The assertion is evaluated in place.
        """

        # Create a fresh spin system for the relaxation-setting test.
        spin_system = sg.SpinSystem(["1H", "19F", "14N"])

        # Assign and verify the requested relaxation attribute.
        setattr(spin_system.relaxation, attribute_name, value)
        self.assertEqual(getattr(spin_system.relaxation, attribute_name), value)

    def test_assign_isotopes(self):
        """
        Test `SpinSystem` construction from isotope arrays and files.
        """

        # Define the reference isotope labels.
        isotopes = np.array(["1H", "19F", "14N"])

        # Construct the spin system from several array-like inputs.
        for value in (isotopes, list(isotopes), tuple(isotopes)):
            spin_system = sg.SpinSystem(value)
            self._assert_array_equal(spin_system.isotopes, isotopes)

        # Construct the spin system from a text file.
        spin_system = sg.SpinSystem(self._get_test_data_path("isotopes.txt"))
        self._assert_array_equal(spin_system.isotopes, isotopes)

        # Verify that an incorrect file path raises an error.
        with self.assertRaises(FileNotFoundError):
            sg.SpinSystem("not_a_real_file.txt")

        # Verify that incorrectly shaped isotope data raise an error.
        with self.assertRaises(ValueError):
            sg.SpinSystem(np.array([["1H", "19F", "14N"]]))

        # Verify that unknown isotopes raise an error.
        with self.assertRaises(ValueError):
            sg.SpinSystem(np.array(["4H"]))

    def test_assign_chemical_shifts(self):
        """
        Test chemical-shift assignment from array-like objects and files.
        """

        # Create the reference spin system.
        spin_system = sg.SpinSystem(["1H", "19F", "14N"])

        # Verify that an incorrect number of chemical shifts raises an error.
        chemical_shifts = np.array([8.00, -127.5])
        with self.assertRaises(ValueError):
            spin_system.chemical_shifts = chemical_shifts

        # Verify assignment from several array-like input types.
        chemical_shifts = np.array([8.00, -127.5, 40.50])
        self._assert_attribute_accepts_array_like(
            spin_system,
            "chemical_shifts",
            chemical_shifts,
        )

        # Verify assignment from a text file.
        spin_system.chemical_shifts = self._get_test_data_path(
            "chemical_shifts.txt"
        )
        self._assert_array_equal(spin_system.chemical_shifts, chemical_shifts)

        # Verify that an incorrect file path raises an error.
        with self.assertRaises(FileNotFoundError):
            spin_system.chemical_shifts = "not_a_real_file.txt"

    def test_assign_J_couplings(self):
        """
        Test J-coupling assignment from array-like objects and files.
        """

        # Create the reference spin system.
        spin_system = sg.SpinSystem(["1H", "19F", "14N"])

        # Verify that an incorrectly sized coupling matrix raises an error.
        j_couplings = np.array([
            [0,    0],
            [1.05, 0],
        ])
        with self.assertRaises(ValueError):
            spin_system.J_couplings = j_couplings

        # Verify assignment from several array-like input types.
        j_couplings = np.array([
            [0,    0,    0],
            [1.05, 0,    0],
            [0.50, 9.17, 0],
        ])
        self._assert_attribute_accepts_array_like(
            spin_system,
            "J_couplings",
            j_couplings,
        )

        # Verify assignment from a text file.
        spin_system.J_couplings = self._get_test_data_path("J_couplings.txt")
        self._assert_array_equal(spin_system.J_couplings, j_couplings)

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
            [0.0014, 1.5578, 2.1146],
        ])
        with self.assertRaises(ValueError):
            spin_system.xyz = xyz

        # Verify assignment from several array-like input types.
        xyz = np.array([
            [1.0527, 2.2566, 0.9925],
            [0.0014, 1.5578, 2.1146],
            [1.3456, 0.3678, 1.4251],
        ])
        self._assert_attribute_accepts_array_like(spin_system, "xyz", xyz)

        # Verify assignment from a text file.
        spin_system.xyz = self._get_test_data_path("xyz.txt")
        self._assert_array_equal(spin_system.xyz, xyz)

        # Verify that an incorrect file path raises an error.
        with self.assertRaises(FileNotFoundError):
            spin_system.xyz = "not_a_real_file.txt"

        # Verify that incorrectly shaped coordinate data raise an error.
        with self.assertRaises(ValueError):
            spin_system.xyz = np.array([
                [
                    [1.0527, 2.2566, 0.9925],
                    [0.0014, 1.5578, 2.1146],
                    [1.3456, 0.3678, 1.4251],
                ]
            ])

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
             [99.7,  -21.1, 11.2]],
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
             [109.7, -91.1, 81.8]],
        ])
        self._assert_attribute_accepts_array_like(
            spin_system,
            "shielding",
            shielding,
        )

        # Verify assignment from a text file.
        spin_system.shielding = self._get_test_data_path("shielding.txt")
        self._assert_array_equal(spin_system.shielding, shielding)

        # Verify that an incorrect file path raises an error.
        with self.assertRaises(FileNotFoundError):
            spin_system.shielding = "not_a_real_file.txt"

        # Verify that incorrectly shaped shielding data raise an error.
        with self.assertRaises(ValueError):
            spin_system.shielding = np.array([
                [
                    [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                    [[101.6, -75.2, 11.1], [30.5, 10.1, 87.4],
                     [99.7, -21.1, 11.2]],
                    [[171.9, -58.6, 91.1], [37.5, 10.7, 86.9],
                     [109.7, -91.1, 81.8]],
                ]
            ])

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
             [ 0.11, 0.16, 0.65]],
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
             [0.29, 0.82, 0.06]],
        ])
        self._assert_attribute_accepts_array_like(spin_system, "efg", efg)

        # Verify assignment from a text file.
        spin_system.efg = self._get_test_data_path("efg.txt")
        self._assert_array_equal(spin_system.efg, efg)

        # Verify that an incorrect file path raises an error.
        with self.assertRaises(FileNotFoundError):
            spin_system.efg = "not_a_real_file.txt"

        # Verify that incorrectly shaped EFG data raise an error.
        with self.assertRaises(ValueError):
            spin_system.efg = np.array([
                [
                    [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                    [[0.31, 0.00, 0.01], [-0.20, 0.04, 0.87],
                     [0.11, 0.16, 0.65]],
                    [[0.34, 0.67, 0.23], [0.38, 0.65, 0.26],
                     [0.29, 0.82, 0.06]],
                ]
            ])

    def test_set_maximum_spin_order(self):
        """
        Test setting the maximum spin order for the basis.
        """

        # Create the reference spin system.
        spin_system = sg.SpinSystem(["1H", "19F", "14N"])

        # Verify that valid spin orders can be assigned.
        for max_spin_order in range(1, spin_system.nspins + 1):
            spin_system.basis.max_spin_order = max_spin_order

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
        spin_system = sg.SpinSystem(["1H", "19F", "14N"])

        # Verify that the default basis build emits a warning.
        with self.assertWarns(Warning):
            spin_system.basis.build()

        # Verify that the basis settings were updated correctly.
        self.assertEqual(spin_system.basis.max_spin_order, spin_system.nspins)
        self.assertIsInstance(spin_system.basis.basis, np.ndarray)

    def test_set_relaxation_theory(self):
        """
        Test assigning the relaxation-theory name.
        """

        # Create the reference spin system.
        spin_system = sg.SpinSystem(["1H", "19F", "14N"])

        # Verify that known theory labels are accepted.
        for relaxation_theory in ("redfield", "phenomenological"):
            spin_system.relaxation.theory = relaxation_theory
            self.assertEqual(spin_system.relaxation.theory, relaxation_theory)

        # Verify that an unknown theory label raises an error.
        with self.assertRaises(ValueError):
            spin_system.relaxation.theory = "unknown_theory"

    def test_set_thermalization(self):
        """
        Test assigning the thermalization setting.
        """

        # Assign and verify the thermalization flag.
        self._set_and_assert_relaxation_attribute("thermalization", True)

    def test_set_tau_c(self):
        """
        Test assigning the rotational correlation time.
        """

        # Assign and verify the correlation time.
        self._set_and_assert_relaxation_attribute("tau_c", 50e-12)

    def test_set_sr2k(self):
        """
        Test assigning the SR2K setting.
        """

        # Assign and verify the SR2K flag.
        self._set_and_assert_relaxation_attribute("sr2k", True)

    def test_set_dynamic_frequency_shift(self):
        """
        Test assigning the dynamic-frequency-shift setting.
        """

        # Assign and verify the dynamic frequency shift flag.
        self._set_and_assert_relaxation_attribute(
            "dynamic_frequency_shift",
            True,
        )

    def test_set_antisymmetric_relaxation(self):
        """
        Test assigning the antisymmetric-relaxation setting.
        """

        # Assign and verify the antisymmetric relaxation flag.
        self._set_and_assert_relaxation_attribute("antisymmetric", True)

    def test_set_relative_error(self):
        """
        Test assigning the relaxation relative-error threshold.
        """

        # Assign and verify the relative error.
        self._set_and_assert_relaxation_attribute("relative_error", 1e-12)

    def test_set_R1(self):
        """
        Test assigning longitudinal relaxation rates.
        """

        # Create the reference spin system.
        spin_system = sg.SpinSystem(["1H", "14N"])

        # Verify that an incorrect number of rates raises an error.
        with self.assertRaises(ValueError):
            spin_system.relaxation.R1 = [1, 0.5, 2]

        # Assign and verify the relaxation rates and derived T1 values.
        r1 = [1, 0.5]
        spin_system.relaxation.R1 = r1
        self._assert_array_equal(spin_system.relaxation.R1, r1)
        self._assert_array_equal(spin_system.relaxation.T1, 1 / np.array(r1))

        # Verify assignment from a text file.
        spin_system.relaxation.R1 = self._get_test_data_path("R1.txt")
        self._assert_array_equal(spin_system.relaxation.R1, r1)

    def test_set_R2(self):
        """
        Test assigning transverse relaxation rates.
        """

        # Create the reference spin system.
        spin_system = sg.SpinSystem(["1H", "14N"])

        # Verify that an incorrect number of rates raises an error.
        with self.assertRaises(ValueError):
            spin_system.relaxation.R2 = [1, 0.5, 2]

        # Assign and verify the relaxation rates and derived T2 values.
        r2 = [1, 0.5]
        spin_system.relaxation.R2 = r2
        self._assert_array_equal(spin_system.relaxation.R2, r2)
        self._assert_array_equal(spin_system.relaxation.T2, 1 / np.array(r2))

        # Verify assignment from a text file.
        spin_system.relaxation.R2 = self._get_test_data_path("R2.txt")
        self._assert_array_equal(spin_system.relaxation.R2, r2)

    def test_set_T1(self):
        """
        Test assigning longitudinal relaxation time constants.
        """

        # Create the reference spin system.
        spin_system = sg.SpinSystem(["1H", "19F", "14N"])

        # Verify that an incorrect number of T1 values raises an error.
        with self.assertRaises(ValueError):
            spin_system.relaxation.T1 = [1, 2]

        # Assign and verify the T1 values.
        t1 = [1, 2, 3]
        spin_system.relaxation.T1 = t1
        self._assert_array_equal(spin_system.relaxation.T1, t1)

        # Verify assignment from a text file.
        spin_system.relaxation.T1 = self._get_test_data_path("T1.txt")
        self._assert_array_equal(spin_system.relaxation.T1, t1)

    def test_set_T2(self):
        """
        Test assigning transverse relaxation time constants.
        """

        # Create the reference spin system.
        spin_system = sg.SpinSystem(["1H", "19F", "14N"])

        # Verify that an incorrect number of T2 values raises an error.
        with self.assertRaises(ValueError):
            spin_system.relaxation.T2 = [1, 2]

        # Assign and verify the T2 values.
        t2 = [1, 2, 3]
        spin_system.relaxation.T2 = t2
        self._assert_array_equal(spin_system.relaxation.T2, t2)

        # Verify assignment from a text file.
        spin_system.relaxation.T2 = self._get_test_data_path("T2.txt")
        self._assert_array_equal(spin_system.relaxation.T2, t2)

    def test_subsystem(self):
        """
        Test creating subsystems with copied properties.
        """

        # Reset parameters to defaults.
        sg.parameters.default()

        # Create the example spin system.
        spin_system = sg.SpinSystem(["1H", "14N", "19F"])

        # Verify that isotopes are copied even without other assigned data.
        subsystem = spin_system.subsystem([0, 1])
        self._assert_array_equal(np.array(["1H", "14N"]), subsystem.isotopes)

        # Verify that invalid subsystem indices raise errors.
        with self.assertRaises(ValueError):
            spin_system.subsystem([0, 1, 1])
        with self.assertRaises(ValueError):
            spin_system.subsystem([1, 3])

        # Create the fully populated reference spin system.
        spin_system = self._build_populated_spin_system()

        # Verify that a full subsystem reproduces all properties.
        subsystem = spin_system.subsystem([0, 1, 2])
        self._assert_array_equal(spin_system.isotopes, subsystem.isotopes)
        self._assert_array_equal(
            spin_system.chemical_shifts,
            subsystem.chemical_shifts,
        )
        self._assert_array_equal(spin_system.J_couplings, subsystem.J_couplings)
        self._assert_array_equal(spin_system.xyz, subsystem.xyz)
        self._assert_array_equal(spin_system.shielding, subsystem.shielding)
        self._assert_array_equal(spin_system.efg, subsystem.efg)

        # Verify that a one-spin subsystem copies all properties correctly.
        subsystem = spin_system.subsystem([1])
        self._assert_array_equal(np.array(["14N"]), subsystem.isotopes)
        self._assert_array_equal(np.array([1]), subsystem.chemical_shifts)
        self._assert_array_equal(np.array([[0]]), subsystem.J_couplings)
        self._assert_array_equal(np.array([[1, 1, 1]]), subsystem.xyz)
        self._assert_array_equal(
            np.array([[[1, 1, 1], [1, 1, 1], [1, 1, 1]]]),
            subsystem.shielding,
        )
        self._assert_array_equal(
            np.array([[[1, 1, 1], [1, 1, 1], [1, 1, 1]]]),
            subsystem.efg,
        )

        # Verify that a two-spin subsystem copies all properties correctly.
        subsystem = spin_system.subsystem([0, 2])
        self._assert_array_equal(np.array(["1H", "19F"]), subsystem.isotopes)
        self._assert_array_equal(np.array([0, 2]), subsystem.chemical_shifts)
        self._assert_array_equal(
            np.array([[0, 0], [2, 0]]),
            subsystem.J_couplings,
        )
        self._assert_array_equal(
            np.array([[0, 0, 0], [2, 2, 2]]),
            subsystem.xyz,
        )
        self._assert_array_equal(
            np.array(
                [
                    [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                    [[2, 2, 2], [2, 2, 2], [2, 2, 2]],
                ]
            ),
            subsystem.shielding,
        )
        self._assert_array_equal(
            np.array(
                [
                    [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                    [[2, 2, 2], [2, 2, 2], [2, 2, 2]],
                ]
            ),
            subsystem.efg,
        )