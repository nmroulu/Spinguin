"""
Tests for reading spin-system input data from text files.
"""

import os
import unittest

import numpy as np

import spinguin as sg
from ._helpers import test_data_path


class TestDataIOMethods(unittest.TestCase):
    """
    Test file-based import of arrays, coordinates, and tensors.
    """

    def test_read_array(self):
        """
        Test reading isotope, chemical-shift, and coupling arrays.
        """

        # Define the reference spin system explicitly.
        ss1 = sg.SpinSystem(['1H', '19F', '14N'])
        ss1.chemical_shifts = [8.00, -127.5, 40.50]
        ss1.J_couplings = [
            [0,     0,     0],
            [1.05,  0,     0],
            [0.50,  9.17,  0]
        ]

        # Read the same values from text files.
        ss2 = sg.SpinSystem(test_data_path('isotopes.txt'))
        ss2.chemical_shifts = test_data_path('chemical_shifts.txt')
        ss2.J_couplings = test_data_path('J_couplings.txt')

        # Compare the loaded values with the hard-coded reference values.
        self.assertTrue((ss1.isotopes == ss2.isotopes).all())
        self.assertTrue((ss1.chemical_shifts == ss2.chemical_shifts).all())
        self.assertTrue((ss1.J_couplings == ss2.J_couplings).all())

    def test_read_xyz(self):
        """
        Test reading Cartesian coordinates from a text file.
        """

        # Define the reference coordinates explicitly.
        xyz_1 = np.array([
            [1.0527, 2.2566, 0.9925],
            [0.0014, 1.5578, 2.1146],
            [1.3456, 0.3678, 1.4251]
        ])

        # Read the coordinates from a text file.
        ss2 = sg.SpinSystem(['1H', '19F', '14N'])
        ss2.xyz = test_data_path('xyz.txt')

        # Compare the loaded coordinates with the reference values.
        self.assertTrue((xyz_1 == ss2.xyz).all())
    
    def test_read_tensors(self):
        """
        Test reading shielding and EFG tensors from text files.
        """

        # Define the reference shielding tensors explicitly.
        shielding_1 = np.zeros((3, 3, 3))
        shielding_1[1] = np.array([
            [101.6, -75.2, 11.1],
            [30.5,   10.1, 87.4],
            [99.7,  -21.1, 11.2]
        ])
        shielding_1[2] = np.array([
            [171.9, -58.6, 91.1],
            [37.5,   10.7, 86.9],
            [109.7, -91.1, 81.8]
        ])

        # Define the reference electric-field-gradient tensors explicitly.
        efg_1 = np.zeros((3, 3, 3))
        efg_1[1] = np.array([
            [ 0.31, 0.00, 0.01],
            [-0.20, 0.04, 0.87],
            [ 0.11, 0.16, 0.65]
        ])
        efg_1[2] = np.array([
            [0.34, 0.67, 0.23],
            [0.38, 0.65, 0.26],
            [0.29, 0.82, 0.06]
        ])

        # Read the tensors from text files.
        ss2 = sg.SpinSystem(['1H', '19F', '14N'])
        ss2.shielding = test_data_path('shielding.txt')
        ss2.efg = test_data_path('efg.txt')

        # Compare the loaded tensors with the reference values.
        self.assertTrue((shielding_1 == ss2.shielding).all())
        self.assertTrue((efg_1 == ss2.efg).all())