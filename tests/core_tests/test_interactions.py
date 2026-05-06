"""
Tests for NMR interactions.
"""

import unittest

import numpy as np

import spinguin as sg

class TestInteractions(unittest.TestCase):
    """
    Tests for various NMR interactions.
    """

    def test_dd_constants(self):
        """
        Test the dipole-dipole (DD) relaxation constant calculation.
        Compares calculated values against tabulated values from the reference:
        Apperley, Harris & Hodgkinson: Solid-state NMR: Basic principles and
        practice.
        """

        # Create test spin systems
        ss1 = sg.SpinSystem(["13C", "13C"])
        ss2 = sg.SpinSystem(["13C", "1H"])
        ss3 = sg.SpinSystem(["13C", "15N"])

        # Assign Cartesian coordinates
        ss1.xyz = [
            [0.00, 0.00, 0.00],
            [1.53, 0.00, 0.00]
        ]
        ss2.xyz = [
            [0.00, 0.00, 0.00],
            [0.00, 1.06, 0.00]
        ]
        ss3.xyz = [
            [0.00, 0.00, 0.00],
            [0.00, 0.00, 1.47]
        ]

        # Calculate DD constants and convert to Hz
        dd_13C_13C = abs(sg.dd_constants(ss1)[1, 0] / (2*np.pi))
        dd_13C_1H = abs(sg.dd_constants(ss2)[1, 0] / (2*np.pi))
        dd_13C_15N = abs(sg.dd_constants(ss3)[1, 0] / (2*np.pi))

        # Compare with tabulated values (in Hz)
        self.assertTrue(np.allclose(2.12e3, dd_13C_13C, rtol=0.01))
        self.assertTrue(np.allclose(25.44e3, dd_13C_1H, rtol=0.01))
        self.assertTrue(np.allclose(0.97e3, dd_13C_15N, rtol=0.01))