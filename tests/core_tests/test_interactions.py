"""
Tests for NMR interactions.
"""

import unittest

import numpy as np

from spinguin._core._interactions import dd_constant
from spinguin._core._nmr_isotopes import ISOTOPES


class TestInteractions(unittest.TestCase):
    """
    Tests for various NMR interactions.
    """

    def test_dd_constant(self):
        """
        Test the dipole-dipole (DD) relaxation constant calculation.
        Compares calculated values against tabulated values from the reference:
        Apperley, Harris & Hodgkinson: Solid-state NMR: Basic principles and
        practice.
        """

        # Get gyromagnetic ratios (gamma) in Hz/T
        y_13C = 2 * np.pi * ISOTOPES['13C'][1] * 1e6
        y_1H = 2 * np.pi * ISOTOPES['1H'][1] * 1e6
        y_15N = 2 * np.pi * ISOTOPES['15N'][1] * 1e6

        # Interatomic distances in meters
        r_13C_13C = 0.153e-9
        r_13C_1H = 0.106e-9
        r_13C_15N = 0.147e-9

        # Calculate DD constants and convert to Hz
        dd_13C_13C = -dd_constant(y_13C, y_13C) / r_13C_13C**3 / (2 * np.pi)
        dd_13C_1H = -dd_constant(y_13C, y_1H) / r_13C_1H**3 / (2 * np.pi)
        dd_13C_15N = dd_constant(y_13C, y_15N) / r_13C_15N**3 / (2 * np.pi)

        # Compare with tabulated values (in Hz)
        self.assertTrue(np.allclose(2.12e3, dd_13C_13C, rtol=0.01))
        self.assertTrue(np.allclose(25.44e3, dd_13C_1H, rtol=0.01))
        self.assertTrue(np.allclose(0.97e3, dd_13C_15N, rtol=0.01))