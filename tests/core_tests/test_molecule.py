"""
Tests for `Molecule` class.
"""

import unittest
import numpy as np
import spinguin as sg

class TestMolecule(unittest.TestCase):
    """
    Test `Molecule` class.
    """

    def test_volume(self):
        """
        Calculate the volume of a single 1H atom and compare against reference.
        """
        molecule = sg.Molecule(["1H"], [[0, 0, 0]])
        V = molecule.volume
        V_ref = 4/3*np.pi*(1.2)**3
        self.assertTrue(abs(V - V_ref)/V_ref < 0.05)

    def test_hydrodynamic_radius(self):
        """
        Calculate the hydrodynamic radius of a single 1H atom and compare
        against reference.
        """
        molecule = sg.Molecule(["1H"], [[0, 0, 0]])
        r = molecule.hydrodynamic_radius
        r_ref = 1.2
        self.assertTrue(abs(r - r_ref)/r_ref < 0.05)