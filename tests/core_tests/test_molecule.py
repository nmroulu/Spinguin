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

    def test_vdw_radii(self):
        """
        Test the default and custom van der Waals radii.
        """
        # Use the Bondi radius of carbon by default.
        molecule = sg.Molecule(["13C"], [[0, 0, 0]])
        self.assertTrue(abs(molecule.hydrodynamic_radius - 1.70)/1.70 < 0.01)

        # Use a custom radius once it has been assigned.
        molecule.vdw_radii = [2.0]
        self.assertTrue(abs(molecule.hydrodynamic_radius - 2.0)/2.0 < 0.01)

        # Require custom radii for elements without a default radius.
        molecule = sg.Molecule(["193Ir"], [[0, 0, 0]])
        with self.assertRaises(ValueError):
            molecule.volume

    def test_semi_axes(self):
        """
        Test the ellipsoid and the principal axes of the van der Waals solid.
        """
        # A single atom is represented by a sphere of the same radius.
        molecule = sg.Molecule(["1H"], [[0, 0, 0]])
        self.assertTrue(np.allclose(
            molecule.semi_axes, molecule.hydrodynamic_radius
        ))

        # Create a diatomic molecule with the bond along the z axis.
        molecule = sg.Molecule(["13C", "13C"], [[0, 0, 0], [0, 0, 1.5]])
        semi_axes = molecule.semi_axes
        axes = molecule.principal_axes

        # The longest semi-axis lies along the bond, with the smallest moment.
        self.assertTrue(semi_axes[0] > semi_axes[1])
        self.assertTrue(abs(semi_axes[1] - semi_axes[2])/semi_axes[1] < 0.01)
        self.assertTrue(abs(abs(axes[2, 0]) - 1) < 1e-6)

        # The ellipsoid has the van der Waals volume of the molecule.
        V_ellipsoid = 4/3*np.pi*np.prod(semi_axes)
        V = molecule.volume
        self.assertTrue(abs(V_ellipsoid - V)/V < 1e-10)
