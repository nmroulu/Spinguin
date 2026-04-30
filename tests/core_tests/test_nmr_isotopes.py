import unittest
import numpy as np
from spinguin._core._nmr_isotopes import spin, gamma, quadrupole_moment

import spinguin as sg

class TestNMRIsotopes(unittest.TestCase):

    def test_spin(self):
        """
        Test acquiring the spin quantum number
        """

        # Test against known values
        self.assertAlmostEqual(spin("1H"), 1/2)
        self.assertAlmostEqual(spin("14N"), 1)
        self.assertAlmostEqual(spin("19F"), 1/2)
        self.assertAlmostEqual(spin("23Na"), 3/2)

    def test_gamma(self):
        """
        Test acquiring the gyromagnetic ratio
        """

        # Test against known values
        self.assertTrue(np.allclose(gamma("1H", "Hz"), 42.577478615342585*1e6))
        self.assertTrue(np.allclose(gamma("1H", "rad/s"),
                                    2*np.pi*42.577478615342585*1e6))
        self.assertTrue(np.allclose(gamma("14N", "Hz"), 3.076272817251739*1e6))
        self.assertTrue(np.allclose(gamma("14N", "rad/s"),
                                    2*np.pi*3.076272817251739*1e6))
        self.assertTrue(np.allclose(gamma("19F", "Hz"), 40.06924371705693*1e6))
        self.assertTrue(np.allclose(gamma("19F", "rad/s"),
                                    2*np.pi*40.06924371705693*1e6))
        self.assertTrue(np.allclose(gamma("23Na", "Hz"),
                                    11.268733657034751*1e6))
        self.assertTrue(np.allclose(gamma("23Na", "rad/s"),
                                    2*np.pi*11.268733657034751*1e6))
        
    def test_quadrupole_moment(self):
        """
        Test acquiring the quadrupole moment
        """

        # Test against known values
        self.assertTrue(np.allclose(quadrupole_moment("1H"), 0))
        self.assertTrue(np.allclose(quadrupole_moment("14N"), 20.44e-3*1e-28))
        self.assertTrue(np.allclose(quadrupole_moment("19F"), 0))
        self.assertTrue(np.allclose(quadrupole_moment("23Na"), 0))

    def test_add_isotope(self):
        """
        Test adding an isotope.
        """
        # Define a new isotope.
        isotope = "MY_OWN"
        spin = 1/2
        gyromagnetic_ratio = 1.57
        quadrupole_moment = 0
        atomic_mass = 125.442
        natural_abundance = 99.2
        sg.add_isotope(
            isotope,
            spin,
            gyromagnetic_ratio,
            quadrupole_moment,
            atomic_mass,
            natural_abundance,
        )

        # Test that the defined properties can be requested.
        self.assertEqual(sg.spin(isotope), spin)
        self.assertEqual(sg.gamma(isotope, "Hz"), gyromagnetic_ratio*1e6)
        self.assertEqual(sg.quadrupole_moment(isotope), quadrupole_moment)
        self.assertEqual(sg.atomic_mass(isotope), atomic_mass)
        self.assertEqual(sg.natural_abundance(isotope), natural_abundance)

        # Defining an existing isotope must result in an error
        with self.assertRaises(ValueError):
            sg.add_isotope("1H")