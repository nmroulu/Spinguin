"""
Tests for the tabulated NMR isotope properties.
"""

import unittest

import numpy as np

from spinguin._core._nmr_isotopes import (
    atomic_mass,
    gamma,
    natural_abundance,
    quadrupole_moment,
    spin,
)


class TestNMRIsotopes(unittest.TestCase):
    """
    Test accessors for tabulated isotope properties.
    """

    def _assert_allclose(
        self,
        value,
        reference,
    ):
        """
        Check that a scalar value matches its numerical reference.

        Parameters
        ----------
        value : float
            Tested value.
        reference : float
            Reference value.

        Returns
        -------
        None
            The assertion is evaluated in place.
        """

        # Compare the tested value with the reference value.
        self.assertTrue(np.allclose(value, reference))

    def test_spin(self):
        """
        Test retrieving spin quantum numbers.
        """

        # Compare against tabulated reference values.
        self.assertAlmostEqual(spin("1H"), 1 / 2)
        self.assertAlmostEqual(spin("14N"), 1)
        self.assertAlmostEqual(spin("19F"), 1 / 2)
        self.assertAlmostEqual(spin("23Na"), 3 / 2)

    def test_gamma(self):
        """
        Test retrieving gyromagnetic ratios in both supported units.
        """

        # Compare against tabulated reference values in hertz.
        self._assert_allclose(gamma("1H", "Hz"), 42.577478615342585 * 1e6)
        self._assert_allclose(gamma("14N", "Hz"), 3.076272817251739 * 1e6)
        self._assert_allclose(gamma("19F", "Hz"), 40.06924371705693 * 1e6)
        self._assert_allclose(gamma("23Na", "Hz"), 11.268733657034751 * 1e6)

        # Compare against tabulated reference values in radians per second.
        self._assert_allclose(
            gamma("1H", "rad/s"),
            2 * np.pi * 42.577478615342585 * 1e6,
        )
        self._assert_allclose(
            gamma("14N", "rad/s"),
            2 * np.pi * 3.076272817251739 * 1e6,
        )
        self._assert_allclose(
            gamma("19F", "rad/s"),
            2 * np.pi * 40.06924371705693 * 1e6,
        )
        self._assert_allclose(
            gamma("23Na", "rad/s"),
            2 * np.pi * 11.268733657034751 * 1e6,
        )

    def test_quadrupole_moment(self):
        """
        Test retrieving quadrupole moments.
        """

        # Compare against tabulated reference values.
        self._assert_allclose(quadrupole_moment("1H"), 0)
        self._assert_allclose(quadrupole_moment("14N"), 20.44e-3 * 1e-28)
        self._assert_allclose(quadrupole_moment("19F"), 0)
        self._assert_allclose(quadrupole_moment("23Na"), 0)

    def test_atomic_mass(self):
        """
        Test retrieving atomic masses.
        """

        # Compare against tabulated reference values.
        self._assert_allclose(atomic_mass("1H"), 1.007825)
        self._assert_allclose(atomic_mass("28Si"), 27.976927)
        self._assert_allclose(atomic_mass("40Ar"), 39.962383)

    def test_natural_abundance(self):
        """
        Test retrieving natural abundances.
        """

        # Compare against tabulated reference values.
        self._assert_allclose(natural_abundance("1H"), 99.9885)
        self._assert_allclose(natural_abundance("13C"), 1.07)
        self._assert_allclose(natural_abundance("28Si"), 92.223)
        self._assert_allclose(natural_abundance("239Pu"), 0.0)

    def test_common_non_nmr_active_isotopes(self):
        """
        Test common non-NMR-active isotopes present in the table.
        """

        # Check that common non-NMR-active isotopes have zero-valued properties.
        self.assertAlmostEqual(spin("4He"), 0.0)
        self._assert_allclose(gamma("40Ar", "Hz"), 0.0)
        self._assert_allclose(quadrupole_moment("56Fe"), 0.0)
