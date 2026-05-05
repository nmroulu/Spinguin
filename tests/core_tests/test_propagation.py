"""
Tests for state propagation utilities.
"""

import unittest

import numpy as np

import spinguin as sg
from ._helpers import build_spin_system


class TestPropagation(unittest.TestCase):
    """
    Test state propagation.
    """

    def test_pulse(self):
        """
        Test the behavior of pulses on spin states.
        """
        # Reset parameters to defaults
        sg.parameters.default()

        # Build the spin system used in the test.
        ss = build_spin_system(["1H", "1H", "14N"], 2)

        # Create the initial basis states used in the pulse checks.
        rho_x = sg.state(ss, "I(x,0)")
        rho_y = sg.state(ss, "I(y,0)")
        rho_z = sg.state(ss, "I(z,0)")
        rho_xz = sg.state(ss, "I(x,0)*I(z,1)")
        rho_yz = sg.state(ss, "I(y,0)*I(z,1)")

        # Test with dense and sparse backends.
        for sparse in [False, True]:
            sg.parameters.sparse_operator = sparse

            # Create the pulse propagators.
            pul_90_x = sg.pulse(ss, "I(x,0)", angle=90)
            pul_90_y = sg.pulse(ss, "I(y,0)", angle=90)
            pul_90_z = sg.pulse(ss, "I(z,0)", angle=90)
            pul_180_x = sg.pulse(ss, "I(x,0)", angle=180)
            pul_180_y = sg.pulse(ss, "I(y,0)", angle=180)
            pul_180_z = sg.pulse(ss, "I(z,0)", angle=180)
            with self.assertWarns(Warning):
                pul_180_zz = sg.pulse(ss, "I(z,0)*I(z,1)", angle=180)

            # Verify the propagated states.
            self.assertTrue(np.allclose(-rho_y, pul_90_x @ rho_z))
            self.assertTrue(np.allclose(rho_x, pul_90_y @ rho_z))
            self.assertTrue(np.allclose(rho_z, pul_90_z @ rho_z))
            self.assertTrue(np.allclose(-rho_z, pul_180_x @ rho_z))
            self.assertTrue(np.allclose(-rho_z, pul_180_y @ rho_z))
            self.assertTrue(np.allclose(rho_z, pul_180_z @ rho_z))
            self.assertTrue(np.allclose(2 * rho_yz, pul_180_zz @ rho_x))
            self.assertTrue(np.allclose(-rho_x, pul_180_zz @ (2 * rho_yz)))
            self.assertTrue(np.allclose(-2 * rho_yz, pul_180_zz @ (-rho_x)))
            self.assertTrue(np.allclose(rho_x, pul_180_zz @ (-2 * rho_yz)))
            self.assertTrue(np.allclose(-2 * rho_xz, pul_180_zz @ rho_y))
            self.assertTrue(np.allclose(-rho_y, pul_180_zz @ (-2 * rho_xz)))
            self.assertTrue(np.allclose(2 * rho_xz, pul_180_zz @ (-rho_y)))
            self.assertTrue(np.allclose(rho_y, pul_180_zz @ (2 * rho_xz)))