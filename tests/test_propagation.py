import unittest
import numpy as np
from spinguin.system.spin_system import SpinSystem
from spinguin.qm.states import state
from spinguin.qm.propagation import pulse

class TestPropagation(unittest.TestCase):
    """
    Unit tests for spin system propagation using pulses.
    """

    def test_pulses(self):
        """
        Test the behavior of pulses on spin states.
        """

        # Define isotopes
        isotopes = np.array(['1H', '1H', '14N'])

        # Initialize the spin system
        spin_system = SpinSystem(isotopes, max_spin_order=2)

        # Create initial states
        rho_x = state(spin_system, "I(x,0)")
        rho_y = state(spin_system, "I(y,0)")
        rho_z = state(spin_system, "I(z,0)")
        rho_xz = state(spin_system, "I(x,0)*I(z,1)")
        rho_yz = state(spin_system, "I(y,0)*I(z,1)")

        # Create pulses
        pul_90_x = pulse(spin_system, "I(x,0)", angle=90)
        pul_90_y = pulse(spin_system, "I(y,0)", angle=90)
        pul_90_z = pulse(spin_system, "I(z,0)", angle=90)
        pul_180_x = pulse(spin_system, "I(x,0)", angle=180)
        pul_180_y = pulse(spin_system, "I(y,0)", angle=180)
        pul_180_z = pulse(spin_system, "I(z,0)", angle=180)
        pul_180_zz = pulse(spin_system, "I(z,0)*I(z,1)", angle=180)

        # Verify the results
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