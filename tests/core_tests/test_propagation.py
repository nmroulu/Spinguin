import unittest
import numpy as np
import spinguin as sg

class TestPropagation(unittest.TestCase):

    def test_pulse(self):
        """
        Test the behavior of pulses on spin states.
        """
        # Reset parameters to defaults
        sg.parameters.default()

        # Define the spin system
        ss = sg.SpinSystem(["1H", "1H", "14N"])

        # Create the basis set
        ss.basis.max_spin_order = 2
        ss.basis.build()

        # Create initial states
        rho_x = sg.state(ss, "I(x,0)")
        rho_y = sg.state(ss, "I(y,0)")
        rho_z = sg.state(ss, "I(z,0)")
        rho_xz = sg.state(ss, "I(x,0)*I(z,1)")
        rho_yz = sg.state(ss, "I(y,0)*I(z,1)")

        # Create pulses in dense format
        sg.parameters.sparse_superoperator = False
        pul_90_x_dense = sg.pulse(ss, "I(x,0)", angle=90)
        pul_90_y_dense = sg.pulse(ss, "I(y,0)", angle=90)
        pul_90_z_dense = sg.pulse(ss, "I(z,0)", angle=90)
        pul_180_x_dense = sg.pulse(ss, "I(x,0)", angle=180)
        pul_180_y_dense = sg.pulse(ss, "I(y,0)", angle=180)
        pul_180_z_dense = sg.pulse(ss, "I(z,0)", angle=180)
        with self.assertWarns(Warning):
            pul_180_zz_dense = sg.pulse(ss, "I(z,0)*I(z,1)", angle=180)

        # Create pulses in sparse format
        sg.parameters.sparse_superoperator = True
        pul_90_x_sparse = sg.pulse(ss, "I(x,0)", angle=90)
        pul_90_y_sparse = sg.pulse(ss, "I(y,0)", angle=90)
        pul_90_z_sparse = sg.pulse(ss, "I(z,0)", angle=90)
        pul_180_x_sparse = sg.pulse(ss, "I(x,0)", angle=180)
        pul_180_y_sparse = sg.pulse(ss, "I(y,0)", angle=180)
        pul_180_z_sparse = sg.pulse(ss, "I(z,0)", angle=180)
        with self.assertWarns(Warning):
            pul_180_zz_sparse = sg.pulse(ss, "I(z,0)*I(z,1)", angle=180)

        # Verify the results (dense)
        self.assertTrue(np.allclose(-rho_y, pul_90_x_dense @ rho_z))
        self.assertTrue(np.allclose(rho_x, pul_90_y_dense @ rho_z))
        self.assertTrue(np.allclose(rho_z, pul_90_z_dense @ rho_z))
        self.assertTrue(np.allclose(-rho_z, pul_180_x_dense @ rho_z))
        self.assertTrue(np.allclose(-rho_z, pul_180_y_dense @ rho_z))
        self.assertTrue(np.allclose(rho_z, pul_180_z_dense @ rho_z))
        self.assertTrue(np.allclose(2 * rho_yz, pul_180_zz_dense @ rho_x))
        self.assertTrue(np.allclose(-rho_x, pul_180_zz_dense @ (2 * rho_yz)))
        self.assertTrue(np.allclose(-2 * rho_yz, pul_180_zz_dense @ (-rho_x)))
        self.assertTrue(np.allclose(rho_x, pul_180_zz_dense @ (-2 * rho_yz)))
        self.assertTrue(np.allclose(-2 * rho_xz, pul_180_zz_dense @ rho_y))
        self.assertTrue(np.allclose(-rho_y, pul_180_zz_dense @ (-2 * rho_xz)))
        self.assertTrue(np.allclose(2 * rho_xz, pul_180_zz_dense @ (-rho_y)))
        self.assertTrue(np.allclose(rho_y, pul_180_zz_dense @ (2 * rho_xz)))

        # Verify the results (sparse)
        self.assertTrue(np.allclose(-rho_y, pul_90_x_sparse @ rho_z))
        self.assertTrue(np.allclose(rho_x, pul_90_y_sparse @ rho_z))
        self.assertTrue(np.allclose(rho_z, pul_90_z_sparse @ rho_z))
        self.assertTrue(np.allclose(-rho_z, pul_180_x_sparse @ rho_z))
        self.assertTrue(np.allclose(-rho_z, pul_180_y_sparse @ rho_z))
        self.assertTrue(np.allclose(rho_z, pul_180_z_sparse @ rho_z))
        self.assertTrue(np.allclose(2 * rho_yz, pul_180_zz_sparse @ rho_x))
        self.assertTrue(np.allclose(-rho_x, pul_180_zz_sparse @ (2 * rho_yz)))
        self.assertTrue(np.allclose(-2 * rho_yz, pul_180_zz_sparse @ (-rho_x)))
        self.assertTrue(np.allclose(rho_x, pul_180_zz_sparse @ (-2 * rho_yz)))
        self.assertTrue(np.allclose(-2 * rho_xz, pul_180_zz_sparse @ rho_y))
        self.assertTrue(np.allclose(-rho_y, pul_180_zz_sparse @ (-2 * rho_xz)))
        self.assertTrue(np.allclose(2 * rho_xz, pul_180_zz_sparse @ (-rho_y)))
        self.assertTrue(np.allclose(rho_y, pul_180_zz_sparse @ (2 * rho_xz)))