import unittest
import numpy as np
from spinguin.qm.basis import make_basis
from spinguin.qm.states import state_from_string
from spinguin.qm.propagation import sop_pulse

class TestPropagation(unittest.TestCase):

    def test_pulses(self):
        """
        Test the behavior of pulses on spin states.
        """

        # Define the spin system
        spins = np.array([1/2, 1/2, 1])
        max_spin_order = 2
        basis = make_basis(spins, max_spin_order)

        # Create initial states in dense format
        rho_x_dense = state_from_string(basis, spins, "I(x,0)", sparse=False)
        rho_y_dense = state_from_string(basis, spins, "I(y,0)", sparse=False)
        rho_z_dense = state_from_string(basis, spins, "I(z,0)", sparse=False)
        rho_xz_dense = state_from_string(basis, spins, "I(x,0)*I(z,1)", sparse=False)
        rho_yz_dense = state_from_string(basis, spins, "I(y,0)*I(z,1)", sparse=False)

        # Create initial states in sparse format
        rho_x_sparse = state_from_string(basis, spins, "I(x,0)", sparse=True)
        rho_y_sparse = state_from_string(basis, spins, "I(y,0)", sparse=True)
        rho_z_sparse = state_from_string(basis, spins, "I(z,0)", sparse=True)
        rho_xz_sparse = state_from_string(basis, spins, "I(x,0)*I(z,1)", sparse=True)
        rho_yz_sparse = state_from_string(basis, spins, "I(y,0)*I(z,1)", sparse=True)

        # Create pulses in dense format
        pul_90_x_dense = sop_pulse(basis, spins, "I(x,0)", angle=90, sparse=False)
        pul_90_y_dense = sop_pulse(basis, spins, "I(y,0)", angle=90, sparse=False)
        pul_90_z_dense = sop_pulse(basis, spins, "I(z,0)", angle=90, sparse=False)
        pul_180_x_dense = sop_pulse(basis, spins, "I(x,0)", angle=180, sparse=False)
        pul_180_y_dense = sop_pulse(basis, spins, "I(y,0)", angle=180, sparse=False)
        pul_180_z_dense = sop_pulse(basis, spins, "I(z,0)", angle=180, sparse=False)
        pul_180_zz_dense = sop_pulse(basis, spins, "I(z,0)*I(z,1)", angle=180, sparse=False)

        # Create pulses in sparse format
        pul_90_x_sparse = sop_pulse(basis, spins, "I(x,0)", angle=90, sparse=True)
        pul_90_y_sparse = sop_pulse(basis, spins, "I(y,0)", angle=90, sparse=True)
        pul_90_z_sparse = sop_pulse(basis, spins, "I(z,0)", angle=90, sparse=True)
        pul_180_x_sparse = sop_pulse(basis, spins, "I(x,0)", angle=180, sparse=True)
        pul_180_y_sparse = sop_pulse(basis, spins, "I(y,0)", angle=180, sparse=True)
        pul_180_z_sparse = sop_pulse(basis, spins, "I(z,0)", angle=180, sparse=True)
        pul_180_zz_sparse = sop_pulse(basis, spins, "I(z,0)*I(z,1)", angle=180, sparse=True)

        # Verify the results (dense@dense)
        self.assertTrue(np.allclose(-rho_y_dense, pul_90_x_dense @ rho_z_dense))
        self.assertTrue(np.allclose(rho_x_dense, pul_90_y_dense @ rho_z_dense))
        self.assertTrue(np.allclose(rho_z_dense, pul_90_z_dense @ rho_z_dense))
        self.assertTrue(np.allclose(-rho_z_dense, pul_180_x_dense @ rho_z_dense))
        self.assertTrue(np.allclose(-rho_z_dense, pul_180_y_dense @ rho_z_dense))
        self.assertTrue(np.allclose(rho_z_dense, pul_180_z_dense @ rho_z_dense))
        self.assertTrue(np.allclose(2 * rho_yz_dense, pul_180_zz_dense @ rho_x_dense))
        self.assertTrue(np.allclose(-rho_x_dense, pul_180_zz_dense @ (2 * rho_yz_dense)))
        self.assertTrue(np.allclose(-2 * rho_yz_dense, pul_180_zz_dense @ (-rho_x_dense)))
        self.assertTrue(np.allclose(rho_x_dense, pul_180_zz_dense @ (-2 * rho_yz_dense)))
        self.assertTrue(np.allclose(-2 * rho_xz_dense, pul_180_zz_dense @ rho_y_dense))
        self.assertTrue(np.allclose(-rho_y_dense, pul_180_zz_dense @ (-2 * rho_xz_dense)))
        self.assertTrue(np.allclose(2 * rho_xz_dense, pul_180_zz_dense @ (-rho_y_dense)))
        self.assertTrue(np.allclose(rho_y_dense, pul_180_zz_dense @ (2 * rho_xz_dense)))

        # Verify the results (sparse@dense)
        self.assertTrue(np.allclose(-rho_y_dense, pul_90_x_sparse @ rho_z_dense))
        self.assertTrue(np.allclose(rho_x_dense, pul_90_y_sparse @ rho_z_dense))
        self.assertTrue(np.allclose(rho_z_dense, pul_90_z_sparse @ rho_z_dense))
        self.assertTrue(np.allclose(-rho_z_dense, pul_180_x_sparse @ rho_z_dense))
        self.assertTrue(np.allclose(-rho_z_dense, pul_180_y_sparse @ rho_z_dense))
        self.assertTrue(np.allclose(rho_z_dense, pul_180_z_sparse @ rho_z_dense))
        self.assertTrue(np.allclose(2 * rho_yz_dense, pul_180_zz_sparse @ rho_x_dense))
        self.assertTrue(np.allclose(-rho_x_dense, pul_180_zz_sparse @ (2 * rho_yz_dense)))
        self.assertTrue(np.allclose(-2 * rho_yz_dense, pul_180_zz_sparse @ (-rho_x_dense)))
        self.assertTrue(np.allclose(rho_x_dense, pul_180_zz_sparse @ (-2 * rho_yz_dense)))
        self.assertTrue(np.allclose(-2 * rho_xz_dense, pul_180_zz_sparse @ rho_y_dense))
        self.assertTrue(np.allclose(-rho_y_dense, pul_180_zz_sparse @ (-2 * rho_xz_dense)))
        self.assertTrue(np.allclose(2 * rho_xz_dense, pul_180_zz_sparse @ (-rho_y_dense)))
        self.assertTrue(np.allclose(rho_y_dense, pul_180_zz_sparse @ (2 * rho_xz_dense)))

        # Verify the results (dense@sparse)
        self.assertTrue(np.allclose(-rho_y_dense, pul_90_x_dense @ rho_z_sparse))
        self.assertTrue(np.allclose(rho_x_dense, pul_90_y_dense @ rho_z_sparse))
        self.assertTrue(np.allclose(rho_z_dense, pul_90_z_dense @ rho_z_sparse))
        self.assertTrue(np.allclose(-rho_z_dense, pul_180_x_dense @ rho_z_sparse))
        self.assertTrue(np.allclose(-rho_z_dense, pul_180_y_dense @ rho_z_sparse))
        self.assertTrue(np.allclose(rho_z_dense, pul_180_z_dense @ rho_z_sparse))
        self.assertTrue(np.allclose(2 * rho_yz_dense, pul_180_zz_dense @ rho_x_sparse))
        self.assertTrue(np.allclose(-rho_x_dense, pul_180_zz_dense @ (2 * rho_yz_sparse)))
        self.assertTrue(np.allclose(-2 * rho_yz_dense, pul_180_zz_dense @ (-rho_x_sparse)))
        self.assertTrue(np.allclose(rho_x_dense, pul_180_zz_dense @ (-2 * rho_yz_sparse)))
        self.assertTrue(np.allclose(-2 * rho_xz_dense, pul_180_zz_dense @ rho_y_sparse))
        self.assertTrue(np.allclose(-rho_y_dense, pul_180_zz_dense @ (-2 * rho_xz_sparse)))
        self.assertTrue(np.allclose(2 * rho_xz_dense, pul_180_zz_dense @ (-rho_y_sparse)))
        self.assertTrue(np.allclose(rho_y_dense, pul_180_zz_dense @ (2 * rho_xz_sparse)))

        # Verify the results (sparse@sparse)
        self.assertTrue(np.allclose(-rho_y_dense, (pul_90_x_sparse @ rho_z_sparse).toarray()))
        self.assertTrue(np.allclose(rho_x_dense, (pul_90_y_sparse @ rho_z_sparse).toarray()))
        self.assertTrue(np.allclose(rho_z_dense, (pul_90_z_sparse @ rho_z_sparse).toarray()))
        self.assertTrue(np.allclose(-rho_z_dense, (pul_180_x_sparse @ rho_z_sparse).toarray()))
        self.assertTrue(np.allclose(-rho_z_dense, (pul_180_y_sparse @ rho_z_sparse).toarray()))
        self.assertTrue(np.allclose(rho_z_dense, (pul_180_z_sparse @ rho_z_sparse).toarray()))
        self.assertTrue(np.allclose(2 * rho_yz_dense, (pul_180_zz_sparse @ rho_x_sparse).toarray()))
        self.assertTrue(np.allclose(-rho_x_dense, (pul_180_zz_sparse @ (2 * rho_yz_sparse)).toarray()))
        self.assertTrue(np.allclose(-2 * rho_yz_dense, (pul_180_zz_sparse @ (-rho_x_sparse)).toarray()))
        self.assertTrue(np.allclose(rho_x_dense, (pul_180_zz_sparse @ (-2 * rho_yz_sparse)).toarray()))
        self.assertTrue(np.allclose(-2 * rho_xz_dense, (pul_180_zz_sparse @ rho_y_sparse).toarray()))
        self.assertTrue(np.allclose(-rho_y_dense, (pul_180_zz_sparse @ (-2 * rho_xz_sparse)).toarray()))
        self.assertTrue(np.allclose(2 * rho_xz_dense, (pul_180_zz_sparse @ (-rho_y_sparse)).toarray()))
        self.assertTrue(np.allclose(rho_y_dense, (pul_180_zz_sparse @ (2 * rho_xz_sparse)).toarray()))