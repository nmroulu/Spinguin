import unittest
import numpy as np
import os
from spinguin.qm.hamiltonian import sop_H_coherent
from spinguin.utils.nmr_isotopes import ISOTOPES
from spinguin.qm.basis import make_basis
from scipy.sparse import load_npz

class TestHamiltonian(unittest.TestCase):
    """
    Unit test for the Hamiltonian generation functionality.
    """

    def test_hamiltonian(self):
        """
        Test the Hamiltonian generation against a previously calculated result.
        """
        
        # Make the spin system
        spins = np.array([1/2, 1/2, 1/2, 1/2, 1/2, 1/2, 1/2, 1])
        max_spin_order = 3
        basis = make_basis(spins, max_spin_order)

        # Get the gyromagnetic ratios (in rad/s/T)
        y_1H = 2*np.pi * ISOTOPES['1H'][1] * 1e6
        y_14N = 2*np.pi * ISOTOPES['14N'][1] * 1e6
        gammas = np.array([y_1H, y_1H, y_1H, y_1H, y_1H, y_1H, y_1H, y_14N])

        # Define chemical shifts (in ppm)
        chemical_shifts = np.array([-22.7, -22.7, 8.34, 8.34, 7.12, 7.12, 7.77, 43.60])

        # Define scalar couplings (in Hz)
        J_couplings = np.array([\
            [ 0,     0,      0,      0,      0,      0,      0,     0],
            [-6.53,  0,      0,      0,      0,      0,      0,     0],
            [ 0.00,  1.66,   0,      0,      0,      0,      0,     0],
            [ 1.40,  0.00,  -0.06,   0,      0,      0,      0,     0],					
            [-0.09,	 0.35,	 6.03,	 0.14,	 0,      0,      0,     0],					
            [ 0.38, -0.13,	 0.09,	 5.93,	 0.06, 	 0,      0,     0],			
            [ 0.01,	 0.03,	 1.12,	-0.02,	 7.75,  -0.01, 	 0,     0],
            [-0.30,  15.91,  4.47,   0.04,   1.79,   0,     -0.46,  0]
        ])

        # Set the magnetic field (7 mT)
        B = 7e-3
        
        # Generate the Hamiltonian using sparse format
        H_comm = sop_H_coherent(basis,
                                gammas,
                                spins,
                                chemical_shifts,
                                J_couplings,
                                B,
                                side="comm",
                                sparse=True,
                                zero_value=1e-12)
        H_left = sop_H_coherent(basis,
                                gammas,
                                spins,
                                chemical_shifts,
                                J_couplings,
                                B,
                                side="left",
                                sparse=True,
                                zero_value=1e-12)
        H_right = sop_H_coherent(basis,
                                 gammas,
                                 spins,
                                 chemical_shifts,
                                 J_couplings,
                                 B,
                                 side="right",
                                 sparse=True,
                                 zero_value=1e-12)

        # Load the previously calculated Hamiltonian for comparison
        test_dir = os.path.dirname(__file__)
        H_previous = load_npz(os.path.join(test_dir, 'test_data', 'hamiltonian.npz'))

        # Assert that the generated Hamiltonian matches the reference
        self.assertTrue(np.allclose(H_comm.toarray(), H_previous.toarray()))

        # Assert that the commutation superoperator can be constructed from left and right
        self.assertTrue(np.allclose((H_left - H_right).toarray(), H_comm.toarray()))