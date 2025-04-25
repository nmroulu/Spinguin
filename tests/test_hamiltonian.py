import unittest
import numpy as np
import os
from spinguin.system.spin_system import SpinSystem
from spinguin.qm.hamiltonian import hamiltonian
from spinguin.config import Settings
from scipy.sparse import load_npz

class TestHamiltonian(unittest.TestCase):
    """
    Unit test for the Hamiltonian generation functionality.
    """

    def test_hamiltonian(self):
        """
        Test the Hamiltonian generation against a previously calculated result.
        """

        # Simulation settings
        Settings.magnetic_field = 7e-3  # Magnetic field strength in Tesla

        # Define isotopes
        isotopes_c = np.array(['1H', '1H', '1H', '1H', '1H', '1H', '1H', '14N'])

        # Define chemical shifts (in ppm)
        chemical_shifts_c = np.array([-22.7, -22.7, 8.34, 8.34, 7.12, 7.12, 7.77, 43.60])

        # Define scalar couplings (in Hz)
        J_couplings_c = np.array([\
            [ 0,     0,      0,      0,      0,      0,      0,     0],
            [-6.53,  0,      0,      0,      0,      0,      0,     0],
            [ 0.00,  1.66,   0,      0,      0,      0,      0,     0],
            [ 1.40,  0.00,  -0.06,   0,      0,      0,      0,     0],					
            [-0.09,	 0.35,	 6.03,	 0.14,	 0,      0,      0,     0],					
            [ 0.38, -0.13,	 0.09,	 5.93,	 0.06, 	 0,      0,     0],			
            [ 0.01,	 0.03,	 1.12,	-0.02,	 7.75,  -0.01, 	 0,     0],
            [-0.30,  15.91,  4.47,   0.04,   1.79,   0,     -0.46,  0]
        ])

        # Maximum spin order
        max_so = 3  

        # Initialize the spin system
        spin_system_c = SpinSystem(isotopes_c, chemical_shifts_c, J_couplings_c, max_spin_order=max_so)

        # Load the previously calculated Hamiltonian for comparison
        test_dir = os.path.dirname(__file__)
        H_c_previous = load_npz(os.path.join(test_dir, 'test_data', 'hamiltonian.npz'))

        # Generate the Hamiltonian
        H_c = hamiltonian(spin_system_c)

        # Assert that the generated Hamiltonian matches the reference
        self.assertTrue(np.allclose(H_c_previous.toarray(), H_c.toarray()))