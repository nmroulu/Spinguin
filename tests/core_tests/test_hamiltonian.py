import unittest
import numpy as np
import os
from scipy.sparse import load_npz, csc_array
import spinguin as sg

class TestHamiltonian(unittest.TestCase):

    # TODO: Tests for dense: too memory-consuming
    def test_hamiltonian(self):
        """
        Test the Hamiltonian generation against a previously calculated result.
        """
        # Set the global parameters
        sg.parameters.default()
        sg.parameters.magnetic_field = 7e-3
        
        # Make the spin system
        ss = sg.SpinSystem(['1H', '1H', '1H', '1H', '1H', '1H', '1H', '14N'])

        # Construct the basis set
        ss.basis.max_spin_order = 3
        ss.basis.build()

        # Define chemical shifts (in ppm)
        ss.chemical_shifts = [-22.7, -22.7, 8.34, 8.34, 7.12, 7.12, 7.77, 43.60]

        # Define scalar couplings (in Hz)
        ss.J_couplings = [
            [ 0,     0,      0,      0,      0,      0,      0,     0],
            [-6.53,  0,      0,      0,      0,      0,      0,     0],
            [ 0.00,  1.66,   0,      0,      0,      0,      0,     0],
            [ 1.40,  0.00,  -0.06,   0,      0,      0,      0,     0],					
            [-0.09,	 0.35,	 6.03,	 0.14,	 0,      0,      0,     0],					
            [ 0.38, -0.13,	 0.09,	 5.93,	 0.06, 	 0,      0,     0],			
            [ 0.01,	 0.03,	 1.12,	-0.02,	 7.75,  -0.01, 	 0,     0],
            [-0.30,  15.91,  4.47,   0.04,   1.79,   0,     -0.46,  0]
        ]

        # Load the previously calculated Hamiltonian for comparison
        test_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'test_data'
        )
        H_previous = csc_array(
            load_npz(os.path.join(test_dir, 'hamiltonian.npz'))
        )
        
        # Generate the Hamiltonian using sparse and dense formalisms and compare
        sg.parameters.sparse_hamiltonian = True
        H_comm = sg.hamiltonian(ss)
        self.assertTrue(np.allclose(H_comm.toarray(), H_previous.toarray()))
        # sg.parameters.sparse_hamiltonian = False
        # H_comm = sg.hamiltonian(ss)
        # self.assertTrue(np.allclose(H_comm, H_previous.toarray()))

        # Perform the same test again (check for cache errors etc.)
        sg.parameters.sparse_hamiltonian = True
        H_comm = sg.hamiltonian(ss)
        self.assertTrue(np.allclose(H_comm.toarray(), H_previous.toarray()))
        # sg.parameters.sparse_hamiltonian = False
        # H_comm = sg.hamiltonian(ss)
        # self.assertTrue(np.allclose(H_comm, H_previous.toarray()))

        # Perform the same test again but by building left and right separately
        sg.parameters.sparse_hamiltonian = True
        H_left = sg.hamiltonian(ss, side='left')
        H_right = sg.hamiltonian(ss, side='right')
        H_comm = H_left - H_right
        self.assertTrue(np.allclose(H_comm.toarray(), H_previous.toarray()))
        # sg.parameters.sparse_hamiltonian = False
        # H_left = sg.hamiltonian(ss, side='left')
        # H_right = sg.hamiltonian(ss, side='right')
        # H_comm = H_left - H_right
        # self.assertTrue(np.allclose(H_comm, H_previous.toarray()))

        # Perform the same test again but build each interaction separately
        sg.parameters.sparse_hamiltonian = True
        H_comm_Z = sg.hamiltonian(ss, ["zeeman"])
        H_comm_CS = sg.hamiltonian(ss, ["chemical_shift"])
        H_comm_J = sg.hamiltonian(ss, ["J_coupling"])
        H_comm = H_comm_Z + H_comm_CS + H_comm_J
        self.assertTrue(np.allclose(H_comm.toarray(), H_previous.toarray()))
        # sg.parameters.sparse_hamiltonian = False
        # H_comm_Z = sg.hamiltonian(ss, ["zeeman"])
        # H_comm_CS = sg.hamiltonian(ss, ["chemical_shift"])
        # H_comm_J = sg.hamiltonian(ss, ["J_coupling"])
        # H_comm = H_comm_Z + H_comm_CS + H_comm_J
        # self.assertTrue(np.allclose(H_comm, H_previous.toarray()))