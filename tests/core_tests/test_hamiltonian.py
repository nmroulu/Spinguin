"""
Tests for Hamiltonian construction in the Spinguin core.
"""

import unittest

import numpy as np
import spinguin as sg
from ._helpers import build_spin_system, test_data_path
from scipy.sparse import csc_array, load_npz


class TestHamiltonian(unittest.TestCase):

    def test_hamiltonian(self):
        """
        Test the Hamiltonian generation against a previously calculated result.
        """

        # Set the global parameters used in the reference calculation.
        sg.parameters.default()
        sg.parameters.magnetic_field = 7e-3
        
        # Build the test spin system.
        ss = build_spin_system(["1H", "1H", "14N"], 3)

        # Define the chemical shifts in ppm.
        ss.chemical_shifts = [4, 6, 150]

        # Define the scalar couplings in Hz.
        ss.J_couplings = [
            [0,  0,  0],
            [8,  0,  0],
            [2,  3,  0],
        ]

        # Load the previously calculated Hamiltonian for comparison.
        H_previous = csc_array(load_npz(test_data_path("hamiltonian.npz")))
        
        # Generate the sparse Hamiltonian and compare it with the reference.
        sg.parameters.sparse_superoperator = True
        H_comm = sg.hamiltonian(ss)
        self.assertTrue(np.allclose(H_comm.toarray(), H_previous.toarray()))

        # Generate the dense Hamiltonian and compare it with the reference.
        sg.parameters.sparse_superoperator = False
        H_comm = sg.hamiltonian(ss)
        self.assertTrue(np.allclose(H_comm, H_previous.toarray()))

        # Repeat the calculation to check cache reuse and consistency.
        sg.parameters.sparse_superoperator = True
        H_comm = sg.hamiltonian(ss)
        self.assertTrue(np.allclose(H_comm.toarray(), H_previous.toarray()))
        sg.parameters.sparse_superoperator = False
        H_comm = sg.hamiltonian(ss)
        self.assertTrue(np.allclose(H_comm, H_previous.toarray()))

        # Build the left and right actions separately and recombine them.
        sg.parameters.sparse_superoperator = True
        H_left = sg.hamiltonian(ss, side='left')
        H_right = sg.hamiltonian(ss, side='right')
        H_comm = H_left - H_right
        self.assertTrue(np.allclose(H_comm.toarray(), H_previous.toarray()))
        sg.parameters.sparse_superoperator = False
        H_left = sg.hamiltonian(ss, side='left')
        H_right = sg.hamiltonian(ss, side='right')
        H_comm = H_left - H_right
        self.assertTrue(np.allclose(H_comm, H_previous.toarray()))

        # Build each interaction contribution separately and sum them.
        sg.parameters.sparse_superoperator = True
        H_comm_Z = sg.hamiltonian(ss, ["zeeman"])
        H_comm_CS = sg.hamiltonian(ss, ["chemical_shift"])
        H_comm_J = sg.hamiltonian(ss, ["J_coupling"])
        H_comm = H_comm_Z + H_comm_CS + H_comm_J
        self.assertTrue(np.allclose(H_comm.toarray(), H_previous.toarray()))
        sg.parameters.sparse_superoperator = False
        H_comm_Z = sg.hamiltonian(ss, ["zeeman"])
        H_comm_CS = sg.hamiltonian(ss, ["chemical_shift"])
        H_comm_J = sg.hamiltonian(ss, ["J_coupling"])
        H_comm = H_comm_Z + H_comm_CS + H_comm_J
        self.assertTrue(np.allclose(H_comm, H_previous.toarray()))