import unittest
import numpy as np
from spinguin.system.spinguin import Spinguin

class TestSpinguin(unittest.TestCase):

    def test_create_operator(self):
        """
        A test for creating Hilbert-space operators.
        """

        # Initialize two SpinSystem objects
        sg = Spinguin()
        ss1 = sg.new_spin_system("test1")
        ss2 = sg.new_spin_system("test2")
        ss1.isotopes = ['1H', '1H']
        ss2.isotopes = ['1H', '1H', '1H']

        # Operator to make
        operator = "I(x,0) * I(y,1)"
        
        # Creating an operator to any of these systems raises no errors
        sg.operator("test1", operator)
        sg.operator("test2", operator)
        sg.operator("all", operator)

    def test_create_superoperator(self):
        """
        A test for creating Liouville-space superoperators.
        """

        # Initialize two SpinSystem objects
        sg = Spinguin()
        ss1 = sg.new_spin_system("test1")
        ss2 = sg.new_spin_system("test2")
        ss1.isotopes = ['1H', '1H']
        ss2.isotopes = ['1H', '1H', '1H']

        # Superoperator to make
        operator = "I(x,0) * I(y,1)"
        
        # Trying to make a superoperator before defining basis causes error
        with self.assertRaises(ValueError):
            sg.superoperator("test1", operator, "comm")

        # When basis is built, constructing superoperator should work
        ss1.basis.max_spin_order = 2
        ss2.basis.max_spin_order = 2
        ss1.basis.build()
        ss2.basis.build()
        sg.superoperator("test1", operator, "comm")
        sg.superoperator("test2", operator, "comm")
        sg.superoperator("all", operator, "comm")

    def test_create_hamiltonian(self):
        """
        A test for creating Hamiltonians.
        """

        # Initialize two SpinSystem objects
        sg = Spinguin()
        ss1 = sg.new_spin_system("test1")
        ss2 = sg.new_spin_system("test2")
        ss1.isotopes = ['1H', '1H']
        ss2.isotopes = ['1H', '1H', '1H']

        # Trying to make Hamiltonian before building basis causes error
        with self.assertRaises(ValueError):
            sg.hamiltonian("test1", interactions="all")

        # Build the basis
        ss1.basis.max_spin_order = 2
        ss2.basis.max_spin_order = 2
        ss1.basis.build()
        ss2.basis.build()

        # Trying to make Zeeman, chemical shift, or "all" Hamiltonian before
        # setting field causes error
        with self.assertRaises(ValueError):
            sg.hamiltonian("test1", interactions="zeeman")
        with self.assertRaises(ValueError):
            sg.hamiltonian("test1", interactions="chemical_shift")
        with self.assertRaises(ValueError):
            sg.hamiltonian("test1", interactions="all")

        # Set the magnetic field
        sg.parameters.magnetic_field = 1

        # Trying to make chemical shift or "all" Hamiltonian before setting
        # chemical shifts causes error
        with self.assertRaises(ValueError):
            sg.hamiltonian("test1", interactions="chemical_shift")
        with self.assertRaises(ValueError):
            sg.hamiltonian("test1", interactions="all")

        # Set the chemical shifts
        ss1.chemical_shifts = [6.00, 7.00]
        ss2.chemical_shifts = [6.00, 7.00, 8.00]

        # Trying to make J-coupling or "all" Hamiltonian before setting
        # J-couplings causes error
        with self.assertRaises(ValueError):
            sg.hamiltonian("test1", interactions="J_coupling")
        with self.assertRaises(ValueError):
            sg.hamiltonian("test1", interactions="all")

        # Set the J-couplings
        ss1.J_couplings = [
            [0, 0],
            [1, 0]
        ]
        ss2.J_couplings = [
            [0, 0, 0],
            [1, 0, 0],
            [2, 1, 0]
        ]

        # After all parameters are set, no errors should be raised
        sg.hamiltonian("all", interactions="all")
        sg.hamiltonian("all", interactions="zeeman")
        sg.hamiltonian("all", interactions="chemical_shift")
        sg.hamiltonian("all", interactions="J_coupling")

    def test_create_relaxation(self):
        """
        Test creating the relaxation superoperator.
        """

        # Initialize two SpinSystem objects
        sg = Spinguin()
        ss1 = sg.new_spin_system("test1")
        ss2 = sg.new_spin_system("test2")
        ss1.isotopes = ['1H', '1H']
        ss2.isotopes = ['1H', '1H', '1H']

        # Build the basis sets
        ss1.basis.max_spin_order = 2
        ss2.basis.max_spin_order = 2
        ss1.basis.build()
        ss2.basis.build()

        # Set the magnetic field
        sg.parameters.magnetic_field = 1

        # Define chemical shifts
        ss1.chemical_shifts = [1, 2]
        ss2.chemical_shifts = [1, 2, 3]

        # Define J-couplings
        ss1.J_couplings = [
            [0, 0],
            [1, 0]
        ]
        ss2.J_couplings = [
            [0, 0, 0],
            [1, 0, 0],
            [2, 1, 0]
        ]

        # Define XYZ coordinates (only for first system)
        ss1.xyz = [
            [1.0153, 0.9882, 0.0000],
            [1.9812, 2.1002, 0.0521]
        ]

        # Define correlation time (only for first system)
        ss1.relaxation.tau_c = 6e-12

        # Define T1 and T2 times (only for second system)
        ss2.relaxation.T1 = [2, 5, 10]
        ss2.relaxation.T2 = [1, 2.5, 5]

        # Set the relaxation theory
        ss1.relaxation.theory = "redfield"
        ss2.relaxation.theory = "phenomenological"

        # Create a relaxation superoperator for both systems separately
        # and simultaneously
        R1 = sg.relaxation("test1")
        R2 = sg.relaxation("test2")
        Rall = sg.relaxation("all")

        # Extract the separate parts
        Rall1 = Rall[:R1.shape[0], :R1.shape[0]]
        Rall2 = Rall[R1.shape[0]:, R1.shape[0]:]

        # Compare
        self.assertTrue(np.allclose(R1.toarray(), Rall1.toarray()))
        self.assertTrue(np.allclose(R2.toarray(), Rall2.toarray()))