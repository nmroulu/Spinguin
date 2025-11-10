import unittest
import numpy as np
import spinguin as sg

class TestSpinguin(unittest.TestCase):

    def test_create_operator(self):
        """
        A test for creating Hilbert-space operators.
        """
        # Reset parameters to defaults
        sg.parameters.default()

        # Initialize two SpinSystem objects
        ss1 = sg.SpinSystem(['1H', '1H'])
        ss2 = sg.SpinSystem(['1H', '1H', '1H'])

        # Operator to make
        operator = "I(x,0) * I(y,1)"
        
        # Creating an operator to either of these systems raises no errors
        sg.operator(ss1, operator)
        sg.operator(ss2, operator)

    def test_create_superoperator(self):
        """
        A test for creating Liouville-space superoperators.
        """
        # Reset parameters to defaults
        sg.parameters.default()

        # Initialize two SpinSystem objects
        ss1 = sg.SpinSystem(['1H', '1H'])
        ss2 = sg.SpinSystem(['1H', '1H', '1H'])

        # Superoperator to make
        operator = "I(x,0) * I(y,1)"
        
        # Trying to make a superoperator before defining basis causes error
        with self.assertRaises(ValueError):
            sg.superoperator(ss1, operator, "comm")

        # When basis is built, constructing superoperator should work
        ss1.basis.max_spin_order = 2
        ss2.basis.max_spin_order = 2
        ss1.basis.build()
        ss2.basis.build()
        sg.superoperator(ss1, operator, "comm")
        sg.superoperator(ss2, operator, "comm")

    def test_create_hamiltonian(self):
        """
        A test for creating Hamiltonians.
        """
        # Reset parameters to defaults
        sg.parameters.default()

        # Initialize two SpinSystem objects
        ss1 = sg.SpinSystem(['1H', '1H'])

        # Trying to make Hamiltonian before building basis causes error
        with self.assertRaises(ValueError):
            sg.hamiltonian(ss1)

        # Build the basis
        ss1.basis.max_spin_order = 2
        ss1.basis.build()

        # Trying to make Hamiltonian before setting field causes error
        with self.assertRaises(ValueError):
            sg.hamiltonian(ss1)

        # Only J-coupling Hamiltonian should be fine
        sg.hamiltonian(ss1, interactions=["J_coupling"])

        # Set the magnetic field
        sg.parameters.magnetic_field = 1

        # After assigning the field, no error is raised
        sg.hamiltonian(ss1)

    def test_create_relaxation(self):
        """
        Test creating the relaxation superoperator.
        """
        # Reset parameters to defaults
        sg.parameters.default()

        # Initialize two SpinSystem objects
        ss1 = sg.SpinSystem(['1H', '1H'])
        ss2 = sg.SpinSystem(['1H', '1H', '1H'])

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
        sg.relaxation(ss1)
        sg.relaxation(ss2)