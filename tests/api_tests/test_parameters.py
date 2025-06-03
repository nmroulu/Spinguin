import unittest
import numpy as np
import os
from spinguin.system.spin_system import SpinSystem
from spinguin.system.parameters import Parameters

class TestParameters(unittest.TestCase):

    def test_set_magnetic_field(self):
        """
        Test setting the magnetic field parameter.
        """

        # Initialize the Parameters object
        parameters = Parameters()

        # Set the magnetic field
        magnetic_field = 1
        parameters.magnetic_field = magnetic_field
        self.assertEqual(parameters.magnetic_field, magnetic_field)

    def test_set_temperature(self):
        """
        Test setting the temperature parameter.
        """

        # Initialize the Parameters object
        parameters = Parameters()

        # Set the temperature
        temperature = 293
        parameters.temperature = temperature
        self.assertEqual(parameters.temperature, temperature)

    def test_set_sparse_operator(self):
        """
        Test setting the sparsity for operators.
        """

        # Initialize the Parameters object
        parameters = Parameters()

        # Set the sparsity for operator
        sparse_operator = False
        parameters.sparse_operator = sparse_operator
        self.assertEqual(parameters.sparse_operator, sparse_operator)

    def test_set_sparse_superoperator(self):
        """
        Test setting the sparsity for superoperators.
        """

        # Initialize the Parameters object
        parameters = Parameters()

        # Set the sparsity for superoperator
        sparse_superoperator = False
        parameters.sparse_superoperator = sparse_superoperator
        self.assertEqual(parameters.sparse_superoperator, sparse_superoperator)

    def test_set_sparse_hamiltonian(self):
        """
        Test setting the sparsity for Hamiltonians.
        """

        # Initialize the Parameters object
        parameters = Parameters()

        # Set the sparsity for Hamiltonian
        sparse_hamiltonian = False
        parameters.sparse_hamiltonian = sparse_hamiltonian
        self.assertEqual(parameters.sparse_hamiltonian, sparse_hamiltonian)

    def test_set_sparse_relaxation(self):
        """
        Test setting the sparsity for relaxation superoperator.
        """

        # Initialize the Parameters object
        parameters = Parameters()

        # Set the sparsity for relaxation superoperator
        sparse_relaxation = False
        parameters.sparse_relaxation = sparse_relaxation
        self.assertEqual(parameters.sparse_relaxation, sparse_relaxation)

    def test_set_propagator_density(self):
        """
        Test setting the threshold for propagator density.
        """

        # Initialize the Parameters object
        parameters = Parameters()

        # Set the density threshold for propagator
        propagator_density = 1.0
        parameters.propagator_density = propagator_density
        self.assertEqual(parameters.propagator_density, propagator_density)

    def test_set_sparse_state(self):
        """
        Test setting the sparsity for state vectors.
        """

        # Initialize the Parameters object
        parameters = Parameters()

        # Set the sparsity for state vectors
        sparse_state = True
        parameters.sparse_state = sparse_state
        self.assertEqual(parameters.sparse_state, sparse_state)

    def test_set_zero_hamiltonian(self):
        """
        Test setting the zero-value threshold for Hamiltonian.
        """

        # Initialize the Parameters object
        parameters = Parameters()

        # Set the zero-value threshold for Hamiltonian
        zero_hamiltonian = 1e-6
        parameters.zero_hamiltonian = zero_hamiltonian
        self.assertEqual(parameters.zero_hamiltonian, zero_hamiltonian)

    def test_set_zero_aux(self):
        """
        Test setting the zero-value threshold for auxiliary matrix method.
        """

        # Initialize the Parameters object
        parameters = Parameters()

        # Set the zero-value threshold for auxiliary matrix method
        zero_aux = 1e-6
        parameters.zero_aux = zero_aux
        self.assertEqual(parameters.zero_aux, zero_aux)

    def test_set_zero_relaxation(self):
        """
        Test setting the zero-value threshold for relaxation superoperator.
        """

        # Initialize the Parameters object
        parameters = Parameters()

        # Set the zero-value threshold for relaxation superoperator
        zero_relaxation = 1e-6
        parameters.zero_relaxation = zero_relaxation
        self.assertEqual(parameters.zero_relaxation, zero_relaxation)

    def test_set_zero_interaction(self):
        """
        Test setting the zero-value threshold for interaction tensors.
        """

        # Initialize the Parameters object
        parameters = Parameters()

        # Set the zero-value threshold for interaction tensors
        zero_interaction = 1e-6
        parameters.zero_interaction = zero_interaction
        self.assertEqual(parameters.zero_interaction, zero_interaction)

    def test_set_zero_propagator(self):
        """
        Test setting the zero-value threshold for propagator.
        """

        # Initialize the Parameters object
        parameters = Parameters()

        # Set the zero-value threshold for propagator
        zero_propagator = 1e-6
        parameters.zero_propagator = zero_propagator
        self.assertEqual(parameters.zero_propagator, zero_propagator)

    def test_set_zero_pulse(self):
        """
        Test setting the zero-value threshold for pulse.
        """

        # Initialize the Parameters object
        parameters = Parameters()

        # Set the zero-value threshold for pulse
        zero_pulse = 1e-6
        parameters.zero_pulse = zero_pulse
        self.assertEqual(parameters.zero_pulse, zero_pulse)

    def test_set_zero_thermalization(self):
        """
        Test setting the zero-value threshold for thermalization.
        """

        # Initialize the Parameters object
        parameters = Parameters()

        # Set the zero-value threshold for thermalization
        zero_thermalization = 1e-6
        parameters.zero_thermalization = zero_thermalization
        self.assertEqual(parameters.zero_thermalization, zero_thermalization)

    def test_set_zero_equilibrium(self):
        """
        Test setting the zero-value threshold for equilibrium state.
        """

        # Initialize the Parameters object
        parameters = Parameters()

        # Set the zero-value threshold for equilibrium state
        zero_equilibrium = 1e-6
        parameters.zero_equilibrium = zero_equilibrium
        self.assertEqual(parameters.zero_equilibrium, zero_equilibrium)
