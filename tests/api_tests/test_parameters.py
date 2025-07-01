import unittest
import spinguin as sg

class TestParameters(unittest.TestCase):

    def test_set_magnetic_field(self):
        """
        Test setting the magnetic field parameter.
        """
        # Set the magnetic field
        magnetic_field = 1
        sg.parameters.magnetic_field = magnetic_field
        self.assertEqual(sg.parameters.magnetic_field, magnetic_field)

    def test_set_temperature(self):
        """
        Test setting the temperature parameter.
        """
        # Set the temperature
        temperature = 293
        sg.parameters.temperature = temperature
        self.assertEqual(sg.parameters.temperature, temperature)

    def test_set_sparse_operator(self):
        """
        Test setting the sparsity for operators.
        """
        # Set the sparsity for operator
        sparse_operator = False
        sg.parameters.sparse_operator = sparse_operator
        self.assertEqual(sg.parameters.sparse_operator, sparse_operator)

    def test_set_sparse_superoperator(self):
        """
        Test setting the sparsity for superoperators.
        """

        # Set the sparsity for superoperator
        sparse_superoperator = False
        sg.parameters.sparse_superoperator = sparse_superoperator
        self.assertEqual(sg.parameters.sparse_superoperator, sparse_superoperator)

    def test_set_sparse_hamiltonian(self):
        """
        Test setting the sparsity for Hamiltonians.
        """
        # Set the sparsity for Hamiltonian
        sparse_hamiltonian = False
        sg.parameters.sparse_hamiltonian = sparse_hamiltonian
        self.assertEqual(sg.parameters.sparse_hamiltonian, sparse_hamiltonian)

    def test_set_sparse_relaxation(self):
        """
        Test setting the sparsity for relaxation superoperator.
        """
        # Set the sparsity for relaxation superoperator
        sparse_relaxation = False
        sg.parameters.sparse_relaxation = sparse_relaxation
        self.assertEqual(sg.parameters.sparse_relaxation, sparse_relaxation)

    def test_set_propagator_density(self):
        """
        Test setting the threshold for propagator density.
        """
        # Set the density threshold for propagator
        propagator_density = 1.0
        sg.parameters.propagator_density = propagator_density
        self.assertEqual(sg.parameters.propagator_density, propagator_density)

    def test_set_sparse_state(self):
        """
        Test setting the sparsity for state vectors.
        """
        # Set the sparsity for state vectors
        sparse_state = True
        sg.parameters.sparse_state = sparse_state
        self.assertEqual(sg.parameters.sparse_state, sparse_state)

    def test_set_zero_hamiltonian(self):
        """
        Test setting the zero-value threshold for Hamiltonian.
        """
        # Set the zero-value threshold for Hamiltonian
        zero_hamiltonian = 1e-6
        sg.parameters.zero_hamiltonian = zero_hamiltonian
        self.assertEqual(sg.parameters.zero_hamiltonian, zero_hamiltonian)

    def test_set_zero_aux(self):
        """
        Test setting the zero-value threshold for auxiliary matrix method.
        """
        # Set the zero-value threshold for auxiliary matrix method
        zero_aux = 1e-6
        sg.parameters.zero_aux = zero_aux
        self.assertEqual(sg.parameters.zero_aux, zero_aux)

    def test_set_zero_relaxation(self):
        """
        Test setting the zero-value threshold for relaxation superoperator.
        """
        # Set the zero-value threshold for relaxation superoperator
        zero_relaxation = 1e-6
        sg.parameters.zero_relaxation = zero_relaxation
        self.assertEqual(sg.parameters.zero_relaxation, zero_relaxation)

    def test_set_zero_interaction(self):
        """
        Test setting the zero-value threshold for interaction tensors.
        """
        # Set the zero-value threshold for interaction tensors
        zero_interaction = 1e-6
        sg.parameters.zero_interaction = zero_interaction
        self.assertEqual(sg.parameters.zero_interaction, zero_interaction)

    def test_set_zero_propagator(self):
        """
        Test setting the zero-value threshold for propagator.
        """
        # Set the zero-value threshold for propagator
        zero_propagator = 1e-6
        sg.parameters.zero_propagator = zero_propagator
        self.assertEqual(sg.parameters.zero_propagator, zero_propagator)

    def test_set_zero_pulse(self):
        """
        Test setting the zero-value threshold for pulse.
        """
        # Set the zero-value threshold for pulse
        zero_pulse = 1e-6
        sg.parameters.zero_pulse = zero_pulse
        self.assertEqual(sg.parameters.zero_pulse, zero_pulse)

    def test_set_zero_thermalization(self):
        """
        Test setting the zero-value threshold for thermalization.
        """
        # Set the zero-value threshold for thermalization
        zero_thermalization = 1e-6
        sg.parameters.zero_thermalization = zero_thermalization
        self.assertEqual(sg.parameters.zero_thermalization, zero_thermalization)

    def test_set_zero_equilibrium(self):
        """
        Test setting the zero-value threshold for equilibrium state.
        """
        # Set the zero-value threshold for equilibrium state
        zero_equilibrium = 1e-6
        sg.parameters.zero_equilibrium = zero_equilibrium
        self.assertEqual(sg.parameters.zero_equilibrium, zero_equilibrium)

    def test_set_parallel_dim(self):
        """
        Test setting the parallel dimension for calculations.
        """
        # Set the parallel dimension
        parallel_dim = 500
        sg.parameters.parallel_dim = parallel_dim
        self.assertEqual(sg.parameters.parallel_dim, parallel_dim)

    def test_set_sparse_pulse(self):
        """
        Test setting the sparsity for pulse.
        """
        # Set the sparsity for pulse
        sparse_pulse = False
        sg.parameters.sparse_pulse = sparse_pulse
        self.assertEqual(sg.parameters.sparse_pulse, sparse_pulse)
