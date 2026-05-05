"""
Tests for the global Spinguin parameter container.
"""

import unittest

import spinguin as sg


class TestParameters(unittest.TestCase):
    """
    Test setting and resetting global package parameters.
    """

    def _set_and_check_parameter(self, parameter: str, value: object) -> None:
        """
        Set one parameter and verify that the assigned value is stored.

        Parameters
        ----------
        parameter_name : str
            Name of the parameter attribute.
        value : object
            Value assigned to the parameter.
        """

        # Reset all parameters before the individual assignment test.
        sg.parameters.default()

        # Assign the test value to the selected parameter.
        setattr(sg.parameters, parameter, value)

        # Verify that the assigned value is stored unchanged.
        self.assertEqual(getattr(sg.parameters, parameter), value)

    def test_set_magnetic_field(self):
        """
        Test setting the magnetic field parameter.
        """

        # Set and verify the magnetic field parameter.
        self._set_and_check_parameter("magnetic_field", 1)

    def test_set_temperature(self):
        """
        Test setting the temperature parameter.
        """

        # Set and verify the temperature parameter.
        self._set_and_check_parameter("temperature", 293)

    def test_set_sparse_operator(self):
        """
        Test setting the sparsity for operators.
        """

        # Set and verify the operator sparsity parameter.
        self._set_and_check_parameter("sparse_operator", False)

    def test_set_sparse_superoperator(self):
        """
        Test setting the sparsity for superoperators.
        """

        # Set and verify the superoperator sparsity parameter.
        self._set_and_check_parameter("sparse_superoperator", False)

    def test_set_propagator_density(self):
        """
        Test setting the threshold for propagator density.
        """

        # Set and verify the propagator density threshold.
        self._set_and_check_parameter("propagator_density", 1.0)

    def test_set_sparse_state(self):
        """
        Test setting the sparsity for state vectors.
        """

        # Set and verify the state sparsity parameter.
        self._set_and_check_parameter("sparse_state", True)

    def test_set_verbose(self):
        """
        Test setting the status messages.
        """

        # Set and verify the verbosity parameter.
        self._set_and_check_parameter("verbose", False)

    def test_set_zero_hamiltonian(self):
        """
        Test setting the zero-value threshold for Hamiltonian.
        """

        # Set and verify the Hamiltonian zero threshold.
        self._set_and_check_parameter("zero_hamiltonian", 1e-6)

    def test_set_zero_aux(self):
        """
        Test setting the zero-value threshold for auxiliary matrix method.
        """

        # Set and verify the auxiliary-matrix zero threshold.
        self._set_and_check_parameter("zero_aux", 1e-6)

    def test_set_zero_relaxation(self):
        """
        Test setting the zero-value threshold for relaxation superoperator.
        """

        # Set and verify the relaxation zero threshold.
        self._set_and_check_parameter("zero_relaxation", 1e-6)

    def test_set_zero_interaction(self):
        """
        Test setting the zero-value threshold for interaction tensors.
        """

        # Set and verify the interaction zero threshold.
        self._set_and_check_parameter("zero_interaction", 1e-6)

    def test_set_zero_propagator(self):
        """
        Test setting the zero-value threshold for propagator.
        """

        # Set and verify the propagator zero threshold.
        self._set_and_check_parameter("zero_propagator", 1e-6)

    def test_set_zero_pulse(self):
        """
        Test setting the zero-value threshold for pulse.
        """

        # Set and verify the pulse zero threshold.
        self._set_and_check_parameter("zero_pulse", 1e-6)

    def test_set_zero_thermalization(self):
        """
        Test setting the zero-value threshold for thermalization.
        """

        # Set and verify the thermalisation zero threshold.
        self._set_and_check_parameter("zero_thermalization", 1e-6)

    def test_set_zero_equilibrium(self):
        """
        Test setting the zero-value threshold for equilibrium state.
        """

        # Set and verify the equilibrium zero threshold.
        self._set_and_check_parameter("zero_equilibrium", 1e-6)

    def test_set_parallel_dim(self):
        """
        Test setting the parallel dimension for calculations.
        """

        # Set and verify the parallelisation threshold.
        self._set_and_check_parameter("parallel_dim", 500)

    def test_zero_time_step(self):
        """
        Test setting the zero-value for one time step.
        """

        # Set and verify the time-step zero threshold.
        self._set_and_check_parameter("zero_time_step", 1e-10)

    def test_zero_zte(self):
        """
        Test setting the zero-value for ZTE.
        """

        # Set and verify the ZTE zero threshold.
        self._set_and_check_parameter("zero_zte", 1e-10)

    def test_default(self):
        """
        Test that resetting to defaults works.
        """

        # Reset to defaults before perturbing the parameters.
        sg.parameters.default()

        # Change each parameter away from its default value.
        sg.parameters.magnetic_field = "changed"
        sg.parameters.temperature = "changed"
        sg.parameters.parallel_dim = "changed"
        sg.parameters.propagator_density = "changed"
        sg.parameters.sparse_operator = "changed"
        sg.parameters.sparse_state = "changed"
        sg.parameters.sparse_superoperator = "changed"
        sg.parameters.verbose = "changed"
        sg.parameters.zero_aux = "changed"
        sg.parameters.zero_equilibrium = "changed"
        sg.parameters.zero_hamiltonian = "changed"
        sg.parameters.zero_interaction = "changed"
        sg.parameters.zero_propagator = "changed"
        sg.parameters.zero_pulse = "changed"
        sg.parameters.zero_relaxation = "changed"
        sg.parameters.zero_thermalization = "changed"
        sg.parameters.zero_time_step = "changed"
        sg.parameters.zero_zte = "changed"
        sg.parameters.nsteps_zte = "changed"

        # Reset the parameter container to its default values.
        sg.parameters.default()

        # Check that the documented default values have been restored.
        self.assertEqual(sg.parameters.magnetic_field, None)
        self.assertEqual(sg.parameters.temperature, None)
        self.assertEqual(sg.parameters.parallel_dim, 1000)
        self.assertEqual(sg.parameters.propagator_density, 0.5)
        self.assertEqual(sg.parameters.sparse_operator, True)
        self.assertEqual(sg.parameters.sparse_state, False)
        self.assertEqual(sg.parameters.sparse_superoperator, True)
        self.assertEqual(sg.parameters.verbose, True)
        self.assertEqual(sg.parameters.zero_aux, 1e-15)
        self.assertEqual(sg.parameters.zero_equilibrium, 1e-18)
        self.assertEqual(sg.parameters.zero_hamiltonian, 1e-12)
        self.assertEqual(sg.parameters.zero_interaction, 1e-9)
        self.assertEqual(sg.parameters.zero_propagator, 1e-18)
        self.assertEqual(sg.parameters.zero_pulse, 1e-18)
        self.assertEqual(sg.parameters.zero_relaxation, 1e-12)
        self.assertEqual(sg.parameters.zero_thermalization, 1e-18)
        self.assertEqual(sg.parameters.zero_time_step, 1e-18)
        self.assertEqual(sg.parameters.zero_zte, 1e-33)
        self.assertEqual(sg.parameters.nsteps_zte, 10)