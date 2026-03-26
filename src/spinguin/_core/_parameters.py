"""
Global parameter definitions for the Spinguin package.

The module provides the `Parameters` class that stores the package-wide
settings used throughout spin-dynamics simulations. A single instance is
created when Spinguin is imported and may be modified as follows::

    import spinguin as sg
    sg.parameters.PARAMETERNAME = VALUE
"""

from spinguin._core._status import status


class Parameters:
    """
    Store the global settings used throughout the Spinguin package.

    Usage: ``Parameters()``.
    """

    def __init__(
        self,
    ) -> None:
        """
        Initialise the parameter container with the default settings.

        Usage: ``Parameters()``.

        Returns
        -------
        None
            The instance is initialised in place.
        """

        # Populate the object with the default parameter values.
        self.default()

    def _report_update(
        self,
        message: str,
    ) -> None:
        """
        Print a status message for a parameter update.

        Usage: ``self._report_update(message)``.

        Parameters
        ----------
        message : str
            Status message to be printed.

        Returns
        -------
        None
            The message is forwarded to the package status printer.
        """

        # Forward the update message through the standard status channel.
        status(message)

    def default(
        self,
    ) -> None:
        """
        Reset all parameters to their default values.

        Usage: ``parameters.default()``.

        Returns
        -------
        None
            All parameter values are reset in place.
        """

        # Set the default experimental conditions.
        self._magnetic_field: float = None
        self._temperature: float = None

        # Set the default parallelisation threshold.
        self._parallel_dim: int = 1000

        # Set the default rotating-frame expansion order.
        self._rotating_frame_order: int = 5

        # Set the default sparsity controls.
        self._propagator_density: float = 0.5
        self._sparse_operator: bool = True
        self._sparse_state: bool = False
        self._sparse_superoperator: bool = True

        # Enable status messages by default.
        self._verbose: bool = True

        # Set the default numerical zero-value thresholds.
        self._zero_aux: float = 1e-15
        self._zero_equilibrium: float = 1e-18
        self._zero_hamiltonian: float = 1e-12
        self._zero_interaction: float = 1e-9
        self._zero_propagator: float = 1e-18
        self._zero_pulse: float = 1e-18
        self._zero_relaxation: float = 1e-12
        self._zero_thermalization: float = 1e-18
        self._zero_time_step: float = 1e-18
        self._zero_zte: float = 1e-33

        # Set the default zero-track elimination controls.
        self._nsteps_zte: int = 10

    @property
    def magnetic_field(
        self,
    ) -> float:
        """
        Return the external magnetic field in tesla.
        """

        return self._magnetic_field

    @magnetic_field.setter
    def magnetic_field(
        self,
        magnetic_field: float,
    ) -> None:
        # Store the new magnetic-field value.
        self._magnetic_field = magnetic_field

        # Report the updated magnetic-field value.
        self._report_update(
            f"Magnetic field set to: {self.magnetic_field} T\n"
        )

    @property
    def temperature(
        self,
    ) -> float:
        """
        Return the sample temperature in kelvin.
        """

        return self._temperature

    @temperature.setter
    def temperature(
        self,
        temperature: float,
    ) -> None:
        # Store the new temperature value.
        self._temperature = temperature

        # Report the updated temperature value.
        self._report_update(f"Temperature set to: {self.temperature} K\n")

    @property
    def parallel_dim(
        self,
    ) -> int:
        """
        Return the basis-size threshold for parallel Redfield calculations.

        If the number of basis elements exceeds this value, parallel execution is
        used when constructing the Redfield relaxation superoperator.
        """

        return self._parallel_dim

    @parallel_dim.setter
    def parallel_dim(
        self,
        parallel_dim: int,
    ) -> None:
        # Store the new parallelisation threshold.
        self._parallel_dim = parallel_dim

        # Report the updated parallelisation threshold.
        self._report_update(
            f"Threshold for parallel Redfield set to: {self.parallel_dim}\n"
        )

    @property
    def rotating_frame_order(
        self,
    ) -> int:
        """
        Return the Taylor-expansion order used in the rotating frame.

        Higher orders give more accurate rotating-frame transformations but
        require more computation time. The default value is 5.
        """

        return self._rotating_frame_order

    @rotating_frame_order.setter
    def rotating_frame_order(
        self,
        rotating_frame_order: int,
    ) -> None:
        # Store the new rotating-frame expansion order.
        self._rotating_frame_order = rotating_frame_order

        # Report the updated rotating-frame expansion order.
        self._report_update(
            f"Rotating frame order set to: {self.rotating_frame_order}\n"
        )

    @property
    def sparse_operator(
        self,
    ) -> bool:
        """
        Return whether Hilbert-space operators are stored sparsely.
        """

        return self._sparse_operator

    @sparse_operator.setter
    def sparse_operator(
        self,
        sparse_operator: bool,
    ) -> None:
        # Store the new Hilbert-space operator sparsity setting.
        self._sparse_operator = sparse_operator

        # Report the updated Hilbert-space operator sparsity setting.
        self._report_update(
            f"Sparsity setting of operator set to: {self.sparse_operator}\n"
        )

    @property
    def sparse_superoperator(
        self,
    ) -> bool:
        """
        Return whether superoperators are stored sparsely.
        """

        return self._sparse_superoperator

    @sparse_superoperator.setter
    def sparse_superoperator(
        self,
        sparse_superoperator: bool,
    ) -> None:
        # Store the new superoperator sparsity setting.
        self._sparse_superoperator = sparse_superoperator

        # Report the updated superoperator sparsity setting.
        self._report_update(
            "Sparsity setting of superoperator set to: "
            f"{self.sparse_superoperator}\n"
        )

    @property
    def sparse_state(
        self,
    ) -> bool:
        """
        Return whether state vectors are stored sparsely.
        """

        return self._sparse_state

    @sparse_state.setter
    def sparse_state(
        self,
        sparse_state: bool,
    ) -> None:
        # Store the new state-vector sparsity setting.
        self._sparse_state = sparse_state

        # Report the updated state-vector sparsity setting.
        self._report_update(
            f"Sparsity setting of state set to: {self.sparse_state}\n"
        )

    @property
    def propagator_density(
        self,
    ) -> float:
        """
        Return the density threshold used for propagator storage selection.

        If the density of the propagator exceeds this value, a dense array is
        returned; otherwise, a sparse array is returned.
        """

        return self._propagator_density

    @propagator_density.setter
    def propagator_density(
        self,
        propagator_density: float,
    ) -> None:
        # Store the new propagator-density threshold.
        self._propagator_density = propagator_density

        # Report the updated propagator-density threshold.
        self._report_update(
            "Propagator density threshold set to: "
            f"{self.propagator_density}\n"
        )

    @property
    def verbose(
        self,
    ) -> bool:
        """
        Return whether status messages are printed to the console.
        """

        return self._verbose

    @verbose.setter
    def verbose(
        self,
        verbose: bool,
    ) -> None:
        # Store the new verbosity setting.
        self._verbose = verbose

        # Report the updated verbosity setting unconditionally.
        print(f"Verbose set to: {self.verbose}\n")

    @property
    def zero_hamiltonian(
        self,
    ) -> float:
        """
        Return the zero-value threshold used for Hamiltonians.
        """

        return self._zero_hamiltonian

    @zero_hamiltonian.setter
    def zero_hamiltonian(
        self,
        zero_hamiltonian: float,
    ) -> None:
        # Store the new Hamiltonian zero-value threshold.
        self._zero_hamiltonian = zero_hamiltonian

        # Report the updated Hamiltonian zero-value threshold.
        self._report_update(
            "Hamiltonian zero-value threshold set to: "
            f"{self.zero_hamiltonian}\n"
        )

    @property
    def zero_aux(
        self,
    ) -> float:
        """
        Return the zero-value threshold used in the auxiliary-matrix method.

        This threshold is used in the Redfield relaxation treatment.
        """

        return self._zero_aux

    @zero_aux.setter
    def zero_aux(
        self,
        zero_aux: float,
    ) -> None:
        # Store the new auxiliary-method zero-value threshold.
        self._zero_aux = zero_aux

        # Report the updated auxiliary-method zero-value threshold.
        self._report_update(
            f"Auxiliary zero-value threshold set to: {self.zero_aux}\n"
        )

    @property
    def zero_relaxation(
        self,
    ) -> float:
        """
        Return the zero-value threshold used for relaxation superoperators.
        """

        return self._zero_relaxation

    @zero_relaxation.setter
    def zero_relaxation(
        self,
        zero_relaxation: float,
    ) -> None:
        # Store the new relaxation zero-value threshold.
        self._zero_relaxation = zero_relaxation

        # Report the updated relaxation zero-value threshold.
        self._report_update(
            "Relaxation zero-value threshold set to: "
            f"{self.zero_relaxation}\n"
        )

    @property
    def zero_interaction(
        self,
    ) -> float:
        """
        Return the threshold below which interaction tensors are ignored.

        If the matrix 1-norm of an interaction tensor, which is an upper bound
        for its eigenvalues, is smaller than this threshold, the interaction is
        neglected when constructing the Redfield relaxation superoperator.
        """

        return self._zero_interaction

    @zero_interaction.setter
    def zero_interaction(
        self,
        zero_interaction: float,
    ) -> None:
        # Store the new interaction zero-value threshold.
        self._zero_interaction = zero_interaction

        # Report the updated interaction zero-value threshold.
        self._report_update(
            "Interaction zero-value threshold set to: "
            f"{self.zero_interaction}\n"
        )

    @property
    def zero_propagator(
        self,
    ) -> float:
        """
        Return the zero-value threshold used in propagator calculations.

        This threshold is used as the convergence criterion for the Taylor
        series employed to evaluate the matrix exponential.
        """

        return self._zero_propagator

    @zero_propagator.setter
    def zero_propagator(
        self,
        zero_propagator: float,
    ) -> None:
        # Store the new propagator zero-value threshold.
        self._zero_propagator = zero_propagator

        # Report the updated propagator zero-value threshold.
        self._report_update(
            "Propagator zero-value threshold set to: "
            f"{self.zero_propagator}\n"
        )

    @property
    def zero_pulse(
        self,
    ) -> float:
        """
        Return the zero-value threshold used in pulse calculations.

        This threshold is used as the convergence criterion for the Taylor
        series employed to evaluate the matrix exponential.
        """

        return self._zero_pulse

    @zero_pulse.setter
    def zero_pulse(
        self,
        zero_pulse: float,
    ) -> None:
        # Store the new pulse zero-value threshold.
        self._zero_pulse = zero_pulse

        # Report the updated pulse zero-value threshold.
        self._report_update(
            f"Pulse zero-value threshold set to: {self.zero_pulse}\n"
        )

    @property
    def zero_thermalization(
        self,
    ) -> float:
        """
        Return the zero-value threshold used in thermalization calculations.

        This threshold is used as the convergence criterion for the Taylor
        series employed during Levitt-di Bari thermalization.
        """

        return self._zero_thermalization

    @zero_thermalization.setter
    def zero_thermalization(
        self,
        zero_thermalization: float,
    ) -> None:
        # Store the new thermalization zero-value threshold.
        self._zero_thermalization = zero_thermalization

        # Report the updated thermalization zero-value threshold.
        self._report_update(
            "Thermalization zero-value threshold set to: "
            f"{self.zero_thermalization}\n"
        )

    @property
    def zero_equilibrium(
        self,
    ) -> float:
        """
        Return the zero-value threshold used for equilibrium-state construction.
        """

        return self._zero_equilibrium

    @zero_equilibrium.setter
    def zero_equilibrium(
        self,
        zero_equilibrium: float,
    ) -> None:
        # Store the new equilibrium zero-value threshold.
        self._zero_equilibrium = zero_equilibrium

        # Report the updated equilibrium zero-value threshold.
        self._report_update(
            "Equilibrium zero-value threshold set to: "
            f"{self.zero_equilibrium}\n"
        )

    @property
    def zero_time_step(
        self,
    ) -> float:
        """
        Return the zero-value threshold used in single-step propagation.
        """

        return self._zero_time_step

    @zero_time_step.setter
    def zero_time_step(
        self,
        zero_time_step: float,
    ) -> None:
        # Store the new time-step zero-value threshold.
        self._zero_time_step = zero_time_step

        # Report the updated time-step zero-value threshold.
        self._report_update(
            "Time step zero-value threshold set to: "
            f"{self.zero_time_step}\n"
        )

    @property
    def zero_zte(
        self,
    ) -> float:
        """
        Return the zero-value threshold used in zero-track elimination.

        The default value, ``1e-33``, is intended to eliminate only basis states
        that remain exactly zero during ZTE basis truncation.
        """

        return self._zero_zte

    @zero_zte.setter
    def zero_zte(
        self,
        zero_zte: float,
    ) -> None:
        # Store the new zero-track-elimination threshold.
        self._zero_zte = zero_zte

        # Report the updated zero-track-elimination threshold.
        self._report_update(
            f"ZTE zero-value threshold set to: {self.zero_zte}\n"
        )

    @property
    def nsteps_zte(
        self,
    ) -> int:
        """
        Return the maximum number of steps used in ZTE basis truncation.

        The default value is 10.
        """

        return self._nsteps_zte

    @nsteps_zte.setter
    def nsteps_zte(
        self,
        nsteps_zte: int,
    ) -> None:
        # Store the new limit for ZTE propagation steps.
        self._nsteps_zte = nsteps_zte

        # Report the updated ZTE propagation-step limit.
        self._report_update(
            f"Maximum number of steps in ZTE set to: {self.nsteps_zte}\n"
        )


# Instantiate the global parameter container.
parameters = Parameters()