"""
This module provides the Config class which contains global settings for the
Spinguin package. For example, user can choose to use dense or sparse arrays as
backend, or to adjust numerical accuracy. The Config object is instantiated
when the Spinguin package is imported and can be accessed by::

    import spinguin as sg
    sg.config.CONFIGNAME = VALUE
"""
class Config:
    """
    Config class contains all the global settings for the Spinguin package.
    """
    # Parallelisation settings
    _parallel_dim: int=1000

    # Sparsity settings
    _propagator_density: float=0.5
    _sparse_hamiltonian: bool=True
    _sparse_operator: bool=True
    _sparse_pulse: bool=True
    _sparse_relaxation: bool=True
    _sparse_state: bool=False
    _sparse_superoperator: bool=True
    
    # Zero-value thresholds
    _zero_aux: float = 1e-18
    _zero_aux_rotframe: float = 1e-18
    _zero_equilibrium: float = 1e-18
    _zero_hamiltonian: float = 1e-12
    _zero_interaction: float = 1e-9
    _zero_propagator: float = 1e-18
    _zero_pulse: float = 1e-18
    _zero_relaxation: float = 1e-12
    _zero_thermalization: float = 1e-18
    _zero_time_step: float = 1e-18
    _zero_zte: float = 1e-24

    @property
    def parallel_dim(self) -> int:
        """
        If the number of items in the basis is larger than this value,
        parallelization is used to speed up the calculation of the Redfield
        relaxation superoperator.
        """
        return self._parallel_dim
    
    @parallel_dim.setter
    def parallel_dim(self, parallel_dim: int):
        self._parallel_dim = parallel_dim
        print(f"Threshold for parallel Redfield set to: {self.parallel_dim}\n")

    @property
    def sparse_operator(self) -> bool:
        """
        Specifies whether to return the Hilbert-space operators as sparse or
        dense arrays.
        """
        return self._sparse_operator
    
    @sparse_operator.setter
    def sparse_operator(self, sparse_operator: bool):
        self._sparse_operator = sparse_operator
        print(f"Sparity setting of operator set to: {self.sparse_operator}\n")

    @property
    def sparse_pulse(self) -> bool:
        """
        Specifies whether to return pulse superoperators as sparse or dense
        arrays.
        """
        return self._sparse_pulse
    
    @sparse_pulse.setter
    def sparse_pulse(self, sparse_pulse: bool):
        self._sparse_pulse = sparse_pulse
        print(f"Sparsity setting of pulses set to: {self.sparse_pulse}\n")

    @property
    def sparse_superoperator(self) -> bool:
        """
        Specifies whether to return the superoperators as sparse or dense
        arrays.
        """
        return self._sparse_superoperator
    
    @sparse_superoperator.setter
    def sparse_superoperator(self, sparse_superoperator: bool):
        self._sparse_superoperator = sparse_superoperator
        print("Sparity setting of superoperator set to: "
              f"{self.sparse_superoperator}\n")

    @property
    def sparse_hamiltonian(self) -> bool:
        """
        Specifies whether to return the Hamiltonian as sparse or dense array.
        """
        return self._sparse_hamiltonian
    
    @sparse_hamiltonian.setter
    def sparse_hamiltonian(self, sparse_hamiltonian: bool):
        self._sparse_hamiltonian = sparse_hamiltonian
        print("Sparity setting of Hamiltonian set to: "
              f"{self.sparse_hamiltonian}\n")

    @property
    def sparse_relaxation(self) -> bool:
        """
        Specifies whether to return the relaxation superoperator as sparse or
        dense array.
        """
        return self._sparse_relaxation
    
    @sparse_relaxation.setter
    def sparse_relaxation(self, sparse_relaxation: bool):
        self._sparse_relaxation = sparse_relaxation
        print("Sparity setting of relaxation set to: "
              f"{self.sparse_relaxation}\n")

    @property
    def sparse_state(self) -> bool:
        """
        Specifies whether to return the state vectors as sparse or dense arrays.
        """
        return self._sparse_state
    
    @sparse_state.setter
    def sparse_state(self, sparse_state: bool):
        self._sparse_state = sparse_state
        print(f"Sparity setting of state set to: {self.sparse_state}\n")

    @property
    def propagator_density(self) -> float:
        """
        Threshold (between 0 and 1) that specifies the array type of the time
        propagator. If the density of the progagator is greater than this value,
        dense array is returned. Otherwise, sparse array is returned.
        """
        return self._propagator_density
    
    @propagator_density.setter
    def propagator_density(self, propagator_density: float):
        self._propagator_density = propagator_density
        print("Propagator density threshold set to: "
              f"{self.propagator_density}\n")

    @property
    def zero_hamiltonian(self) -> float:
        """
        Threshold under which a value is considered to be zero in Hamiltonian.
        """
        return self._zero_hamiltonian
    
    @zero_hamiltonian.setter
    def zero_hamiltonian(self, zero_hamiltonian: float):
        self._zero_hamiltonian = zero_hamiltonian
        print("Hamiltonian zero-value threshold set to: "
              f"{self.zero_hamiltonian}\n")

    @property
    def zero_aux(self) -> float:
        """
        Threshold under which a value is considered to be zero in auxiliary
        matrix method which is used in the Redfield relaxation theory.
        """
        return self._zero_aux
    
    @zero_aux.setter
    def zero_aux(self, zero_aux: float):
        self._zero_aux = zero_aux
        print(f"Auxiliary zero-value threshold set to: {self.zero_aux}\n")

    @property
    def zero_relaxation(self) -> float:
        """
        Threshold under which a value is considered to be zero in relaxation
        superoperator.
        """
        return self._zero_relaxation
    
    @zero_relaxation.setter
    def zero_relaxation(self, zero_relaxation: float):
        self._zero_relaxation = zero_relaxation
        print("Relaxation zero-value threshold set to: "
              f"{self.zero_relaxation}\n")

    @property
    def zero_interaction(self) -> float:
        """
        If the 1-norm of an interaction tensor (upper bound for its eigenvalues)
        is smaller than this threshold, the interaction is ignored when
        constructing the Redfield relaxation superoperator.
        """
        return self._zero_interaction
    
    @zero_interaction.setter
    def zero_interaction(self, zero_interaction: float):
        self._zero_interaction = zero_interaction
        print("Interaction zero-value threshold set to: "
              f"{self.zero_interaction}\n")

    @property
    def zero_propagator(self) -> float:
        """
        Threshold under which a value is considered to be zero in propagator.
        This value is used to as the convergence criterion of the Taylor series
        that is used to compute the matrix exponential.
        """
        return self._zero_propagator
    
    @zero_propagator.setter
    def zero_propagator(self, zero_propagator: float):
        self._zero_propagator = zero_propagator
        print("Propagator zero-value threshold set to: "
              f"{self.zero_propagator}\n")

    @property
    def zero_pulse(self) -> float:
        """
        Threshold under which a value is considered to be zero in pulse. This
        value is used as the convergence criterion of the Taylor series that is
        used to compute the matrix exponential.
        """
        return self._zero_pulse
    
    @zero_pulse.setter
    def zero_pulse(self, zero_pulse: float):
        self._zero_pulse = zero_pulse
        print(f"Pulse zero-value threshold set to: {self.zero_pulse}\n")

    @property
    def zero_thermalization(self) -> float:
        """
        Threshold under which a value is considered to be zero when performing
        the matrix exponential while applying the Levitt-di Bari thermalization.
        This value is used as the convergence criterion of the Taylor series.
        """
        return self._zero_thermalization
    
    @zero_thermalization.setter
    def zero_thermalization(self, zero_thermalization: float):
        self._zero_thermalization = zero_thermalization
        print("Thermalization zero-value threshold set to: "
              f"{self.zero_thermalization}\n")

    @property
    def zero_equilibrium(self) -> float:
        """
        Threshold under which a value is considered to be zero when performing
        the matrix exponential while constructing the equilibrium state.
        """
        return self._zero_equilibrium
    
    @zero_equilibrium.setter
    def zero_equilibrium(self, zero_equilibrium: float):
        self._zero_equilibrium = zero_equilibrium
        print("Equilibrium zero-value threshold set to: "
              f"{self.zero_equilibrium}\n")
        
    @property
    def zero_time_step(self) -> float:
        """
        Threshold under which a value is considered to be zero when advancing
        the state vector forward for one time step.
        """
        return self._zero_time_step
    
    @zero_time_step.setter
    def zero_time_step(self, zero_time_step: float):
        self._zero_time_step = zero_time_step
        print("Time step zero-value threshold set to: "
              f"{self.zero_time_step}\n")
        
    @property
    def zero_zte(self) -> float:
        """
        Threshold under which a value is considered to be zero when performing
        the zero-track elimination (ZTE) basis truncation.
        """
        return self._zero_zte
    
    @zero_zte.setter
    def zero_zte(self, zero_zte: float):
        self._zero_zte = zero_zte
        print(f"ZTE zero-value threshold set to: {self.zero_zte}\n")

    @property
    def zero_aux_rotframe(self) -> float:
        """
        Zero-value threshold for the auxiliary matrix method used in the
        rotating frame transformation.
        """
        return self._zero_aux_rotframe
    
    @zero_aux_rotframe.setter
    def zero_aux_rotframe(self, zero_aux_rotframe: float):
        self._zero_aux_rotframe = zero_aux_rotframe
        print(f"Rotating frame auxiliary zero-value threshold set to: "
              f"{self.zero_aux_rotframe}\n")
        
# Instantiate the Config object
config = Config()