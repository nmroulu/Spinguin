"""
config.py

This module provides the Config class which contains all the global settings
for the Spinguin package.
"""

class Config:
    """
    Config class contains all the global settings for the Spinguin package.

    Attributes
    ----------
    custom_dot : bool, default=False
        Specifies whether to use custom matrix product implementation in the
        time propagators. This implementation is parallelized and more memory-
        friendly than the default SciPy implementation, but suffers from worse 
        single-core performance.
    propagator_density : float, default=0.5
        Threshold that specifies when to use dense or sparse arrays for the
        propagators.
    sparse_hamiltonian : bool, default=True
        Specifies whether to use sparse or dense arrays for Hamiltonian.
    sparse_operator : bool, default=True
        Specifies whether to use sparse or dense arrays for operators.
    sparse_pulse : bool, default=True
        Specifies whether to use sparse or dense arrays for pulses.
    sparse_relaxation : bool, default=True
        Specifies whether to use sparse or dense arrays for relaxation
        superoperator.
    sparse_state : bool, default=False
        Specifies whether to use sparse or dense arrays for states.
    sparse_superoperator : bool, default=True
        Specifies whether to use sparse or dense arrays for superoperators.
    zero_aux : float, default=1e-18
        Threshold under which a value is considered to be zero in auxiliary
        matrix method which is used in the Redfield relaxation theory.
    zero_equilibrium : float, default=1e-18
        Threshold under which a value is considered to be zero when performing
        the matrix exponential while constructing the equilibrium state.
    zero_hamiltonian : float, default=1e-12
        Threshold under which a value is considered to be zero in Hamiltonian.
    zero_interaction : float, default=1e-9
        If the 1-norm of an interaction tensor (upper bound for its eigenvalues)
        is smaller than this threshold, the interaction is ignored when
        constructing the Redfield relaxation superoperator.
    zero_propagator : float, default=1e-18
        Threshold under which a value is considered to be zero in propagator.
    zero_pulse : float, default=1e-18
        Threshold under which a value is considered to be zero in pulse.
    zero_relaxation : float, default=1e-12
        Threshold under which a value is considered to be zero in relaxation
        superoperator.
    zero_thermalization : float, default=1e-18
        Threshold under which a value is considered to be zero when performing
        the matrix exponential while applying the Levitt-di Bari thermalization.
    """

    # Matrix product settings
    _custom_dot: bool=False

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
    _zero_equilibrium: float = 1e-18
    _zero_hamiltonian: float = 1e-12
    _zero_interaction: float = 1e-9
    _zero_propagator: float = 1e-18
    _zero_pulse: float = 1e-18
    _zero_relaxation: float = 1e-12
    _zero_thermalization: float = 1e-18

    @property
    def custom_dot(self) -> bool:
        return self._custom_dot
    
    @custom_dot.setter
    def custom_dot(self, custom_dot: bool):
        """
        Specifies whether to use custom matrix product implementation in the
        time propagators. This implementation is parallelized and more memory-
        friendly than the default SciPy implementation, but suffers from worse 
        single-core performance.
        """
        self._custom_dot = custom_dot
        print(f"Custom dot product setting set to: {self.custom_dot}\n")

    @property
    def sparse_operator(self) -> bool:
        return self._sparse_operator
    
    @sparse_operator.setter
    def sparse_operator(self, sparse_operator: bool):
        """
        Specifies whether to use sparse or dense arrays for operators.
        """
        self._sparse_operator = sparse_operator
        print(f"Sparity setting of operator set to: {self.sparse_operator}\n")

    @property
    def sparse_pulse(self) -> bool:
        return self._sparse_pulse
    
    @sparse_pulse.setter
    def sparse_pulse(self, sparse_pulse: bool):
        """
        Specifies whether to use sparse or dense arrays for pulse
        superoperarors.
        """
        self._sparse_pulse = sparse_pulse
        print(f"Sparsity setting of pulses set to: {self.pulse}\n")

    @property
    def sparse_superoperator(self) -> bool:
        return self._sparse_superoperator
    
    @sparse_superoperator.setter
    def sparse_superoperator(self, sparse_superoperator: bool):
        """
        Specifies whether to use sparse or dense arrays for superoperators.
        """
        self._sparse_superoperator = sparse_superoperator
        print("Sparity setting of superoperator set to: "
              f"{self.sparse_superoperator}\n")

    @property
    def sparse_hamiltonian(self) -> bool:
        return self._sparse_hamiltonian
    
    @sparse_hamiltonian.setter
    def sparse_hamiltonian(self, sparse_hamiltonian: bool):
        """
        Specifies whether to use sparse or dense arrays for Hamiltonian.
        """
        self._sparse_hamiltonian = sparse_hamiltonian
        print("Sparity setting of Hamiltonian set to: "
              f"{self.sparse_hamiltonian}\n")

    @property
    def sparse_relaxation(self) -> bool:
        return self._sparse_relaxation
    
    @sparse_relaxation.setter
    def sparse_relaxation(self, sparse_relaxation: bool):
        """
        Specifies whether to use sparse or dense arrays for relaxation
        superoperator.
        """
        self._sparse_relaxation = sparse_relaxation
        print("Sparity setting of relaxation set to: "
              f"{self.sparse_relaxation}\n")

    @property
    def sparse_state(self) -> bool:
        return self._sparse_state
    
    @sparse_state.setter
    def sparse_state(self, sparse_state: bool):
        """
        Specifies whether to use sparse or dense arrays for states.
        """
        self._sparse_state = sparse_state
        print(f"Sparity setting of state set to: {self.sparse_state}\n")

    @property
    def propagator_density(self) -> float:
        return self._propagator_density
    
    @propagator_density.setter
    def propagator_density(self, propagator_density: float):
        """
        Threshold that specifies when to use dense or sparse arrays for the
        propagators.
        """
        self._propagator_density = propagator_density
        print("Propagator density threshold set to: "
              f"{self.propagator_density}\n")

    @property
    def zero_hamiltonian(self) -> float:
        return self._zero_hamiltonian
    
    @zero_hamiltonian.setter
    def zero_hamiltonian(self, zero_hamiltonian: float):
        """
        Threshold under which a value is considered to be zero in Hamiltonian.
        """
        self._zero_hamiltonian = zero_hamiltonian
        print("Hamiltonian zero-value threshold set to: "
              f"{self.zero_hamiltonian}\n")

    @property
    def zero_aux(self) -> float:
        return self._zero_aux
    
    @zero_aux.setter
    def zero_aux(self, zero_aux: float):
        """
        Threshold under which a value is considered to be zero in auxiliary
        matrix method which is used in the Redfield relaxation theory.
        """
        self._zero_aux = zero_aux
        print(f"Auxiliary zero-value threshold set to: {self.zero_aux}\n")

    @property
    def zero_relaxation(self) -> float:
        return self._zero_relaxation
    
    @zero_relaxation.setter
    def zero_relaxation(self, zero_relaxation: float):
        """
        Threshold under which a value is considered to be zero in relaxation
        superoperator.
        """
        self._zero_relaxation = zero_relaxation
        print("Relaxation zero-value threshold set to: "
              f"{self.zero_relaxation}\n")

    @property
    def zero_interaction(self) -> float:
        return self._zero_interaction
    
    @zero_interaction.setter
    def zero_interaction(self, zero_interaction: float):
        """
        If the 1-norm of an interaction tensor (upper bound for its eigenvalues)
        is smaller than this threshold, the interaction is ignored when
        constructing the Redfield relaxation superoperator.
        """
        self._zero_interaction = zero_interaction
        print("Interaction zero-value threshold set to: "
              f"{self.zero_interaction}\n")

    @property
    def zero_propagator(self) -> float:
        return self._zero_propagator
    
    @zero_propagator.setter
    def zero_propagator(self, zero_propagator: float):
        """
        Threshold under which a value is considered to be zero in propagator.
        """
        self._zero_propagator = zero_propagator
        print("Propagator zero-value threshold set to: "
              f"{self.zero_propagator}\n")

    @property
    def zero_pulse(self) -> float:
        return self._zero_pulse
    
    @zero_pulse.setter
    def zero_pulse(self, zero_pulse: float):
        """
        Threshold under which a value is considered to be zero in pulse.
        """
        self._zero_pulse = zero_pulse
        print(f"Pulse zero-value threshold set to: {self.zero_pulse}\n")

    @property
    def zero_thermalization(self) -> float:
        return self._zero_thermalization
    
    @zero_thermalization.setter
    def zero_thermalization(self, zero_thermalization: float):
        """
        Threshold under which a value is considered to be zero when performing
        the matrix exponential while applying the Levitt-di Bari thermalization.
        """
        self._zero_thermalization = zero_thermalization
        print("Thermalization zero-value threshold set to: "
              f"{self.zero_thermalization}\n")

    @property
    def zero_equilibrium(self) -> float:
        return self._zero_equilibrium
    
    @zero_equilibrium.setter
    def zero_equilibrium(self, zero_equilibrium: float):
        """
        Threshold under which a value is considered to be zero when performing
        the matrix exponential while constructing the equilibrium state.
        """
        self._zero_equilibrium = zero_equilibrium
        print("Equilibrium zero-value threshold set to: "
              f"{self.zero_equilibrium}\n")

# Instantiate the Config object
config = Config()