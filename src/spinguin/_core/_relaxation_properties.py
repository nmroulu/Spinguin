"""
This module provides a RelaxationProperties class that stores information on the
relaxation theory settings. It is assigned as part of `SpinSystem` upon its
instantiation. The relaxation properties can be accessed as follows::

    import spinguin as sg                       # Import the package     
    spin_system = sg.SpinSystem(["1H"])         # Create an example spin system
    spin_system.relaxation.theory = "redfield"  # Set relaxation theory
"""

# Referencing SpinSystem class
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from spinguin._core._spin_system import SpinSystem

# Imports
import numpy as np
from typing import Literal
from spinguin._core._data_io import read_array
from spinguin._core._la import arraylike_to_array
from spinguin._core._molecule import Molecule

class RelaxationProperties:
    """
    This class stores information on the relaxation properties of a spin system.
    """

    # Relaxation properties
    _antisymmetric: bool=False
    _dynamic_frequency_shift: bool=False
    _relative_error: float = 1e-6
    _sr2k: bool=False
    _tau_c: float | np.ndarray = None
    _theory: Literal["redfield", "phenomenological"] = None
    _thermalization: bool = False
    _T1: np.ndarray = None
    _T2: np.ndarray = None
    _molecule: Molecule = None

    def __init__(self, spin_system: SpinSystem):
        print("Relaxation theory settings have been initialized with the "
              "following defaults: ")
        print(f"antisymmetric: {self.antisymmetric}")
        print(f"dynamic_frequency_shift: {self.dynamic_frequency_shift}")
        print(f"relative_error: {self.relative_error}")
        print(f"sr2k: {self.sr2k}")
        print(f"tau_c: {self.tau_c}")
        print(f"theory: {self.theory}")
        print(f"thermalization: {self.thermalization}")
        print(f"T1: {self.T1}")
        print(f"T2: {self.T2}")
        print()

        # Store a reference to the SpinSystem
        self._spin_system = spin_system

    @property
    def antisymmetric(self) -> bool:
        """
        Specifies whether to consider the antisymmetric part of the interaction
        tensors in the Redfield relaxation theory.
        """
        return self._antisymmetric
    
    @antisymmetric.setter
    def antisymmetric(self, antisymmetric: bool):
        self._antisymmetric = antisymmetric
        print("Antisymmetric part of the interaction tensors set to: "
              f"{self.antisymmetric}\n")

    @property
    def dynamic_frequency_shift(self) -> bool:
        """
        Specifies whether to include the dynamic frequency shift in the Redfield
        relaxation theory. This corresponds to the imaginary part of the
        relaxation superoperator.
        """
        return self._dynamic_frequency_shift
    
    @dynamic_frequency_shift.setter
    def dynamic_frequency_shift(self, dynamic_frequency_shift: bool):
        self._dynamic_frequency_shift = dynamic_frequency_shift
        print("Dynamic frequency shift set to: "
              f"{self.dynamic_frequency_shift}\n")
        
    @property
    def relative_error(self) -> float:
        """
        Specifies the relative error for the Redfield relaxation theory. This
        corresponds to the convergence criterion for the Redfield integral.
        """
        return self._relative_error
    
    @relative_error.setter
    def relative_error(self, relative_error: float):
        self._relative_error = relative_error
        print(f"Relative error set to: {self.relative_error}\n")

    @property
    def sr2k(self) -> bool:
        """
        Specifies whether to include the scalar relaxation of the second kind
        (SR2K) in the relaxation superoperator.
        """
        return self._sr2k
    
    @sr2k.setter
    def sr2k(self, sr2k: bool):
        self._sr2k = sr2k
        print(f"SR2K set to: {self.sr2k}\n")

    @property
    def tau_c(self) -> float | np.ndarray:
        """
        Specifies the correlation time(s) for the Redfield relaxation theory.

        For isotropic rotational diffusion, a single value is used. Example::

            spin_system.relaxation.tau_c = 50e-12

        For symmetric top rotational diffusion, two values are used,
        corresponding to the perpendicular and parallel components of the
        diffusion tensor. Example::
            
            spin_system.relaxation.tau_c = [50e-12, 100e-12]
        
        For anisotropic rotational diffusion, an array of three values is used,
        corresponding to the principal components of the diffusion tensor.
        Example::

            spin_system.relaxation.tau_c = [50e-12, 100e-12, 150e-12]
        """
        return self._tau_c
    
    @tau_c.setter
    def tau_c(self, tau_c: float | list[float] | tuple[float, ...] | np.ndarray):
        if isinstance(tau_c, (float, int)):
            self._tau_c = float(tau_c)
        elif isinstance(tau_c, (list, tuple)) and len(tau_c) == 2:
            self._tau_c = np.array([float(tau_c[0]), float(tau_c[0]), float(tau_c[1])])
        elif isinstance(tau_c, (list, tuple)) and len(tau_c) == 3:
            self._tau_c = np.array([float(x) for x in tau_c])
        elif isinstance(tau_c, np.ndarray) and tau_c.shape == (2,):
            self._tau_c = np.array([tau_c[0], tau_c[0], tau_c[1]], dtype=float)
        elif isinstance(tau_c, np.ndarray) and tau_c.shape == (3,):
            self._tau_c = tau_c.astype(float)
        else:
            raise ValueError("tau_c must be either a single float (for isotropic "
                             "rotational diffusion), a list/tuple of two or three floats "
                             "(for symmetric top or anisotropic rotational diffusion), "
                             "or a numpy array of shape (2,) or (3,).")
        print("Rotational correlation time(s) set to: " f"{self.tau_c}\n")

    @property
    def theory(self) -> str:
        """
        Specifies the relaxation theory to be used. Can be either "redfield" or
        "phenomenological".
        """
        return self._theory
    
    @theory.setter
    def theory(self, theory: Literal["redfield", "phenomenological"]):
        if theory not in ["redfield", "phenomenological"]:
            raise ValueError("Relaxation theory must be 'redfield' or "
                             "'phenomenological'.")
        self._theory = theory
        print(f"Relaxation theory set to: {self.theory}\n")

    @property
    def thermalization(self) -> bool:
        """
        Specifies whether to apply Levitt-di Bari thermalization to the
        relaxation superoperator.
        """
        return self._thermalization
    
    @thermalization.setter
    def thermalization(self, thermalization: bool):
        self._thermalization = thermalization
        print(f"Thermalization set to: {self.thermalization}\n")

    @property
    def T1(self) -> np.ndarray:
        """
        Specifies the longitudinal relaxation time constants for each spin.
        These are used to create the phenomenological relaxation superoperator.
        Two input types are supported:

        - If `ArrayLike`: A 1D array of size N containing T1 times.
        - If `str`: Path to the file containing the T1 times.

        The input will be converted and stored as a NumPy array.

        Examples::

            # Using array input
            spin_system.relaxation.T1 = np.array([5.5, 6.0, 2.7])

            # Using string input
            spin_system.relaxation.T1 = "/path/to/the/file/T1.txt"

        """
        return self._T1
    
    @T1.setter
    def T1(self, T1: list | tuple | np.ndarray | str):
        # Handle string input
        if isinstance(T1, str):
            T1 = read_array(T1, data_type=float)
            
        # Handle array like input
        elif isinstance(T1, (list, tuple, np.ndarray)):
            T1 = arraylike_to_array(T1)

        # Otherwise throw an error
        else:
            raise TypeError("T1 should be a 1-dimensional array or a string.")
        
        # Check that the input is valid
        if T1.shape != self._spin_system.isotopes.shape:
            raise ValueError("Mismatch between the given T1 times and the "
                             "number of spins in the system.")
        
        self._T1 = T1
        print(f"T1 set to: {self.T1}\n")

    @property
    def T2(self) -> np.ndarray:
        """
        Specifies the transverse relaxation time constants for each spin. These
        are used to create the phenomenological relaxation superoperator.
        Two input types are supported:

        - If `ArrayLike`: A 1D array of size N containing T2 times. Example:
        - If `str`: Path to the file containing the T2 times.

        The input will be converted and stored as a NumPy array.

        Examples::

            # Using array input
            spin_system.relaxation.T2 = np.array([5.5, 6.0, 2.7])

            # Using string input
            spin_system.relaxation.T2 = "/path/to/the/file/T2.txt"
        """
        return self._T2
    
    @T2.setter
    def T2(self, T2: list | tuple | np.ndarray | str):
        # Handle string input
        if isinstance(T2, str):
            T2 = read_array(T2, data_type=float)
            
        # Handle array like input
        elif isinstance(T2, (list, tuple, np.ndarray)):
            T2 = arraylike_to_array(T2)

        # Otherwise throw an error
        else:
            raise TypeError("T2 should be a 1-dimensional array or a string.")
    
        # Check that the input is valid
        if T2.shape != self._spin_system.isotopes.shape:
            raise ValueError("Mismatch between the given T2 times and the "
                             "number of spins in the system.")
        
        self._T2 = T2
        print(f"T2 set to: {self.T2}\n")

    @property
    def R1(self) -> np.ndarray:
        """
        Contains the longitudinal relaxation rates for each spin in the system.
        """
        return 1 / self.T1
    
    @property
    def R2(self) -> np.ndarray:
        """
        Contains the transverse relaxation rates for each spin in the system.
        """
        return 1 / self.T2
    
    @property
    def molecule(self) -> Molecule:
        """
        Molecule that the spin system to be simulated is part of. Used to
        define the rotational principal axes.
        """
        return self._molecule
    
    @molecule.setter
    def molecule(self, molecule: Molecule):
        # Check that the input is valid
        if not isinstance(molecule, Molecule):
            raise ValueError("Invalid input type for molecule.")
        
        self._molecule = molecule
        print("Molecule has been assigned.\n")