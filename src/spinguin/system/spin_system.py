"""
spin_system.py

Defines a class for the spin system. Once a spin system is initialized,
other modules can be used to calculate its properties.
"""

# Imports
from __future__ import annotations
import numpy as np
import scipy.sparse as sp
import warnings
from spinguin.utils.data_io import read_array, read_tensors, read_xyz
from spinguin.utils.hide_prints import HidePrints
from spinguin.utils.la import arraylike_to_array
from spinguin.utils.nmr_isotopes import ISOTOPES
from spinguin.qm.basis import make_basis
from spinguin.qm.hamiltonian import sop_H_coherent, sop_H_CS, sop_H_J, sop_H_Z
from spinguin.qm.operators import op_from_string
from spinguin.qm.relaxation import sop_R_phenomenological, sop_R_redfield, sop_R_sr2k, ldb_thermalization
from spinguin.qm.superoperators import sop_from_string
from typing import Literal

class SpinSystem:
    """
    Initializes the spin system with the default parameters.

    Attributes
    ----------
    isotopes : ndarray
        Specifies the isotopes that constitute the spin system and determine
        other properties, such as spin quantum numbers and gyromagnetic ratios.
        Example:

        ```python
        np.array(['1H', '15N', '19F'])
        ```
    
    chemical_shifts : ndarray
        Chemical shifts arising from the isotropic component of the nuclear
        shielding tensors. Used when calculating the coherent Hamiltonian.
        Example:

        ```python
        np.array([8.00, -200, -130])
        ```

    J_couplings : ndarray
        Specifies the scalar coupling constants between each spin pair in the
        spin system. Used when calculating the coherent Hamiltonian. Example:

        ```python
        np.array([
            [0,    0,    0],
            [1,    0,    0],
            [0.2,  8,    0]
        ])
        ```

    xyz : ndarray
        Coordinates in the XYZ format for each nucleus in the spin system. Used
        in Redfield relaxation theory when calculating the dipole-dipole
        coupling tensors. Example:

        ```python
        np.array([
            [1.025, 2.521, 1.624],
            [0.667, 2.754, 0.892]
        ])
        ```

    shielding : ndarray
        Specifies the nuclear shielding tensors for each nucleus. Note that the
        isotropic part of the tensor is handled by `chemical_shifts`. The
        shielding tensors are used only for relaxation. Example:

        ```python
        np.array([
            [[0, 0, 0],
             [0, 0, 0],
             [0, 0, 0]],
            [[101.6, -75.2, 11.1],
             [30.5,   10.1, 87.4],
             [99.7,  -21.1, 11.2]]
        ])
        ```

    efg : ndarray   TODO: Merkin määrittely selväksi. (Perttu)
        Electric field gradient tensors used for incorporating the quadrupolar
        interaction relaxation mechanism. Example:

        ```python
        efg = np.array([
            [[0, 0, 0],
             [0, 0, 0],
             [0, 0, 0]],
            [[ 0.31, 0.00, 0.01],
             [-0.20, 0.04, 0.87],
             [ 0.11, 0.16, 0.65]]
        ])
        ```

    tau_c : float
        Isotropic rotational correlation time in units of s. Must be defined to
        use the Redfield relaxation theory.

    max_spin_order : int
        Defines the maximum spin order included in the basis set.
        If left empty, the spin order is set to the size of the system.
    """

    # Spin system
    _isotopes: np.ndarray = None
    _chemical_shifts: np.ndarray = None
    _J_couplings: np.ndarray = None
    _xyz: np.ndarray = None
    _shielding: np.ndarray = None
    _efg: np.ndarray = None

    # Basis set
    _max_spin_order: int = None
    _basis: np.ndarray = None

    # Experimental conditions
    _magnetic_field: float = None
    _temperature: float = None

    # Relaxation properties
    _relaxation_theory: None | Literal["redfield", "phenomenological"] = None
    _thermalization: bool = False
    _tau_c: float = None
    _sr2k: bool=False
    _dynamic_frequency_shift: bool=False
    _antisymmetric_relaxation: bool=False
    _relative_error: float = 1e-6
    _T1: np.ndarray = None
    _T2: np.ndarray = None

    # Sparsity settings
    _sparse_operator: bool=True
    _sparse_superoperator: bool=True
    _sparse_hamiltonian: bool=True
    _sparse_relaxation: bool=True
    _propagator_density: float=0.5
    _sparse_state: bool=False

    # Zero-value thresholds
    _zero_hamiltonian: float = 1e-12
    _zero_aux: float = 1e-18
    _zero_relaxation: float = 1e-12
    _zero_interaction: float = 1e-9
    _zero_propagator: float = 1e-18
    _zero_pulse: float = 1e-18
    _zero_thermalization: float = 1e-18
    _zero_equilibrium: float = 1e-18

    def __init__(self):
        print("Spin system has been initialized with the following defaults:")
        print(f"isotopes: {self.isotopes}")
        print(f"chemical_shifts: {self.chemical_shifts}")
        print(f"J_couplings: {self.J_couplings}")
        print(f"xyz: {self.xyz}")
        print(f"shielding: {self.shielding}")
        print(f"efg: {self.efg}")
        print(f"max_spin_order: {self.max_spin_order}")
        print(f"magnetic_field: {self.magnetic_field}")
        print(f"temperature: {self.temperature}")
        print(f"relaxation_theory: {self.relaxation_theory}")
        print(f"tau_c: {self.tau_c}")
        print(f"sr2k: {self.sr2k}")
        print(f"dynamic_frequency_shift: {self.dynamic_frequency_shift}")
        print(f"antisymmetric_relaxation: {self.antisymmetric_relaxation}")
        print(f"relative_error: {self.relative_error}")

    def _check_attributes(self, *attributes: str, function_name: str):
        """
        This method checks whether the requested attributes have been set.

        Parameters
        ----------
        *attributes : str
            The names of the attributes to be checked.
        function_name : str
            The name of the function which is calling this function.
        """
        for attribute in attributes:
            if getattr(self, attribute) is None:
                raise ValueError(f"Please assign {attribute} "
                                 f"before calling {function_name}.")

    def _check_consistency(self, configuring_isotopes: bool=False):
        """
        This method checks the consistency of the SpinSystem object by comparing
        the shapes of the attributes.
        """

        # Insist that isotopes must be configured first
        if self.isotopes is None and not configuring_isotopes:
            raise ValueError("Please set isotopes before setting other "
                                 "properties.")
        
        # Check that the isotopes array is one-dimensional
        if self.isotopes.ndim != 1:
            raise ValueError("Isotopes must be a 1D array containing the "
                             "names of the isotopes as strings.")

        # Check that each isotope exists in the dictionary
        for isotope in self.isotopes:
            if isotope not in ISOTOPES:
                raise ValueError(f"Isotope '{isotope}' is not defined in the "
                                 "ISOTOPES dictionary.")

        # Check that the chemical shifts array is of correct size
        if self.chemical_shifts is not None:
            if self.chemical_shifts.shape != (self.nspins, ):
                raise ValueError("Chemical shifts must be a 1D array with a "
                                 "length equal to the number of isotopes.")
            
        # Check that the J-couplings array is of correct size
        if self.J_couplings is not None:
            if self.J_couplings.shape != (self.nspins, self.nspins):
                raise ValueError("J-couplings must be a 2D array with both of "
                                 "the dimensions equal to the number of isotopes.")
            
        # Check that the XYZ array is of correct size
        if self.xyz is not None:
            if self.xyz.shape != (self.nspins, 3):
                raise ValueError("XYZ coordinates must be a 2D array with the "
                                 "number of rows equal to the number of isotopes.")
            
        # Check that shielding tensors array is of correct size
        if self.shielding is not None:
            if self.shielding.shape != (self.nspins, 3, 3):
                raise ValueError("Shielding tensors must be a 3D array with the "
                                 "number of 3x3 tensors equal to the number of "
                                 "isotopes.")
            
        # Check that EFG tensors array is of correct size
        if self.efg is not None:
            if self.efg.shape != (self.nspins, 3, 3):
                raise ValueError("EFG tensors must be a 3D array with the "
                                 "number of 3x3 tensors equal to the number of "
                                 "isotopes.")
            
        # Check that maximum spin order is reasonable
        if self.max_spin_order is not None:
            if self.max_spin_order > self.nspins:
                raise ValueError("Maximum spin order should not exceed the number "
                                "of spins in the system.")
            elif self.max_spin_order < 1:
                raise ValueError("Maximum spin order should be at least one.")
            
        # Check that T1 array is of correct size
        if self.T1 is not None:
            if self.T1.shape != (self.nspins, ):
                raise ValueError("T1 must be a 1D array containing the longitudinal" \
                                 "relaxation time constants for each spin.")
            
        # Check that T2 array is of correct size
        if self.T2 is not None:
            if self.T2.shape != (self.nspins, ):
                raise ValueError("T2 must be a 1D array containing the transverse" \
                                 "relaxation time constants for each spin.")

    ##########################
    # SPIN SYSTEM PROPERTIES #
    ##########################

    @property
    def isotopes(self) -> np.ndarray:
        return self._isotopes

    @isotopes.setter
    def isotopes(self, isotopes):
        """
        Specifies the isotopes that constitute the spin system and determine
        other properties, such as spin quantum numbers and gyromagnetic ratios.
        Two input types are supported:

        - If `ArrayLike`: A 1D array of size N containing isotope names as
          strings. Example:

        ```python
        np.array(['1H', '15N', '19F'])
        ```

        - If `str`: Path to the file containing the isotopes.

        The input will be stored as a NumPy array.
        """
        # Handle string input
        if isinstance(isotopes, str):
            self._isotopes = read_array(isotopes, data_type=str)
        
        # Handle array like input
        elif isinstance(isotopes, (list, tuple, np.ndarray)):
            self._isotopes = arraylike_to_array(isotopes)

        # Otherwise throw an exception
        else:
            raise TypeError("Isotopes should be a 1-dimensional array or a string.")
        
        # Check input consistency
        self._check_consistency(configuring_isotopes=True)
        
        print(f"Assigned the following isotopes:\n{self.isotopes}\n")
            
    @property
    def chemical_shifts(self) -> np.ndarray:
        return self._chemical_shifts
    
    @chemical_shifts.setter
    def chemical_shifts(self, chemical_shifts):
        """
        Chemical shifts arising from the isotropic component of the nuclear
        shielding tensors. Used when calculating the coherent Hamiltonian.

        - If `ArrayLike`: A 1D array of size N containing the chemical shifts
          in ppm. Example:

        ```python
        np.array([8.00, -200, -130])
        ```

        - If `str`: Path to the file containing the chemical shifts.

        The input will be stored as a NumPy array.
        """
        # Handle string input
        if isinstance(chemical_shifts, str):
            self._chemical_shifts = read_array(chemical_shifts, data_type=float)
            
        # Handle array like input
        elif isinstance(chemical_shifts, (list, tuple, np.ndarray)):
            self._chemical_shifts = arraylike_to_array(chemical_shifts)

        # Otherwise throw an error
        else:
            raise TypeError("Chemical shifts should be a 1-dimensional array or a string.")
        
        # Check input consistency
        self._check_consistency()
            
        print(f"Assigned the following chemical shifts:\n{self.chemical_shifts}\n")
            
    @property
    def J_couplings(self) -> np.ndarray:
        return self._J_couplings
    
    @J_couplings.setter
    def J_couplings(self, J_couplings):
        """
        Specifies the J-coupling constants between each spin pair in the spin
        system. Used when calculating the coherent Hamiltonian.

        - If `ArrayLike`: A 2D array of size (N, N) specifying the scalar
          couplings between nuclei in Hz. Only the lower triangle is specified.
          Example:

        ```python
        np.array([
            [0,    0,    0],
            [1,    0,    0],
            [0.2,  8,    0]
        ])
        ```

        - If `str`: Path to the file containing the scalar couplings.

        The input will be stored as a NumPy array.
        """
        # Handle string input
        if isinstance(J_couplings, str):
            self._J_couplings = read_array(J_couplings, data_type=float)

        # Handle array like input
        elif isinstance(J_couplings, (list, tuple, np.ndarray)):
            self._J_couplings = arraylike_to_array(J_couplings)

        # Otherwise throw an error
        else:
            raise TypeError("J-couplings should be a 2-dimensional array or a string.")
        
        # Check input consistency
        self._check_consistency()
        
        print(f"Assigned the following J-couplings:\n{self.J_couplings}\n")

    @property
    def xyz(self) -> np.ndarray:
        return self._xyz
    
    @xyz.setter
    def xyz(self, xyz):
        """
        Coordinates in the XYZ format for each nucleus in the spin system. Used
        in Redfield relaxation theory when calculating the dipole-dipole
        coupling tensors.  
    
        - If `ArrayLike`: A 2D array of size (N, 3) containing the Cartesian
          coordinates in Å. Example:

        ```python
        np.array([
            [1.025, 2.521, 1.624],
            [0.667, 2.754, 0.892]
        ])
        ```

        - If `str`: Path to the file containing the XYZ coordinates.

        The input will be stored as a NumPy array.
        """
        # Handle string input
        if isinstance(xyz, str):
            self._xyz = read_xyz(xyz)

        # Handle array like input
        elif isinstance(xyz, (list, tuple, np.ndarray)):
            self._xyz = arraylike_to_array(xyz)
        
        # Otherwise throw an exception
        else:
            raise TypeError("XYZ should be a 2-dimensional array or a string.")
        
        # Check input consistency
        self._check_consistency()
        
        print(f"Assigned the following XYZ coordinates:\n{self.xyz}\n")

    @property
    def shielding(self) -> np.ndarray:
        return self._shielding
    
    @shielding.setter
    def shielding(self, shielding):
        """
        Specifies the nuclear shielding tensors for each nucleus. Note that the
        isotropic part of the tensor is handled by `chemical_shifts`. The
        shielding tensors are used only for Redfield relaxation theory.

        - If `ArrayLike`: A 3D array of size (N, 3, 3) containing the 3x3
          shielding tensors in ppm. Example:

        ```python
        np.array([
            [[0, 0, 0],
             [0, 0, 0],
             [0, 0, 0]],
            [[101.6, -75.2, 11.1],
             [30.5,   10.1, 87.4],
             [99.7,  -21.1, 11.2]]
        ])
          ```

        - If `str`: Path to the file containing the shielding tensors.

        The input will be stored as a NumPy array.
        """

        # Handle string input
        if isinstance(shielding, str):
            self._shielding = read_tensors(shielding)

        # Handle array like input
        elif isinstance(shielding, (list, tuple, np.ndarray)):
            self._shielding = arraylike_to_array(shielding)

        # Otherwise throw an error
        else:
            raise TypeError("Shielding should be a 3-dimensional array or a string.")
        
        # Check input consistency
        self._check_consistency()

        print(f"Assigned the following shielding tensors:\n{self.shielding}\n")
        
    @property
    def efg(self) -> np.ndarray:
        return self._efg
    
    @efg.setter
    def efg(self, efg):
        """
        Electric field gradient tensors used for incorporating the quadrupolar
        interaction relaxation mechanism.

        - If `ArrayLike`: A 3D array of size (N, 3, 3) containing the 3x3 EFG
          tensors in atomic units. Example:

        ```python
        efg = np.array([
            [[0, 0, 0],
             [0, 0, 0],
             [0, 0, 0]],
            [[ 0.31, 0.00, 0.01],
             [-0.20, 0.04, 0.87],
             [ 0.11, 0.16, 0.65]]
        ])
        ```

        - If `str`: Path to the file containing the EFG tensors.

        The input will be stored as a NumPy array.
        """
        # Handle string input
        if isinstance(efg, str):
            self._efg = read_tensors(efg)

        # Handle array like input
        elif isinstance(efg, (list, tuple, np.ndarray)):
            self._efg = arraylike_to_array(efg)

        # Otherwise throw an error
        else:
            raise TypeError("EFG should be a 3-dimensional array or a string.")
        
        # Check input consistency
        self._check_consistency()

        print(f"Assigned the following EFG tensors:\n{self.efg}\n")

    @property
    def nspins(self) -> int:
        """Returns the number of spins in the spin system."""
        return len(self.isotopes)
    
    @property
    def spins(self) -> np.ndarray:
        """Returns the spin quantum numbers of the spin system."""
        return np.array([ISOTOPES[isotope][0] for isotope in self.isotopes])
    
    @property
    def mults(self) -> np.ndarray:
        """Returns the spin multiplicities of the spin system."""
        return np.array([int(2 * ISOTOPES[isotope][0] + 1) for isotope in self.isotopes], dtype=int)
    
    @property
    def gammas(self) -> np.ndarray:
        """Returns the gyromagnetic ratios in rad/s/T."""
        return np.array([2 * np.pi * ISOTOPES[isotope][1] * 1e6 for isotope in self.isotopes])
    
    @property
    def quad(self) -> np.ndarray:
        """Returns the quadrupolar moments in m^2."""
        return np.array([ISOTOPES[isotope][2] * 1e-28 for isotope in self.isotopes])

    ########################
    # BASIS SET PROPERTIES #
    ########################

    @property
    def max_spin_order(self) -> float:
        return self._max_spin_order
    
    @max_spin_order.setter
    def max_spin_order(self, max_spin_order):
        self._max_spin_order = max_spin_order
        self._check_consistency()
        print(f"Maximum spin order set to: {self.max_spin_order}\n")

    @property
    def basis(self) -> np.ndarray:
        return self._basis
    
    def build_basis(self):
        """
        Builds the basis set for the spin system. Prior to building the basis,
        the isotopes must be defined.
        """
        # Check that the isotopes have been set
        if self.isotopes is None:
            raise ValueError("Isotopes must be set prior to building basis.")

        # If maximum spin order is not specified, set it equal to number of spins
        if self.max_spin_order is None:
            warnings.warn("Maximum spin order not specified.\
                          Defaulting to the number of spins.")
            self.max_spin_order = self.nspins

        # Build the basis
        self._basis = make_basis(spins = self.spins,
                                 max_spin_order = self.max_spin_order)
        
    ###########################
    # EXPERIMENTAL CONDITIONS #
    ###########################
        
    @property
    def magnetic_field(self) -> float:
        return self._magnetic_field
    
    @magnetic_field.setter
    def magnetic_field(self, magnetic_field):
        self._magnetic_field = magnetic_field
        print(f"Magnetic field set to: {self.magnetic_field}\n")

    @property
    def temperature(self) -> float:
        return self._temperature
    
    @temperature.setter
    def temperature(self, temperature):
        self._temperature = temperature
        print(f"Temperature set to: {self.temperature}\n")

    #######################
    # RELAXATION SETTINGS #
    #######################

    @property
    def relaxation_theory(self) -> str:
        return self._relaxation_theory
    
    @relaxation_theory.setter
    def relaxation_theory(self, relaxation_theory):
        self._relaxation_theory = relaxation_theory
        print(f"Relaxation theory set to: {self.relaxation_theory}")

    @property
    def thermalization(self) -> bool:
        return self._thermalization
    
    @thermalization.setter
    def thermalization(self, thermalization):
        self._thermalization = thermalization
        print(f"Thermalization of relaxation superoperator set to: {self.thermalization}")

    @property
    def tau_c(self) -> float:
        return self._tau_c
    
    @tau_c.setter
    def tau_c(self, tau_c):
        self._tau_c = tau_c
        print(f"Correlation time (tau_c) set to: {self.tau_c}")

    @property
    def sr2k(self) -> bool:
        return self._sr2k
    
    @sr2k.setter
    def sr2k(self, sr2k):
        self._sr2k = sr2k
        print(f"Scalar relaxation of the second kind set to: {self.sr2k}")

    @property
    def dynamic_frequency_shift(self) -> bool:
        return self._dynamic_frequency_shift
    
    @dynamic_frequency_shift.setter
    def dynamic_frequency_shift(self, dfs):
        self._dynamic_frequency_shift = dfs
        print(f"Dynamic frequency shifts set to: {self.dynamic_frequency_shift}")

    @property
    def antisymmetric_relaxation(self) -> bool:
        return self._antisymmetric_relaxation
    
    @antisymmetric_relaxation.setter
    def antisymmetric_relaxation(self, antisym):
        self._antisymmetric_relaxation = antisym
        print(f"Antisymmetric relaxation set to: {self.antisymmetric_relaxation}")

    @property
    def relative_error(self) -> float:
        return self._relative_error
    
    @relative_error.setter
    def relative_error(self, relative_error):
        self._relative_error = relative_error
        print(f"Relative error set to: {self.relative_error}")

    @property
    def T1(self) -> np.ndarray:
        return self._T1
    
    @T1.setter
    def T1(self, T1):
        """
        Specifies the longitudinal relaxation time constants for each spin.
        These are used to create the phenomenological relaxation superoperator.
        Two input types are supported:

        - If `ArrayLike`: A 1D array of size N containing T1 times. Example:

        ```python
        np.array([5.5, 6.0, 2.7])
        ```

        - If `str`: Path to the file containing the T1 times.

        The input will be stored as a NumPy array.
        """
        # Handle string input
        if isinstance(T1, str):
            self._T1 = read_array(T1, data_type=float)
            
        # Handle array like input
        elif isinstance(T1, (list, tuple, np.ndarray)):
            self._T1 = arraylike_to_array(T1)

        # Otherwise throw an error
        else:
            raise TypeError("T1 should be a 1-dimensional array or a string.")
        
        # Check input consistency
        self._check_consistency()
            
        print(f"Assigned the following T1 values:\n{self.T1}\n")

    @property
    def T2(self) -> np.ndarray:
        return self._T2
    
    @T2.setter
    def T2(self, T2):
        """
        Specifies the transverse relaxation time constants for each spin. These
        are used to create the phenomenological relaxation superoperator.
        Two input types are supported:

        - If `ArrayLike`: A 1D array of size N containing T2 times. Example:

        ```python
        np.array([5.5, 6.0, 2.7])
        ```

        - If `str`: Path to the file containing the T2 times.

        The input will be stored as a NumPy array.
        """
        # Handle string input
        if isinstance(T2, str):
            self._T2 = read_array(T2, data_type=float)
            
        # Handle array like input
        elif isinstance(T2, (list, tuple, np.ndarray)):
            self._T2 = arraylike_to_array(T2)

        # Otherwise throw an error
        else:
            raise TypeError("T2 should be a 1-dimensional array or a string.")
        
        # Check input consistency
        self._check_consistency()
            
        print(f"Assigned the following T2 values:\n{self.T2}\n")

    @property
    def R1(self) -> np.ndarray:
        return 1 / self.T1
    
    @property
    def R2(self) -> np.ndarray:
        return 1 / self.T2

    #####################
    # SPARSITY SETTINGS #
    #####################

    @property
    def sparse_operator(self) -> bool:
        return self._sparse_operator
    
    @sparse_operator.setter
    def sparse_operator(self, sparse):
        self._sparse_operator = sparse
        print(f"Sparsity setting of operators set to: {self.sparse_operator}")

    @property
    def sparse_superoperator(self) -> bool:
        return self._sparse_superoperator
    
    @sparse_superoperator.setter
    def sparse_superoperator(self, sparse):
        self._sparse_superoperator = sparse
        print(f"Sparsity setting of superoperators set to: {self.sparse_superoperator}")

    @property
    def sparse_hamiltonian(self) -> bool:
        return self._sparse_hamiltonian
    
    @sparse_hamiltonian.setter
    def sparse_hamiltonian(self, sparse):
        self._sparse_hamiltonian = sparse
        print(f"Sparsity setting of the Hamiltonian set to: {self.sparse_hamiltonian}")

    @property
    def sparse_relaxation(self) -> bool:
        return self._sparse_relaxation
    
    @sparse_relaxation.setter
    def sparse_relaxation(self, sparse):
        self._sparse_relaxation = sparse
        print("Sparsity setting of the relaxation superoperator "
              f"set to: {self.sparse_relaxation}")
        
    @property
    def propagator_density(self) -> float:
        return self._propagator_density
    
    @propagator_density.setter
    def propagator_density(self, density):
        self._propagator_density = density
        print(f"Propagator density threshold set to: {self.propagator_density}")

    @property
    def sparse_state(self) -> bool:
        return self._sparse_state
    
    @sparse_state.setter
    def sparse_state(self, sparse):
        self._sparse_state = sparse
        print(f"Sparsity of states set to: {self.sparse_state}")

    #########################
    # ZERO-VALUE THRESHOLDS #
    #########################

    @property
    def zero_hamiltonian(self) -> float:
        return self._zero_hamiltonian
    
    @zero_hamiltonian.setter
    def zero_hamiltonian(self, zero):
        self._zero_hamiltonian = zero
        print(f"Zero-value threshold for Hamiltonian set to: {self.zero_hamiltonian}")

    @property
    def zero_aux(self) -> float:
        return self._zero_aux
    
    @zero_aux.setter
    def zero_aux(self, zero):
        self._zero_aux = zero
        print(f"Zero-value threshold for auxiliary matrix method set to: {self.zero_aux}")

    @property
    def zero_relaxation(self) -> float:
        return self._zero_relaxation
    
    @zero_relaxation.setter
    def zero_relaxation(self, zero):
        self._zero_relaxation = zero
        print(f"Zero-value threshold for relaxation superoperator set to: {self.zero_relaxation}")

    @property
    def zero_interaction(self) -> float:
        return self._zero_interaction
    
    @zero_interaction.setter
    def zero_interaction(self, zero):
        self._zero_interaction = zero
        print(f"Zero-value threshold for interaction tensors set to: {self.zero_interaction}")

    @property
    def zero_propagator(self) -> float:
        return self._zero_propagator
    
    @zero_propagator.setter
    def zero_propagator(self, zero):
        self._zero_propagator = zero
        print(f"Zero-value threshold for propagator set to: {self.zero_propagator}")

    @property
    def zero_pulse(self) -> float:
        return self._zero_pulse
    
    @zero_pulse.setter
    def zero_pulse(self, zero):
        self._zero_pulse = zero
        print(f"Zero-value threshold for pulses set to: {self.zero_pulse}")

    @property
    def zero_thermalization(self) -> float:
        return self._zero_thermalization
    
    @zero_thermalization.setter
    def zero_thermalization(self, zero):
        self._zero_thermalization = zero
        print(f"Zero-value for thermalization set to: {self.zero_thermalization}")

    @property
    def zero_equilibrium(self) -> float:
        return self._zero_equilibrium
    
    @zero_equilibrium.setter
    def zero_equilibrium(self, zero):
        self._zero_equilibrium = zero
        print(f"Zero-value for the equilibrium state set to: {self.zero_equilibrium}")

    def operator(self,
                 operator: str):
        """
        Generates an operator for the `spin_system` in Hilbert space from the
        user-specified `operator` string.

        Parameters
        ----------
        operator : str
            Defines the operator to be generated. The operator string must
            follow the rules below:

            - Cartesian and ladder operators: `I(component,index)` or
              `I(component)`. Examples:

                - `I(x,4)` --> Creates x-operator for spin at index 4.
                - `I(x)`--> Creates x-operator for all spins.

            - Spherical tensor operators: `T(l,q,index)` or `T(l,q)`. Examples:

                - `T(1,-1,3)` --> Creates operator with `l=1`, `q=-1` for spin at index 3.
                - `T(1, -1) --> Creates operator with `l=1`, `q=-1` for all spins.
                
            - Product operators have `*` in between the single-spin operators:
              `I(z,0) * I(z,1)`
            - Sums of operators have `+` in between the operators:
              `I(x,0) + I(x,1)`
            - Unit operators are ignored in the input. Interpretation of these
              two is identical: `E * I(z,1)`, `I(z,1)`
            
            Special case: An empty `operator` string is considered as unit
            operator.

            Whitespace will be ignored in the input.

            NOTE: Indexing starts from 0!

        Returns
        -------
        op : ndarray or csc_array
            An array representing the requested operator.
        """
        self._check_attributes("isotopes", function_name="operator")
        op = op_from_string(spins = self.spins,
                            operator = operator,
                            sparse = self.sparse_operator)
        return op
    
    def superoperator(self,
                      operator,
                      side: Literal["comm", "left", "right"] = "comm"):
        """
        Generates a Liouville-space superoperator for the `spin_system` in from
        the user-specified `operator` string.

        Parameters
        ----------
        operator : str
            Defines the superoperator to be generated. The operator string must
            follow the rules below:

            - Cartesian and ladder operators: `I(component,index)` or
              `I(component)`. Examples:

                - `I(x,4)` --> Creates x-operator for spin at index 4.
                - `I(x)`--> Creates x-operator for all spins.

            - Spherical tensor operators: `T(l,q,index)` or `T(l,q)`. Examples:

                - `T(1,-1,3)` --> Creates operator with `l=1`, `q=-1` for spin at index 3.
                - `T(1, -1) --> Creates operator with `l=1`, `q=-1` for all spins.
                
            - Product operators have `*` in between the single-spin operators:
              `I(z,0) * I(z,1)`
            - Sums of operators have `+` in between the operators:
              `I(x,0) + I(x,1)`
            - Unit operators are ignored in the input. Interpretation of these
              two is identical: `E * I(z,1)`, `I(z,1)`
            
            Special case: An empty `operator` string is considered as unit
            operator.

            Whitespace will be ignored in the input.

            NOTE: Indexing starts from 0!

        Returns
        -------
        sop : ndarray or csc_array
            An array representing the requested superoperator.
        """
        self._check_attributes("isotopes", "basis", function_name="superoperator")
        sop = sop_from_string(operator = operator,
                              basis = self.basis,
                              spins = self.spins,
                              side = side,
                              sparse = self.sparse_superoperator)
        return sop

    def hamiltonian(self,
                    interactions: Literal["all", "zeeman", "chemical_shift", "J_coupling"] = "all",
                    side: Literal["comm", "left", "right"] = "comm"
                    ) -> np.ndarray | sp.csc_array:
        """
        Creates the requested Hamiltonian superoperator for the spin system.

        Parameters
        ----------
        interactions : {'all', 'zeeman', 'chemical_shift', 'J_coupling'}
            Specifies which interactions are taken into account.
        side : {'comm', 'left', 'right'}
            The type of superoperator:
            - 'comm' -- commutation superoperator (default)
            - 'left' -- left superoperator
            - 'right' -- right superoperator

        Returns
        -------
        H : ndarray or csc_array
            Hamiltonian superoperator.
        """
        self._check_attributes("isotopes", "basis", function_name="hamiltonian")
        match interactions:
            case "all":
                self._check_attributes("magnetic_field",
                                       "chemical_shifts",
                                       "J_couplings",
                                       function_name="hamiltonian:all")       
                H = sop_H_coherent(basis = self.basis,
                                   gammas = self.gammas,
                                   spins = self.spins,
                                   chemical_shifts = self.chemical_shifts,
                                   J_couplings = self.J_couplings,
                                   B = self.magnetic_field,
                                   side = side,
                                   sparse = self.sparse_hamiltonian,
                                   zero_value = self.zero_hamiltonian)
            case "zeeman":
                self._check_attributes("magnetic_field",
                                       function_name="hamiltonian:zeeman")
                H = sop_H_Z(basis = self.basis,
                            gammas = self.gammas,
                            spins = self.spins,
                            B = self.magnetic_field,
                            side = side,
                            sparse = self.sparse_hamiltonian)    
            case "chemical_shift":
                self._check_attributes("magnetic_field",
                                       "chemical_shifts",
                                       function_name="hamiltonian:chemical_shift")
                H = sop_H_CS(basis = self.basis,
                             gammas = self.gammas,
                             spins = self.spins,
                             chemical_shifts = self.chemical_shifts,
                             B = self.magnetic_field,
                             side = side,
                             sparse = self.sparse_hamiltonian)
            case "J_coupling":
                self._check_attributes("J_couplings",
                                       function_name="hamiltonian:J_coupling")
                H = sop_H_J(basis = self.basis,
                            spins = self.spins,
                            J_couplings = self.J_couplings,
                            side = side,
                            sparse = self.sparse_hamiltonian)   
        return H
    
    def relaxation(self, H: np.ndarray | sp.csc_array = None):
        """
        Creates the relaxation superoperator using the requested relaxation
        theory.

        Parameters
        ----------
        H : ndarray or csc_array, default=None
            Coherent Hamiltonian commutation superoperator. Must be given when
            using Redfield relaxation theory.

        Returns
        -------
        R : ndarray or csc_array
            Relaxation superoperator. 
        """

        # Make phenomenological relaxation superoperator
        if self.relaxation_theory == "phenomenological":
            self._check_attributes("basis", "T1", "T2")
            R = sop_R_phenomenological(basis = self.basis,
                                       R1 = self.R1,
                                       R2 = self.R2,
                                       sparse = self.sparse_relaxation)

        # Make relaxation superoperator using Redfield theory
        elif self.relaxation_theory == "redfield":
            if H is None:
                raise ValueError("Coherent Hamiltonian superoperator must be "
                                 "given as input when using Redfield relaxation "
                                 "theory.")
            self._check_attributes("basis", "tau_c", "B")
            R = sop_R_redfield(basis = self.basis,
                               sop_H = H,
                               tau_c = self.tau_c,
                               spins = self.spins,
                               B = self.magnetic_field,
                               gammas = self.gammas,
                               quad = self.quad,
                               xyz = self.xyz,
                               shielding = self.shielding,
                               efg = self.efg,
                               include_antisymmetric = self.antisymmetric_relaxation,
                               include_dynamic_frequency_shift = self.dynamic_frequency_shift,
                               relative_error = self.relative_error,
                               interaction_zero = self.zero_interaction,
                               aux_zero = self.zero_aux,
                               relaxation_zero = self.zero_relaxation,
                               sparse = self.sparse_relaxation
                               )

        # Other relaxation theories not supported
        else:
            raise ValueError(f"Relaxation theory '{self.relaxation_theory}' is "
                             "not supported. Available theorie are 'redfield' "
                             "and 'phenomenological'.")
        
        # Apply scalar relaxation of the second kind if requested
        if self.sr2k:
            R = R + sop_R_sr2k(basis = self.basis,
                               spins = self.spins,
                               gammas = self.gammas,
                               chemical_shifts = self.chemical_shifts,
                               J_couplings = self.J_couplings,
                               sop_R = R,
                               B = self.magnetic_field,
                               sparse = self.sparse_relaxation)
            
        # Apply thermalization if requested
        if self.thermalization:
            with HidePrints():
                H_left = sop_H_coherent(basis = self.basis,
                                        gammas = self.gammas,
                                        spins = self.spins,
                                        chemical_shifts = self.chemical_shifts,
                                        J_couplings = self.J_couplings,
                                        side = "left",
                                        sparse = self.sparse_hamiltonian,
                                        zero_value = self.zero_hamiltonian)
            R = ldb_thermalization(R = R,
                                   H_left = H_left,
                                   T = self.temperature,
                                   zero_value = self.zero_thermalization)

        return R

