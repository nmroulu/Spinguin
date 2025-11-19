"""
spin_system.py

Defines a class for the spin system. Once a spin system is initialized, other
modules can be used to calculate its properties.
"""

# Imports
import numpy as np
from spinguin._core._basis import Basis
from spinguin._core._relaxation_properties import RelaxationProperties
from spinguin._core._data_io import read_array, read_tensors, read_xyz
from spinguin._core._la import arraylike_to_array
from spinguin._core._nmr_isotopes import ISOTOPES
from typing import Self

class SpinSystem:
    """
    Initializes a spin system with the given `isotopes`. Examples::

        spin_system = SpinSystem(['1H', '15N', '19F'])
        spin_system = SpinSystem("/path/to/isotopes.txt")

    Parameters
    ----------
    isotopes : list or tuple or ndarray or str
        Specifies the isotopes that constitute the spin system and determine
        other properties, such as spin quantum numbers and gyromagnetic
        ratios.

        Two input types are supported:

        - If `ArrayLike`: A 1D array of size N containing isotope names as
          strings. 
        - If `str`: Path to the file containing the isotopes.

        The input will be converted and stored as a NumPy array.  
    """

    def __init__(self, isotopes: list | tuple | np.ndarray | str):

        # Assign the isotopes
        if isinstance(isotopes, str):
            self._isotopes = read_array(isotopes, data_type=str)
        elif isinstance(isotopes, (list, tuple, np.ndarray)):
            self._isotopes = arraylike_to_array(isotopes)
        else:
            raise TypeError("Isotopes should be a 1-dimensional array or a "
                            "string.")
        
        # Set the other spin system properties
        self._chemical_shifts = np.zeros(self.nspins)
        self._J_couplings = np.zeros((self.nspins, self.nspins))
        self._xyz: np.ndarray = None
        self._shielding: np.ndarray = None
        self._efg: np.ndarray = None

        # Check consistency
        self._check_consistency()

        print("Spin system has been initialized with the following values:")
        print(f"isotopes: {self.isotopes}")
        print(f"chemical_shifts: {self.chemical_shifts}")
        print(f"J_couplings:\n{self.J_couplings}")
        print(f"xyz: {self.xyz}")
        print(f"shielding: {self.shielding}")
        print(f"efg: {self.efg}")
        print()

        # Initialize basis set
        self._basis = Basis(self)

        # Initialize relaxation theory settings
        self._relaxation = RelaxationProperties(self)

    def _check_consistency(self):
        """
        This method checks the consistency of the SpinSystem object by comparing
        the shapes of the attributes.
        """
        
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
                                 "the dimensions equal to the number of " 
                                 "isotopes.")
            
        # Check that the XYZ array is of correct size
        if self.xyz is not None:
            if self.xyz.shape != (self.nspins, 3):
                raise ValueError("XYZ coordinates must be a 2D array with the "
                                 "number of rows equal to the number of " 
                                 "isotopes.")
            
        # Check that shielding tensors array is of correct size
        if self.shielding is not None:
            if self.shielding.shape != (self.nspins, 3, 3):
                raise ValueError("Shielding tensors must be a 3D array with "
                                 "the number of 3x3 tensors equal to the "
                                 "number of isotopes.")
            
        # Check that EFG tensors array is of correct size
        if self.efg is not None:
            if self.efg.shape != (self.nspins, 3, 3):
                raise ValueError("EFG tensors must be a 3D array with the "
                                 "number of 3x3 tensors equal to the number of "
                                 "isotopes.")
            
    def subsystem(self, spins: list) -> Self:
        """
        Creates a new `SpinSystem` object containing only the spins indicated
        by the `spins` list. The new spin system is assigned the appropriate
        properties from the original spin system. However, the basis set and
        relaxation properties are not copied to the new spin system.

        Parameters
        ----------
        spins : list
            List of indices that define which spins to include in the subsystem.

        Returns
        -------
        sub : SpinSystem
            A new `SpinSystem` object containing only the specified spins.
        """
        # Check that each spin has been given only once
        if len(set(spins)) != len(spins):
            raise ValueError("Each spin must be unique in 'spins'")
        
        # Check that the spins isn't empty
        if len(spins) == 0:
            raise ValueError("'spins' cannot be empty")
        
        # Check that the maximum index makes sense
        if max(spins) >= self.nspins:
            raise ValueError(f"Spin system does not have spin: {max(spins)}")
        
        # Create the new SpinSystem object
        spin_system = SpinSystem(self.isotopes[spins])

        # Assign chemical shifts
        if self.chemical_shifts is not None:
            spin_system.chemical_shifts = self.chemical_shifts[spins]

        # Assign J-couplings
        if self.J_couplings is not None:
            spin_system.J_couplings = self.J_couplings[np.ix_(spins, spins)]

        # Assign XYZ
        if self.xyz is not None:
            spin_system.xyz = self.xyz[spins]

        # Assign shielding tensors
        if self.shielding is not None:
            spin_system.shielding = self.shielding[spins]

        # Assign EFG tensors
        if self.efg is not None:
            spin_system.efg = self.efg[spins]

        return spin_system

    ##########################
    # SPIN SYSTEM PROPERTIES #
    ##########################

    @property
    def isotopes(self) -> np.ndarray:
        """
        Specifies the isotopes that constitute the spin system and determine
        other properties, such as spin quantum numbers and gyromagnetic ratios.
        Example::

            np.array(['1H', '15N', '19F'])

        Isotopes are set during the initialization of the spin system.
        """
        return self._isotopes
            
    @property
    def chemical_shifts(self) -> np.ndarray:
        return self._chemical_shifts
    
    @chemical_shifts.setter
    def chemical_shifts(self, chemical_shifts: list | tuple | np.ndarray | str):
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
        # Assign chemical shifts
        if isinstance(chemical_shifts, str):
            self._chemical_shifts = read_array(chemical_shifts, data_type=float)
        elif isinstance(chemical_shifts, (list, tuple, np.ndarray)):
            self._chemical_shifts = arraylike_to_array(chemical_shifts)
        else:
            raise TypeError("Chemical shifts should be a 1-dimensional array "
                            "or a string.")
        
        # Check input consistency
        self._check_consistency()
            
        print("Assigned the following chemical shifts:")
        print(f"{self.chemical_shifts}\n")
            
    @property
    def J_couplings(self) -> np.ndarray:
        return self._J_couplings
    
    @J_couplings.setter
    def J_couplings(self, J_couplings: list | tuple | np.ndarray | str):
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
        # Assign J-couplings
        if isinstance(J_couplings, str):
            self._J_couplings = read_array(J_couplings, data_type=float)
        elif isinstance(J_couplings, (list, tuple, np.ndarray)):
            self._J_couplings = arraylike_to_array(J_couplings)
        else:
            raise TypeError("J-couplings should be a 2-dimensional array or a "
                            "string.")
        
        # Check input consistency
        self._check_consistency()
        
        print(f"Assigned the following J-couplings:\n{self.J_couplings}\n")

    @property
    def xyz(self) -> np.ndarray:
        return self._xyz
    
    @xyz.setter
    def xyz(self, xyz: list | tuple | np.ndarray | str):
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
        # Assign XYZ coordinates
        if isinstance(xyz, str):
            self._xyz = read_xyz(xyz)
        elif isinstance(xyz, (list, tuple, np.ndarray)):
            self._xyz = arraylike_to_array(xyz)
        else:
            raise TypeError("XYZ should be a 2-dimensional array or a string.")
        
        # Check input consistency
        self._check_consistency()

        print(f"Assigned the following XYZ coordinates:\n{self.xyz}\n")

    @property
    def shielding(self) -> np.ndarray:
        return self._shielding
    
    @shielding.setter
    def shielding(self, shielding: list | tuple | np.ndarray | str):
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

        # Assign shielding tensors
        if isinstance(shielding, str):
            self._shielding = read_tensors(shielding)
        elif isinstance(shielding, (list, tuple, np.ndarray)):
            self._shielding = arraylike_to_array(shielding)
        else:
            raise TypeError("Shielding should be a 3-dimensional array or a "
                            "string.")
        
        # Check input consistency
        self._check_consistency()

        print(f"Assigned the following shielding tensors:\n{self.shielding}\n")
        
    @property
    def efg(self) -> np.ndarray:
        return self._efg
    
    @efg.setter
    def efg(self, efg: list | tuple | np.ndarray | str):
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
        # Assign EFG tensors
        if isinstance(efg, str):
            self._efg = read_tensors(efg)
        elif isinstance(efg, (list, tuple, np.ndarray)):
            self._efg = arraylike_to_array(efg)
        else:
            raise TypeError("EFG should be a 3-dimensional array or a string.")
        
        # Check input consistency
        self._check_consistency()

        print(f"Assigned the following EFG tensors:\n{self.efg}\n")

    @property
    def nspins(self) -> int:
        """Number of spins in the spin system."""
        return len(self.isotopes)
    
    @property
    def spins(self) -> np.ndarray:
        """Spin quantum numbers for each isotope the spin system."""
        return np.array([ISOTOPES[isotope][0] for isotope in self.isotopes])
    
    @property
    def mults(self) -> np.ndarray:
        """
        Spin multiplicities for each isotope in the `SpinSystem`.
        """
        return np.array([int(2 * ISOTOPES[isotope][0] + 1)
                         for isotope in self.isotopes], dtype=int)
    
    @property
    def gammas(self) -> np.ndarray:
        """
        Gyromagnetic ratios of each isotope in the `SpinSystem` in rad/s/T.
        """
        return np.array([2 * np.pi * ISOTOPES[isotope][1] * 1e6
                         for isotope in self.isotopes])
    
    @property
    def quad(self) -> np.ndarray:
        """Returns the quadrupolar moments in m^2."""
        return np.array([ISOTOPES[isotope][2] * 1e-28
                         for isotope in self.isotopes])

    ########################
    # BASIS SET PROPERTIES #
    ########################

    @property
    def basis(self) -> Basis:
        """
        Contains the basis set for the `SpinSystem`. Includes functionality for
        restricting the maximum spin order, building the basis set, and applying
        more advanced truncation to the basis set.
        """
        return self._basis
    
    ################################
    # RELAXATION THEORY PROPERTIES #
    ################################

    @property
    def relaxation(self) -> RelaxationProperties:
        """
        Contains the properties that define the relaxation of the `SpinSystem`.
        Allows the definition of relaxation theory, correlation time, relaxation
        times, etc.
        """
        return self._relaxation