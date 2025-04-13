"""
spin_system.py

Defines a class for the spin system. Once a spin system is initialized,
other modules can be used to calculate its properties.
"""

# Imports
from __future__ import annotations
import numpy as np
import hashlib
from pickle import dumps
from spinguin._nmr_isotopes import ISOTOPES
from spinguin._data_io import read_array, read_tensors, read_xyz
from spinguin._basis import Basis

class SpinSystem:

    def __hash__(self):
        return self.basis.uid
    
    def __eq__(self, other: SpinSystem):
        return self.basis.uid == other.basis.uid

    def __init__(
        self, 
        isotopes: np.ndarray | str, 
        chemical_shifts: np.ndarray | str = None, 
        J_couplings: np.ndarray | str = None,
        xyz: np.ndarray | str = None,
        shielding: np.ndarray | str = None,
        efg: np.ndarray | str = None,
        max_spin_order: int = None):
        """
        Initializes the spin system.

        Parameters
        ----------
        isotopes : numpy.ndarray or str
            Specifies the isotopes that constitute the spin system and determine other properties,
            such as spin quantum numbers and gyromagnetic ratios.

            - If a `numpy.ndarray`: A 1D array of size N containing isotope names as strings. Example:

            ```python
            np.array(['1H', '15N', '19F'])
            ```

            - If a `str`: Path to the file containing the isotopes.
        
        chemical_shifts : numpy.ndarray or str
            Chemical shifts arising from the isotropic component of the nuclear shielding tensors.
            Used when calculating the coherent Hamiltonian.

            - If a `numpy.ndarray`: A 1D array of size N containing the chemical shifts in ppm. Example:

            ```python
            np.array([8.00, -200, -130])
            ```

            - If a `str`: Path to the file containing the chemical shifts.

        J_couplings : numpy.ndarray or str
            Specifies the scalar coupling constants between each spin pair in the spin system. Used when
            calculating the coherent Hamiltonian.

            - If a `numpy.ndarray`: A 2D array of size (N, N) specifying the scalar couplings between nuclei in Hz.
            Only the lower triangle is specified. Example:

            ```python
            np.array([
                [0,    0,    0],
                [1,    0,    0],
                [0.2,  8,    0]
            ])
            ```

            - If a `str`: Path to the file containing the scalar couplings.

        xyz : numpy.ndarray or str
            Coordinates in the XYZ format for each nucleus in the spin system. Used in relaxation when calculating
            the dipole-dipole coupling tensors.  
        
            - If a `numpy.ndarray`: A 2D array of size (N, 3) containing the Cartesian coordinates in Å.
            - If a `str`: Path to the file containing the XYZ coordinates.

        shielding : numpy.ndarray or str
            Specifies the nuclear shielding tensors for each nucleus. Note that the isotropic part of the tensor
            is handled by `chemical_shifts`. The shielding tensors are used only for relaxation.

            - If a `numpy.ndarray`: A 3D array of size (N, 3, 3) containing the 3x3 shielding tensors in ppm.
            - If a `str`: Path to the file containing the shielding tensors.

        efg : numpy.ndarray or str
            Electric field gradient tensors used for incorporating the quadrupolar interaction relaxation
            mechanism.

            - If a `numpy.ndarray`: A 3D array of size (N, 3, 3) containing the 3x3 EFG tensors in atomic units.
            - If a `str`: Path to the file containing the EFG tensors.

        TODO: Merkin määrittely selväksi. (Perttu)

        max_spin_order : int
            Defines the maximum spin order included in the basis set.
            If left empty, the spin order is set to the size of the system.
        """

        # Assign isotopes
        if isinstance(isotopes, np.ndarray):
            self.isotopes = isotopes
        elif isinstance(isotopes, str):
            self.isotopes = read_array(isotopes, data_type=str)
        else:
            raise TypeError("Isotopes should be a NumPy array or a string.")
        
        # Assign chemical shifts
        if isinstance(chemical_shifts, np.ndarray):
            self.chemical_shifts = chemical_shifts
        elif isinstance(chemical_shifts, str):
            self.chemical_shifts = read_array(chemical_shifts, data_type=float)
        elif chemical_shifts is None:
            self.chemical_shifts = np.zeros(self.size, dtype=float)
        else:
            raise TypeError("Chemical shifts should be a NumPy array, a string, or None.")

        # Assign scalar couplings
        if isinstance(J_couplings, np.ndarray):
            self.J_couplings = J_couplings
        elif isinstance(J_couplings, str):
            self.J_couplings = read_array(J_couplings, data_type=float)
        elif J_couplings is None:
            self.J_couplings = np.zeros((self.size, self.size), dtype=float)
        else:
            raise TypeError("Scalar couplings should be a NumPy array, a string, or None.")
        
        # Assign XYZ coordinates
        if isinstance(xyz, np.ndarray):
            self.xyz = xyz
        elif isinstance(xyz, str):
            self.xyz = read_xyz(xyz)
        elif xyz is None:
            self.xyz = None
        else:
            raise TypeError("XYZ should be a NumPy array, a string, or None.")
        
        # Assign shielding tensors
        if isinstance(shielding, np.ndarray):
            self.shielding = shielding
        elif isinstance(shielding, str):
            self.shielding = read_tensors(shielding)
        elif shielding is None:
            self.shielding = None
        else:
            raise TypeError("Shielding should be a NumPy array, a string, or None.")
        
        # Assign EFG tensors
        if isinstance(efg, np.ndarray):
            self.efg = efg
        elif isinstance(efg, str):
            self.efg = read_tensors(efg)
        elif efg is None:
            self.efg = None
        else:
            raise TypeError("EFG should be a NumPy array, a string, or None.")

        # Assign maximum spin order
        if isinstance(max_spin_order, int):
            if max_spin_order > self.isotopes.shape[0]:
                raise ValueError("Maximum spin order should not exceed the size of the spin system.")
            elif max_spin_order < 1:
                raise ValueError("Maximum spin order should be at least one.")
            else:
                self.max_spin_order = max_spin_order
        elif max_spin_order is None:
            self.max_spin_order = self.size

        # Check for consistent sizes in the arrays
        if self.chemical_shifts.size != self.isotopes.size:
            raise ValueError("Mismatch between the sizes of the isotopes and chemical shifts.")
        if not (self.J_couplings.shape[0] == self.J_couplings.shape[1] == self.isotopes.size):
            raise ValueError("Mismatch between the sizes of the isotopes and scalar couplings.")
        if self.xyz is not None and self.xyz.shape[0] != self.isotopes.size:
            raise ValueError("Mismatch between the sizes of the isotopes and XYZ coordinates.")
        if self.shielding is not None and self.shielding.shape[0] != self.isotopes.size:
            raise ValueError("Mismatch between the sizes of the isotopes and shielding tensors.")
        if self.efg is not None and self.efg.shape[0] != self.isotopes.size:
            raise ValueError("Mismatch between the sizes of the isotopes and EFG tensors.")

        # Assign the basis set
        self.basis = Basis(self)

    @property
    def size(self) -> int:
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
    
    @property
    def unique_id(self) -> float:
        """Computes a unique ID for the spin system based on its properties."""
        properties_bytes = dumps(self.isotopes) \
                         + dumps(self.chemical_shifts) \
                         + dumps(self.J_couplings) \
                         + dumps(self.xyz) \
                         + dumps(self.shielding) \
                         + dumps(self.efg) \
                         + dumps(self.max_spin_order) \
                         + dumps(self.basis.arr)
        return hashlib.md5(properties_bytes).hexdigest()