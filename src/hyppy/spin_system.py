"""
spin_system.py

Provides a class for the spin system. Once a spin system has been initialized,
the other modules can be used to calculate properties for the spin system.
"""

# Imports
from __future__ import annotations
import numpy as np
import hashlib
from pickle import dumps
from hyppy import data_io, nmr_isotopes
from hyppy.basis import Basis
from typing import Union

class SpinSystem:

    def __hash__(self):
        return self.basis.uid
    
    def __eq__(self, other: SpinSystem):
        return self.basis.uid == other.basis.uid

    def __init__(
        self, 
        isotopes: Union[np.ndarray, str], 
        chemical_shifts: Union[np.ndarray, str]=None, 
        scalar_couplings: Union[np.ndarray, str]=None,
        xyz: Union[np.ndarray, str]=None,
        shielding: Union[np.ndarray, str]=None,
        efg: Union[np.ndarray, str]=None,
        max_spin_order: int=None):
        """
        Initialization of the spin system.

        Parameters
        ----------
        isotopes : numpy.ndarray or str
            If a numpy.ndarray, includes the names of the isotopes. Example: np.array(['1H', '15N', '19F'])
            If a string, includes the file path to the data.
        
        chemical_shifts : numpy.ndarray or str
            If a numpy.ndarray, includes the chemical shifts in ppm. Example: np.array([8.00, -200, -130])
            If a string, includes the file path to the data.

        scalar_couplings : numpy.ndarray or str
            If a numpy.ndarray, has to be a 2D array of size (N, N) that includes the scalar couplings between nuclei in Hz.
            Only the bottom triangle is specified. Example:
            np.array([
                [0,    0,    0],
                [1,    0.5,  0],
                [0.2,  8,    0]])
            If a string, includes the file path to the data.

        xyz : numpy.ndarray or str
            If a numpy.ndarray, has to be a 2D array of size (N, 3) that includes the cartesian coordinates in Å.
            If a string, specifies the path to the .xyz file.

        shielding : numpy.ndarray or str
            If a numpy.ndarray, has to be a 3D array of size (N, 3, 3) that includes the 3x3 shielding tensors in ppm.
            If a string, specifies the path to the .txt file.

        efg : numpy.ndarray or str
            If a numpy.ndarray, has to be a 3D array of size (N, 3, 3) that includes the 3x3 EFG tensors in atomic units.
            If a string, specifies the path to the .txt file.

        max_spin_order : int
            Defines the maximum spin order that is taken into account.
            If left empty, spin order is set to the size of the system.
        """

        # Assign isotopes
        if isinstance(isotopes, np.ndarray):
            self.isotopes = isotopes
        elif isinstance(isotopes, str):
            self.isotopes = data_io.read_array(isotopes, data_type=str)
        else:
            raise TypeError(f"Isotopes should be a NumPy array or a string.")
        
        # Assign chemical shifts
        if isinstance(chemical_shifts, np.ndarray):
            self.chemical_shifts = chemical_shifts
        elif isinstance(chemical_shifts, str):
            self.chemical_shifts = data_io.read_array(chemical_shifts, data_type=float)
        elif chemical_shifts is None:
            self.chemical_shifts = np.zeros(self.size, dtype=float)
        else:
            raise TypeError(f"Chemical shifts should be a NumPy array or a string, or None.")

        # Assign scalar couplings
        if isinstance(scalar_couplings, np.ndarray):
            self.scalar_couplings = scalar_couplings
        elif isinstance(scalar_couplings, str):
            self.scalar_couplings = data_io.read_array(scalar_couplings, data_type=float)
        elif scalar_couplings is None:
            self.scalar_couplings = np.zeros((self.size, self.size), dtype=float)
        else:
            raise TypeError(f"Scalar couplings should be a NumPy array or a string, or None.")
        
        # Assign XYZ coordinates
        if isinstance(xyz, np.ndarray):
            self.xyz = xyz
        elif isinstance(xyz, str):
            self.xyz = data_io.read_xyz(xyz)
        elif xyz is None:
            self.xyz = None
        else:
            raise TypeError(f"XYZ should be a NumPy array or a string, or None.")
        
        # Assign shielding tensors
        if isinstance(shielding, np.ndarray):
            self.shielding = shielding
        elif isinstance(shielding, str):
            self.shielding = data_io.read_tensors(shielding)
        elif shielding is None:
            self.shielding = None
        else:
            raise TypeError(f"Shielding should be a NumPy array or a string, or None.")
        
        # Assign EFG tensors
        if isinstance(efg, np.ndarray):
            self.efg = efg
        elif isinstance(efg, str):
            self.efg = data_io.read_tensors(efg)
        elif efg is None:
            self.efg = None
        else:
            raise TypeError(f"EFG should be a NumPy array or a string, or None.")

        # Assign maximum spin order
        if isinstance(max_spin_order, int):
            if max_spin_order > self.isotopes.shape[0]:
                raise ValueError(f"Maximum spin order should not exceed the size of the spin system.")
            elif max_spin_order < 1:
                raise ValueError(f"Maximum spin order should be at least one.")
            else:
                self.max_spin_order = max_spin_order
        elif max_spin_order is None:
            self.max_spin_order = self.size

        # Check for consistent sizes in the arrays
        if self.chemical_shifts.size != self.isotopes.size:
            raise ValueError("Mismatch between the sizes of the isotopes and chemical shifts.")
        if not (self.scalar_couplings.shape[0] == self.scalar_couplings.shape[1] == self.isotopes.size):
            raise ValueError("Mismatch between the sizes of the isotopes and scalar couplings.")
        if self.xyz is not None and self.xyz.shape[0] != self.isotopes.size:
            raise ValueError("Mismatch between the sizes of the isotopes and xyz coordinates.")
        if self.shielding is not None and self.shielding.shape[0] != self.isotopes.size:
            raise ValueError("Mismatch between the sizes of the isotopes and shielding tensors.")
        if self.efg is not None and self.efg.shape[0] != self.isotopes.size:
            raise ValueError("Mismatch between the sizes of the isotopes and the efg tensors.")

        # Assign the basis set
        self.basis = Basis(self)

    @property
    def size(self) -> int:
        """Number of spins in the spin system."""
        return len(self.isotopes)
    
    @property
    def spins(self) -> np.ndarray:
        """Spin quantum numbers of the spin system."""
        return np.array([nmr_isotopes.ISOTOPES[isotope][0] for isotope in self.isotopes])
    
    @property
    def mults(self) -> np.ndarray:
        """Spin multiplicities of the spin system."""
        return np.array([int(2*nmr_isotopes.ISOTOPES[isotope][0] + 1) for isotope in self.isotopes], dtype=int)
    
    @property
    def gammas(self) -> np.ndarray:
        """Gyromagnetic ratios in rad/s/T."""
        return np.array([2*np.pi*nmr_isotopes.ISOTOPES[isotope][1] * 1e6 for isotope in self.isotopes])
    
    @property
    def quad(self) -> np.ndarray:
        """Quadrupolar moments in m^2."""
        return np.array([nmr_isotopes.ISOTOPES[isotope][2] * 1e-28 for isotope in self.isotopes])
    
    @property
    def unique_id(self) -> float:
        """Computes an unique ID for the spin system based on the properties."""
        properties_bytes = dumps(self.isotopes) \
                         + dumps(self.chemical_shifts) \
                         + dumps(self.scalar_couplings) \
                         + dumps(self.xyz) \
                         + dumps(self.shielding) \
                         + dumps(self.efg) \
                         + dumps(self.max_spin_order) \
                         + dumps(self.basis.arr)
        return hashlib.md5(properties_bytes).hexdigest()