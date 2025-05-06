"""
spin_system.py

Defines a class for the spin system. Once a spin system is initialized,
other modules can be used to calculate its properties.
"""

# Imports
from __future__ import annotations
import numpy as np
import hashlib
from numpy.typing import ArrayLike
from pickle import dumps
from spinguin.utils.nmr_isotopes import ISOTOPES
from spinguin.utils.data_io import read_array, read_tensors, read_xyz
from spinguin.system.basis import Basis
from spinguin.utils.la import arraylike_to_array

class SpinSystem:

    def __hash__(self):
        return self.basis.uid
    
    def __eq__(self, other: SpinSystem):
        return self.basis.uid == other.basis.uid

    def __init__(
        self, 
        isotopes: ArrayLike | str, 
        chemical_shifts: ArrayLike | str = None, 
        J_couplings: ArrayLike | str = None,
        xyz: ArrayLike | str = None,
        shielding: ArrayLike | str = None,
        efg: ArrayLike | str = None,
        tau_c: float = None,
        max_spin_order: int = None):
        """
        Initializes the spin system.

        Parameters
        ----------
        isotopes : ArrayLike or str
            Specifies the isotopes that constitute the spin system and determine other properties,
            such as spin quantum numbers and gyromagnetic ratios.

            - If `ArrayLike`: A 1D array of size N containing isotope names as strings. Example:

            ```python
            np.array(['1H', '15N', '19F'])
            ```

            - If `str`: Path to the file containing the isotopes.
        
        chemical_shifts : ArrayLike or str
            Chemical shifts arising from the isotropic component of the nuclear shielding tensors.
            Used when calculating the coherent Hamiltonian. If left empty, zero ppm is assumed for
            each spin in the system.

            - If `ArrayLike`: A 1D array of size N containing the chemical shifts in ppm. Example:

            ```python
            np.array([8.00, -200, -130])
            ```

            - If `str`: Path to the file containing the chemical shifts.

        J_couplings : ArrayLike or str
            Specifies the scalar coupling constants between each spin pair in the spin system. Used when
            calculating the coherent Hamiltonian.

            - If `ArrayLike`: A 2D array of size (N, N) specifying the scalar couplings between nuclei in Hz.
            Only the lower triangle is specified. Example:

            ```python
            np.array([
                [0,    0,    0],
                [1,    0,    0],
                [0.2,  8,    0]
            ])
            ```

            - If `str`: Path to the file containing the scalar couplings.

        xyz : ArrayLike or str
            Coordinates in the XYZ format for each nucleus in the spin system. Used in relaxation when calculating
            the dipole-dipole coupling tensors.  
        
            - If `ArrayLike`: A 2D array of size (N, 3) containing the Cartesian coordinates in Å.
            - If `str`: Path to the file containing the XYZ coordinates.

        shielding : ArrayLike or str
            Specifies the nuclear shielding tensors for each nucleus. Note that the isotropic part of the tensor
            is handled by `chemical_shifts`. The shielding tensors are used only for relaxation.

            - If `ArrayLike`: A 3D array of size (N, 3, 3) containing the 3x3 shielding tensors in ppm.
            - If `str`: Path to the file containing the shielding tensors.

        efg : ArrayLike or str
            Electric field gradient tensors used for incorporating the quadrupolar interaction relaxation
            mechanism.

            - If `ArrayLike`: A 3D array of size (N, 3, 3) containing the 3x3 EFG tensors in atomic units.
            - If `str`: Path to the file containing the EFG tensors.

        tau_c : float
            Isotropic rotational correlation time in units of s. Must be defined to use
            the Redfield relaxation theory.

        TODO: Merkin määrittely selväksi. (Perttu)

        max_spin_order : int
            Defines the maximum spin order included in the basis set.
            If left empty, the spin order is set to the size of the system.
        """

        # Assign isotopes
        if isinstance(isotopes, str):
            self.isotopes = read_array(isotopes, data_type=str)
        else:
            try:
                self.isotopes = arraylike_to_array(isotopes)
            except Exception:
                raise TypeError("Isotopes should be a 1-dimensional array or a string.")
        
        # Assign chemical shifts
        if isinstance(chemical_shifts, str):
            self.chemical_shifts = read_array(chemical_shifts, data_type=float)
        elif chemical_shifts is None:
            self.chemical_shifts = np.zeros(self.size, dtype=float)
        else:
            try:
                self.chemical_shifts = arraylike_to_array(chemical_shifts)
            except Exception:
                raise TypeError("Chemical shifts should be a 1-dimensional array, a string, or None.")

        # Assign scalar couplings
        if isinstance(J_couplings, str):
            self.J_couplings = read_array(J_couplings, data_type=float)
        elif J_couplings is None:
            self.J_couplings = np.zeros((self.size, self.size), dtype=float)
        else:
            try:
                self.J_couplings = arraylike_to_array(J_couplings)
            except Exception:
                raise TypeError("J-couplings should be a 2-dimensional array, a string, or None.")
        
        # Assign XYZ coordinates
        if isinstance(xyz, str):
            self.xyz = read_xyz(xyz)
        elif xyz is None:
            self.xyz = None
        else:
            try:
                self.xyz = arraylike_to_array(xyz)
            except Exception:
                raise TypeError("XYZ should be a 2-dimensional array, a string, or None.")
        
        # Assign shielding tensors
        if isinstance(shielding, str):
            self.shielding = read_tensors(shielding)
        elif shielding is None:
            self.shielding = None
        else:
            try:
                self.shielding = arraylike_to_array(shielding)
            except Exception:
                raise TypeError("Shielding should be a 3-dimensional array, a string, or None.")
        
        # Assign EFG tensors
        if isinstance(efg, str):
            self.efg = read_tensors(efg)
        elif efg is None:
            self.efg = None
        else:
            try:
                self.efg = arraylike_to_array(efg)
            except Exception:
                raise TypeError("EFG should be a 3-dimensional array, a string, or None.")
        
        # Assign correlation time
        if isinstance(tau_c, float) or tau_c is None:
            self.tau_c = tau_c
        else:
            raise TypeError("Correlation time tau_c must be a float or None.")

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