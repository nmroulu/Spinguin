"""
_molecule.py

Defines a Molecule class, which can be assigned as a part of the
RelaxationProperties of a SpinSystem.
"""

# Imports
import numpy as np
from spinguin._core._data_io import read_array, read_xyz
from spinguin._core._la import arraylike_to_array
from spinguin._core._nmr_isotopes import ISOTOPES

class Molecule:
    """
    Initializes a molecule with the given `isotopes` and `xyz`. Examples::

        molecule = Molecule(
            isotopes = ['1H', '15N', '19F'],
            xyz = [
                [1.0527, 2.2566, 0.9925],
                [0.0014, 1.5578, 2.1146],
                [1.3456, 0.3678, 1.4251]
            ]
        )
        molecule = Molecule(
            isotopes = "/path/to/isotopes.txt",
            xyz = "/path/to/xyz.txt"
        )

    Parameters
    ----------
    isotopes : list or tuple or ndarray or str
        Specifies the isotopes in the molecule.

        Two input types are supported:

        - If `ArrayLike`: A 1D array of size N containing isotope names as
          strings. 
        - If `str`: Path to the file containing the isotopes.
    
    xyz : list or tuple or ndarray or str
        Coordinates in the XYZ format for each nucleus in the molecule.
    
        - If `ArrayLike`: A 2D array of size (N, 3) containing the Cartesian
          coordinates in Å.
        - If `str`: Path to the file containing the XYZ coordinates.
    """

    def __init__(
        self, 
        isotopes: list | tuple | np.ndarray | str,
        xyz : list | tuple | np.ndarray | str
    ):

        # Assign the isotopes
        if isinstance(isotopes, str):
            self._isotopes = read_array(isotopes, data_type=str)
        elif isinstance(isotopes, (list, tuple, np.ndarray)):
            self._isotopes = arraylike_to_array(isotopes)
        else:
            raise TypeError("Isotopes should be a 1-dimensional array or a "
                            "string.")
        
        # Assign the XYZ
        if isinstance(xyz, str):
            self._xyz = read_xyz(xyz)
        elif isinstance(xyz, (list, tuple, np.ndarray)):
            self._xyz = arraylike_to_array(xyz)
        else:
            raise TypeError("XYZ should be a 2-dimensional array or a string.")
        
        # Ensure that the input is consistent
        if self.xyz.shape != (len(self.isotopes), 3):
            raise ValueError("Mismatch between the assigned isotopes and XYZ.")

    @property
    def isotopes(self) -> np.ndarray:
        """
        Specifies the isotopes in the molecule. They are set during the
        initialization of the molecule.
        """
        return self._isotopes

    @property
    def xyz(self) -> np.ndarray:
        """
        Coordinates in the XYZ format for each isotope in the molecule. They
        are set during the initialization of the molecule.
        """
        return self._xyz
    
    @property
    def masses(self) -> np.ndarray:
        """Atomic masses of each isotope in atomic mass units (u)."""
        return np.array([ISOTOPES[isotope][3] for isotope in self.isotopes])