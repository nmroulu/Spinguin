"""
Molecule container used in multiple Spinguin modules.

The `Molecule` class stores information about the
"parent" molecule of a spin system, such as isotope 
labels, Cartesian nuclear coordinates, and isotope masses.
"""

# Imports
import numpy as np

from spinguin._core._data_io import read_array, read_xyz
from spinguin._core._la import arraylike_to_array
from spinguin._core._nmr_isotopes import ISOTOPES


class Molecule:
    """
    Store various properties of a molecule that are relevant for
    spin dynamics simulations.

    .. automethod:: __init__
    """

    def __init__(
        self,
        isotopes: list | tuple | np.ndarray | str,
        xyz: list | tuple | np.ndarray | str,
    ):
        """
        Initialise the molecule from isotope labels and coordinates.

        Parameters
        ----------
        isotopes : list or tuple or ndarray or str
            Isotope specification.

            Supported inputs are:

            - `ArrayLike`: one-dimensional array containing isotope names.
            - `str`: path to a file containing the isotope names.

        xyz : list or tuple or ndarray or str
            Cartesian coordinates for the nuclei.

            Supported inputs are:

            - `ArrayLike`: two-dimensional array of shape `(N, 3)` containing
            the coordinates in Å.
            - `str`: path to an XYZ file containing the coordinates.

        Examples
        --------
        Construct a molecule directly from array-like inputs::

            molecule = Molecule(
                isotopes=['1H', '15N', '19F'],
                xyz=[
                    [1.0527, 2.2566, 0.9925],
                    [0.0014, 1.5578, 2.1146],
                    [1.3456, 0.3678, 1.4251]
                ]
            )

        Construct a molecule from text files::

            molecule = Molecule(
                isotopes="/path/to/isotopes.txt",
                xyz="/path/to/xyz.txt"
            )

        Raises
        ------
        ValueError
            Raised if the number of isotopes and coordinates is inconsistent.
        """

        # Read or convert the isotope labels.
        self._isotopes = self._parse_isotopes(isotopes)

        # Read or convert the Cartesian coordinates.
        self._xyz = self._parse_xyz(xyz)

        # Confirm that each isotope has exactly one Cartesian coordinate.
        if self.xyz.shape != (len(self.isotopes), 3):
            raise ValueError(
                "Shape mismatch between isotopes and xyz coordinates."
            )

    @staticmethod
    def _parse_isotopes(
        isotopes: list | tuple | np.ndarray | str,
    ) -> np.ndarray:
        """
        Parse isotope labels from an array-like object or a file path.

        Parameters
        ----------
        isotopes : list or tuple or ndarray or str
            Isotope labels or a path to a file containing them.

        Returns
        -------
        isotopes : ndarray
            One-dimensional array of isotope labels.

        Raises
        ------
        TypeError
            Raised if the input type is unsupported.
        """

        # Accept either a file path or an in-memory array-like object.
        if isinstance(isotopes, str):
            return read_array(isotopes, data_type=str)

        if isinstance(isotopes, (list, tuple, np.ndarray)):
            return arraylike_to_array(isotopes)

        raise TypeError(
            "Isotopes must be a one-dimensional array-like object or "
            "a string."
        )

    @staticmethod
    def _parse_xyz(
        xyz: list | tuple | np.ndarray | str,
    ) -> np.ndarray:
        """
        Parse Cartesian coordinates from an array-like object or file path.

        Parameters
        ----------
        xyz : list or tuple or ndarray or str
            Cartesian coordinates or a path to an XYZ file.

        Returns
        -------
        xyz : ndarray
            Coordinate array.

        Raises
        ------
        TypeError
            Raised if the input type is unsupported.
        """

        # Accept either an XYZ file path or an in-memory coordinate array.
        if isinstance(xyz, str):
            return read_xyz(xyz)

        if isinstance(xyz, (list, tuple, np.ndarray)):
            return arraylike_to_array(xyz)

        raise TypeError(
            "xyz must be a two-dimensional array-like object or a string."
        )

    @property
    def isotopes(self) -> np.ndarray:
        """
        Isotope labels assigned to the molecule.

        Returns
        -------
        isotopes : ndarray
            One-dimensional array of isotope labels.
        """

        return self._isotopes

    @property
    def xyz(self) -> np.ndarray:
        """
        Cartesian coordinates assigned to the molecule.

        Returns
        -------
        xyz : ndarray
            Array of Cartesian coordinates with shape `(N, 3)`.
        """

        return self._xyz

    @property
    def masses(self) -> np.ndarray:
        """
        Atomic masses of the isotopes in atomic mass units.

        Returns
        -------
        masses : ndarray
            Atomic masses corresponding to `isotopes`.
        """

        # Look up the isotope masses in the isotope database.
        return np.array([ISOTOPES[isotope][3] for isotope in self.isotopes])