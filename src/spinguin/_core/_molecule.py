"""
Molecule container used in multiple Spinguin modules.

The `Molecule` class stores information about the
"parent" molecule of a spin system, such as isotope 
labels, Cartesian nuclear coordinates, and isotope masses.
It also describes the van der Waals solid of the molecule,
which defines the molecular volume and the ellipsoid used
in the automatic rotational correlation times.
"""

# Imports
import numpy as np

from spinguin._core._data_io import read_array, read_xyz
from spinguin._core._la import arraylike_to_array
from spinguin._core._nmr_isotopes import ISOTOPES

# Bondi van der Waals radii in Å (A. Bondi, J. Phys. Chem. 68, 441 (1964)).
VDW_RADII = {
    "H": 1.20,
    "He": 1.40,
    "Li": 1.82,
    "C": 1.70,
    "N": 1.55,
    "O": 1.52,
    "F": 1.47,
    "Ne": 1.54,
    "Na": 2.27,
    "Mg": 1.73,
    "Si": 2.10,
    "P": 1.80,
    "S": 1.80,
    "Cl": 1.75,
    "Ar": 1.88,
    "K": 2.75,
    "Se": 1.90,
    "Br": 1.85,
    "Kr": 2.02,
    "I": 1.98,
    "Xe": 2.16,
}

# Edge length of the cubic voxels used to sample the van der Waals solid in Å.
VDW_GRID_SPACING = 0.1


def _sample_vdw_solid(
    xyz: np.ndarray,
    radii: np.ndarray,
    grid_spacing: float,
) -> tuple[float, np.ndarray]:
    """
    Sample the van der Waals solid of a molecule on a uniform Cartesian grid.

    The solid is the union of atomic spheres. Each voxel inside the solid
    carries the same weight, so that the solid is treated as a body of uniform
    density.

    Parameters
    ----------
    xyz : ndarray
        Cartesian coordinates of the atoms in Å with shape `(N, 3)`.
    radii : ndarray
        Van der Waals radii of the atoms in Å.
    grid_spacing : float
        Edge length of the cubic voxels in Å.

    Returns
    -------
    volume : float
        Volume of the solid in Å^3.
    covariance : ndarray
        Second central moments of the solid per unit volume in Å^2, with
        shape `(3, 3)`.
    """

    # Construct the voxel-centre axes of a grid enclosing all atomic spheres.
    lower = np.min(xyz - radii[:, np.newaxis], axis=0)
    upper = np.max(xyz + radii[:, np.newaxis], axis=0)
    axes = [
        np.arange(lower[k] + grid_spacing / 2, upper[k], grid_spacing)
        for k in range(3)
    ]
    inside = np.zeros([len(axis) for axis in axes], dtype=bool)

    # Mark the voxels inside each atomic sphere within its bounding box.
    for centre, radius in zip(xyz, radii):
        start = [np.searchsorted(axes[k], centre[k] - radius) for k in range(3)]
        stop = [
            np.searchsorted(axes[k], centre[k] + radius, side="right")
            for k in range(3)
        ]
        d2 = [(axes[k][start[k]:stop[k]] - centre[k])**2 for k in range(3)]
        inside[start[0]:stop[0], start[1]:stop[1], start[2]:stop[2]] |= (
            d2[0][:, np.newaxis, np.newaxis]
            + d2[1][np.newaxis, :, np.newaxis]
            + d2[2][np.newaxis, np.newaxis, :]
            <= radius**2
        )

    # Count the voxels inside the solid along each Cartesian axis.
    nvoxels = np.sum(inside)
    counts = [
        np.sum(inside, axis=(1, 2)),
        np.sum(inside, axis=(0, 2)),
        np.sum(inside, axis=(0, 1)),
    ]

    # Calculate the first and second moments of the solid per voxel.
    mean = np.array([axes[k] @ counts[k] for k in range(3)]) / nvoxels
    moments = np.diag([(axes[k]**2) @ counts[k] for k in range(3)])
    moments[0, 1] = axes[0] @ np.sum(inside, axis=2) @ axes[1]
    moments[0, 2] = axes[0] @ np.sum(inside, axis=1) @ axes[2]
    moments[1, 2] = axes[1] @ np.sum(inside, axis=0) @ axes[2]
    moments = np.triu(moments) + np.triu(moments, 1).T
    moments /= nvoxels

    # Convert the moments to the central second moments and the volume.
    covariance = moments - np.outer(mean, mean)
    volume = nvoxels * grid_spacing**3

    return volume, covariance


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

        # Use the default van der Waals radii until custom ones are assigned.
        self._vdw_radii = None
        self._vdw_solid_cache = None

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

    @property
    def elements(self) -> np.ndarray:
        """
        Element symbols of the atoms, obtained from the isotope labels.

        Returns
        -------
        elements : ndarray
            One-dimensional array of element symbols.
        """

        # Remove the mass numbers from the isotope labels.
        return np.array(
            [isotope.lstrip("0123456789") for isotope in self.isotopes]
        )

    @property
    def vdw_radii(self) -> np.ndarray:
        """
        Van der Waals radii of the atoms in Å.

        The Bondi radii in `VDW_RADII` are used by default. Custom radii can be
        assigned, for example for elements that are not tabulated::

            molecule.vdw_radii = [2.00, 1.70, 1.20, 1.20]

        Returns
        -------
        vdw_radii : ndarray
            One-dimensional array of van der Waals radii.

        Raises
        ------
        ValueError
            Raised if a default radius is requested for an element that is not
            tabulated in `VDW_RADII`.
        """

        # Return the custom radii when they have been assigned.
        if self._vdw_radii is not None:
            return self._vdw_radii

        # Confirm that every element has a tabulated radius.
        missing = sorted(set(self.elements) - set(VDW_RADII))
        if missing:
            raise ValueError(
                f"No default van der Waals radius for: {', '.join(missing)}. "
                "Assign the radii of all atoms to Molecule.vdw_radii."
            )

        # Look up the Bondi radii of the elements.
        return np.array([VDW_RADII[element] for element in self.elements])

    @vdw_radii.setter
    def vdw_radii(self, vdw_radii: list | tuple | np.ndarray) -> None:
        # Convert the radii into a floating-point array.
        vdw_radii = np.asarray(arraylike_to_array(vdw_radii), dtype=float)

        # Confirm that each atom has exactly one radius.
        if vdw_radii.shape != (len(self.isotopes),):
            raise ValueError(
                "The number of van der Waals radii must match the number of "
                "atoms."
            )

        # Store the radii and discard the solid sampled with the old radii.
        self._vdw_radii = vdw_radii
        self._vdw_solid_cache = None

    def _vdw_solid(self) -> tuple[float, np.ndarray]:
        """
        Return the volume and the central second moments of the van der Waals
        solid, sampling the solid only once.

        Returns
        -------
        volume : float
            Volume of the solid in Å^3.
        covariance : ndarray
            Second central moments of the solid per unit volume in Å^2.
        """

        # Sample the solid on the first request and reuse the result.
        if self._vdw_solid_cache is None:
            self._vdw_solid_cache = _sample_vdw_solid(
                self.xyz, self.vdw_radii, VDW_GRID_SPACING
            )

        return self._vdw_solid_cache

    def _principal_moments(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Diagonalise the inertia tensor of the uniform-density van der Waals
        solid per unit mass.

        Returns
        -------
        moments : ndarray
            Principal moments in Å^2 in ascending order.
        axes : ndarray
            Principal axes as the columns of a 3x3 array, in the order of
            `moments`.
        """

        # Construct the inertia tensor from the central second moments.
        _, covariance = self._vdw_solid()
        inertia = np.trace(covariance) * np.eye(3) - covariance

        return np.linalg.eigh(inertia)

    @property
    def volume(self) -> float:
        """
        Van der Waals volume of the molecule.

        The volume is that of the union of atomic spheres with the radii in
        `vdw_radii`, evaluated on a uniform grid with the voxel edge length
        `VDW_GRID_SPACING`.

        Returns
        -------
        volume : float
            Volume of the molecule (in Å^3).
        """

        # Obtain the volume of the sampled van der Waals solid.
        volume, _ = self._vdw_solid()

        return volume

    @property
    def hydrodynamic_radius(self) -> float:
        """
        Hydrodynamic radius of the molecule, defined as the radius of the
        sphere that has the van der Waals volume of the molecule.

        Returns
        -------
        hydrodynamic_radius : float
            Hydrodynamic radius of the molecule (in Å).
        """

        # Approximate the molecule as a sphere of equal volume.
        r = (3/4 * self.volume / np.pi)**(1/3)

        return r

    @property
    def principal_axes(self) -> np.ndarray:
        """
        Principal axes of the inertia tensor of the van der Waals solid of
        uniform density. They define the frame of anisotropic rotational
        diffusion.

        Returns
        -------
        principal_axes : ndarray
            Unit vectors of the principal axes as the columns of a 3x3 array,
            in ascending order of the principal moments. The axes form a
            right-handed frame, so that they can be reached from the
            laboratory frame by a proper rotation.
        """

        # Obtain the eigenvectors of the inertia tensor.
        _, axes = self._principal_moments()

        # Reverse the last axis if needed to obtain a right-handed frame.
        if np.linalg.det(axes) < 0:
            axes[:, 2] = -axes[:, 2]

        return axes

    @property
    def semi_axes(self) -> np.ndarray:
        """
        Semi-axes of the ellipsoid that represents the molecule in Perrin
        theory.

        The axis ratios are those of the ellipsoid that has the inertia tensor
        of the uniform van der Waals solid. For a uniform ellipsoid of mass
        ``M``, the principal moments are ``I_x = M (a_y^2 + a_z^2) / 5`` and
        cyclic permutations. The ellipsoid is scaled to the van der Waals
        volume, so that the ellipsoid of a spherical top is the sphere of
        `hydrodynamic_radius`.

        Returns
        -------
        semi_axes : ndarray
            Semi-axes in Å along `principal_axes`.
        """

        # Invert the principal moments of a uniform ellipsoid for the semi-axes.
        moments, _ = self._principal_moments()
        semi_axes = np.sqrt(np.maximum(5/2 * (np.sum(moments) - 2*moments), 0))

        # Scale the ellipsoid to the van der Waals volume.
        semi_axes *= (3/4 * self.volume / np.pi / np.prod(semi_axes))**(1/3)

        return semi_axes
