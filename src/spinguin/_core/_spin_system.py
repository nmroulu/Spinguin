"""
spin_system.py

Defines the `SpinSystem` class, which stores isotope identities together with
the spin-resolved data required for Hamiltonian and relaxation calculations.
"""

# Imports
from __future__ import annotations

from typing import Callable, Self

import numpy as np

from spinguin._core._data_io import read_array, read_tensors, read_xyz
from spinguin._core._basis import Basis
from spinguin._core._la import arraylike_to_array
from spinguin._core._nmr_isotopes import ISOTOPES
from spinguin._core._relaxation_properties import RelaxationProperties
from spinguin._core._status import status


ArrayInput = list | tuple | np.ndarray | str
OptionalArray = np.ndarray | None


class SpinSystem:
    """
    Store the isotopes and associated data of a spin system.

    Examples::

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

        The input is converted and stored as a NumPy array.
    """

    def __init__(
        self,
        isotopes: ArrayInput,
    ) -> None:
        """
        Initialise a spin system from isotope labels.

        Parameters
        ----------
        isotopes : list or tuple or ndarray or str
            One-dimensional isotope labels, or a path to a file containing
            them.

        Returns
        -------
        None
            The instance is initialised in place.
        """

        # Store the isotope labels as a NumPy array.
        self._isotopes = self._read_spin_data(
            isotopes,
            name="Isotopes",
            ndim_description="1-dimensional",
            file_reader=read_array,
            data_type=str,
        )

        # Initialise the remaining spin-resolved properties.
        self._chemical_shifts = np.zeros(self.nspins)
        self._J_couplings = np.zeros((self.nspins, self.nspins))
        self._xyz: OptionalArray = None
        self._shielding: OptionalArray = None
        self._efg: OptionalArray = None

        # Validate the newly initialised object state.
        self._check_consistency()

        # Create the basis-set handler for the spin system.
        self._basis = Basis(self)

        # Create the relaxation settings for the spin system.
        self._relaxation = RelaxationProperties(self)

        # Report the newly created spin system.
        status("Spin system has been created with the following isotopes:")
        status(f"{self.isotopes}\n")

    def _read_spin_data(
        self,
        values: ArrayInput,
        name: str,
        ndim_description: str,
        file_reader: Callable,
        data_type: type | None=None,
    ) -> np.ndarray:
        """
        Convert file-based or array-like input to a NumPy array.

        Parameters
        ----------
        values : list or tuple or ndarray or str
            Input values or a path to a text file.
        name : str
            Name of the quantity for error reporting.
        ndim_description : str
            Human-readable description of the required array dimensionality.
        file_reader : callable
            Reader used when `values` is supplied as a file path.
        data_type : type or None, default=None
            Optional data type passed to `read_array`.

        Returns
        -------
        ndarray
            Converted NumPy array.

        Raises
        ------
        TypeError
            Raised if `values` is neither array-like nor a string.
        """

        # Read values from file when a path is supplied.
        if isinstance(values, str):
            if data_type is None:
                return file_reader(values)
            return file_reader(values, data_type=data_type)

        # Convert array-like input directly to a NumPy array.
        if isinstance(values, (list, tuple, np.ndarray)):
            return arraylike_to_array(values)

        # Reject unsupported input types explicitly.
        raise TypeError(
            f"{name} should be a {ndim_description} array or a string."
        )

    def _validate_optional_shape(
        self,
        values: OptionalArray,
        expected_shape: tuple[int, ...],
        error_message: str,
    ) -> None:
        """
        Validate the shape of an optional spin-resolved array.

        Parameters
        ----------
        values : ndarray or None
            Array to be validated. `None` is accepted without checks.
        expected_shape : tuple of int
            Shape required for the given quantity.
        error_message : str
            Error message raised when the shape does not match.

        Returns
        -------
        None
            Validation is performed for its side effect only.

        Raises
        ------
        ValueError
            Raised if `values` is not `None` and the shape is incorrect.
        """

        # Accept missing optional arrays without further validation.
        if values is None:
            return

        # Reject arrays whose shape does not match the expected one.
        if values.shape != expected_shape:
            raise ValueError(error_message)

    def _check_consistency(self) -> None:
        """
        Check that the stored spin-system data are internally consistent.

        Returns
        -------
        None
            Validation is performed for its side effect only.
        """

        # Check that the isotope labels form a one-dimensional array.
        if self.isotopes.ndim != 1:
            raise ValueError("Isotopes must be a 1D array containing the "
                             "names of the isotopes as strings.")

        # Check that every isotope label is defined in the isotope table.
        for isotope in self.isotopes:
            if isotope not in ISOTOPES:
                raise ValueError(f"Isotope '{isotope}' is not defined in the "
                                 "ISOTOPES dictionary.")

        # Check the isotropic chemical-shift array.
        self._validate_optional_shape(
            self.chemical_shifts,
            (self.nspins,),
            "Chemical shifts must be a 1D array with a length equal to the "
            "number of isotopes.",
        )

        # Check the scalar-coupling matrix.
        self._validate_optional_shape(
            self.J_couplings,
            (self.nspins, self.nspins),
            "J-couplings must be a 2D array with both dimensions equal to "
            "the number of isotopes.",
        )

        # Check the Cartesian coordinate array.
        self._validate_optional_shape(
            self.xyz,
            (self.nspins, 3),
            "XYZ coordinates must be a 2D array with the number of rows "
            "equal to the number of isotopes.",
        )

        # Check the shielding-tensor array.
        self._validate_optional_shape(
            self.shielding,
            (self.nspins, 3, 3),
            "Shielding tensors must be a 3D array with the number of 3x3 "
            "tensors equal to the number of isotopes.",
        )

        # Check the electric-field-gradient tensor array.
        self._validate_optional_shape(
            self.efg,
            (self.nspins, 3, 3),
            "EFG tensors must be a 3D array with the number of 3x3 tensors "
            "equal to the number of isotopes.",
        )

    def subsystem(
        self,
        spins: list,
    ) -> Self:
        """
        Create a new spin system from a selected subset of spins.

        The basis-set and relaxation settings are reinitialised for the new
        object rather than copied from the parent spin system.

        Parameters
        ----------
        spins : list
            List of spin indices to be retained in the subsystem.

        Returns
        -------
        SpinSystem
            New spin system containing only the selected spins.
        """

        # Check that each selected spin index is unique.
        if len(set(spins)) != len(spins):
            raise ValueError("Each spin must be unique in 'spins'")

        # Check that at least one spin has been selected.
        if len(spins) == 0:
            raise ValueError("'spins' cannot be empty")

        # Check that the selected indices exist in the spin system.
        if max(spins) >= self.nspins:
            raise ValueError(f"Spin system does not have spin: {max(spins)}")

        # Create the subsystem with the selected isotope labels.
        spin_system = SpinSystem(self.isotopes[spins])

        # Transfer the isotropic chemical shifts.
        if self.chemical_shifts is not None:
            spin_system.chemical_shifts = self.chemical_shifts[spins]

        # Transfer the scalar-coupling matrix.
        if self.J_couplings is not None:
            spin_system.J_couplings = self.J_couplings[np.ix_(spins, spins)]

        # Transfer the Cartesian coordinates.
        if self.xyz is not None:
            spin_system.xyz = self.xyz[spins]

        # Transfer the shielding tensors.
        if self.shielding is not None:
            spin_system.shielding = self.shielding[spins]

        # Transfer the electric-field-gradient tensors.
        if self.efg is not None:
            spin_system.efg = self.efg[spins]

        return spin_system

    ##########################
    # SPIN SYSTEM PROPERTIES #
    ##########################

    @property
    def isotopes(self) -> np.ndarray:
        """
        Isotope labels that define the spin system.

        Example::

            np.array(['1H', '15N', '19F'])

        The isotope array is defined during initialisation.
        """
        return self._isotopes

    @property
    def chemical_shifts(self) -> np.ndarray:
        """
        Isotropic chemical shifts used in the coherent Hamiltonian.

        - If `ArrayLike`: A 1D array of size N containing the chemical shifts
          in ppm.
        - If `str`: Path to the file containing the chemical shifts.

        Example::

            spin_system.chemical_shifts = [8.49, 7.78, 7.46]

        The input will be stored as a NumPy array.
        """
        return self._chemical_shifts

    @chemical_shifts.setter
    def chemical_shifts(
        self,
        chemical_shifts: ArrayInput,
    ) -> None:
        # Store the chemical-shift data.
        self._chemical_shifts = self._read_spin_data(
            chemical_shifts,
            name="Chemical shifts",
            ndim_description="1-dimensional",
            file_reader=read_array,
            data_type=float,
        )

        # Validate the updated object state.
        self._check_consistency()

        # Report the assigned chemical shifts.
        status("Assigned the following chemical shifts:")
        status(f"{self.chemical_shifts}\n")

    @property
    def J_couplings(self) -> np.ndarray:
        """
        Scalar J-coupling matrix used in the coherent Hamiltonian.

        Only the lower triangle needs to be specified.

        - If `ArrayLike`: A 2D array of size (N, N) specifying the scalar
          couplings between nuclei in Hz.
        - If `str`: Path to the file containing the scalar couplings.

        Example::

            spin_system.J_couplings = [
                [0,    0,    0],
                [1,    0,    0],
                [0.2,  8,    0]
            ]

        The input will be stored as a NumPy array.
        """
        return self._J_couplings

    @J_couplings.setter
    def J_couplings(
        self,
        J_couplings: ArrayInput,
    ) -> None:
        # Store the scalar-coupling matrix.
        self._J_couplings = self._read_spin_data(
            J_couplings,
            name="J-couplings",
            ndim_description="2-dimensional",
            file_reader=read_array,
            data_type=float,
        )

        # Validate the updated object state.
        self._check_consistency()

        # Report the assigned scalar couplings.
        status(f"Assigned the following J-couplings:\n{self.J_couplings}\n")

    @property
    def xyz(self) -> OptionalArray:
        """
        Cartesian coordinates of the nuclei in Å.

        The coordinates are used in Redfield relaxation theory when evaluating
        dipole-dipole coupling tensors.

        - If `ArrayLike`: A 2D array of size (N, 3).
        - If `str`: Path to the file containing the XYZ coordinates.

        Example::

            spin_system.xyz = [
                [1.025, 2.521, 1.624],
                [0.667, 2.754, 0.892]
            ]

        The input will be stored as a NumPy array.
        """
        return self._xyz

    @xyz.setter
    def xyz(
        self,
        xyz: ArrayInput,
    ) -> None:
        # Store the Cartesian coordinates.
        self._xyz = self._read_spin_data(
            xyz,
            name="XYZ",
            ndim_description="2-dimensional",
            file_reader=read_xyz,
        )

        # Validate the updated object state.
        self._check_consistency()

        # Report the assigned Cartesian coordinates.
        status(f"Assigned the following XYZ coordinates:\n{self.xyz}\n")

    @property
    def shielding(self) -> OptionalArray:
        """
        Nuclear shielding tensors for each spin.

        The isotropic contribution is handled separately through
        `chemical_shifts`. The full tensors are used only in Redfield
        relaxation theory.

        - If `ArrayLike`: A 3D array of size (N, 3, 3) containing the 3x3
          shielding tensors in ppm.
        - If `str`: Path to the file containing the shielding tensors.

        Example::

            spin_system.shielding = [
                [
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0]
                ],
                [
                    [101.6, -75.2, 11.1],
                    [30.5,   10.1, 87.4],
                    [99.7,  -21.1, 11.2]
                ]
            ]

        The input will be stored as a NumPy array.
        """
        return self._shielding

    @shielding.setter
    def shielding(
        self,
        shielding: ArrayInput,
    ) -> None:
        # Store the shielding tensors.
        self._shielding = self._read_spin_data(
            shielding,
            name="Shielding",
            ndim_description="3-dimensional",
            file_reader=read_tensors,
        )

        # Validate the updated object state.
        self._check_consistency()

        # Report the assigned shielding tensors.
        status(f"Assigned the following shielding tensors:\n{self.shielding}\n")

    @property
    def efg(self) -> OptionalArray:
        """
        Electric-field-gradient tensors for quadrupolar relaxation.

        - If `ArrayLike`: A 3D array of size (N, 3, 3) containing the 3x3 EFG
          tensors in atomic units.
        - If `str`: Path to the file containing the EFG tensors.

        Example::

            spin_system.efg = [
                [
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0]
                ],
                [
                    [ 0.31, 0.00, 0.01],
                    [-0.20, 0.04, 0.87],
                    [ 0.11, 0.16, 0.65]
                ]
            ]

        The input will be stored as a NumPy array.
        """
        return self._efg

    @efg.setter
    def efg(
        self,
        efg: ArrayInput,
    ) -> None:
        # Store the electric-field-gradient tensors.
        self._efg = self._read_spin_data(
            efg,
            name="EFG",
            ndim_description="3-dimensional",
            file_reader=read_tensors,
        )

        # Validate the updated object state.
        self._check_consistency()

        # Report the assigned electric-field-gradient tensors.
        status(f"Assigned the following EFG tensors:\n{self.efg}\n")

    @property
    def nspins(self) -> int:
        """
        Number of spins in the spin system.
        """

        return len(self.isotopes)

    @property
    def spins(self) -> np.ndarray:
        """
        Spin quantum numbers of the isotopes in the spin system.
        """

        return np.array([ISOTOPES[isotope][0] for isotope in self.isotopes])

    @property
    def mults(self) -> np.ndarray:
        """
        Spin multiplicities of the isotopes in the spin system.
        """

        return np.array([int(2 * ISOTOPES[isotope][0] + 1)
                         for isotope in self.isotopes], dtype=int)

    @property
    def gammas(self) -> np.ndarray:
        """
        Gyromagnetic ratios of the isotopes in rad/s/T.
        """

        return np.array([2 * np.pi * ISOTOPES[isotope][1] * 1e6
                         for isotope in self.isotopes])

    @property
    def quad(self) -> np.ndarray:
        """
        Quadrupolar moments of the isotopes in $\mathrm{m^2}$.
        """

        return np.array([ISOTOPES[isotope][2] * 1e-28
                         for isotope in self.isotopes])

    ########################
    # BASIS SET PROPERTIES #
    ########################

    @property
    def basis(self) -> Basis:
        """
        Basis-set handler associated with the spin system.

        The object provides the functionality required to restrict the maximum
        spin order, build the basis set, and apply more advanced truncation
        strategies.
        """

        return self._basis

    ################################
    # RELAXATION THEORY PROPERTIES #
    ################################

    @property
    def relaxation(self) -> RelaxationProperties:
        """
        Relaxation settings associated with the spin system.

        The object stores the relaxation-theory choice together with the
        corresponding correlation times, relaxation times, and related
        settings.
        """

        return self._relaxation