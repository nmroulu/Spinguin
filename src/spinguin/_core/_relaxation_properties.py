"""
Relaxation-property definitions attached to a spin system.

The module provides the `RelaxationProperties` class, which stores the
relaxation-theory settings associated with a `SpinSystem` instance. The
properties may be accessed as follows::

    import spinguin as sg
    spin_system = sg.SpinSystem(["1H"])
    spin_system.relaxation.theory = "redfield"
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

from spinguin._core._data_io import read_array
from spinguin._core._la import arraylike_to_array
from spinguin._core._molecule import Molecule
from spinguin._core._relaxation import (
    rotational_correlation_time_SED,
    rotational_correlation_times_Perrin,
)
from spinguin._core._status import status

if TYPE_CHECKING:
    from spinguin._core._spin_system import SpinSystem


class RelaxationProperties:
    """
    Store the relaxation settings associated with a spin system.

    Usage: ``RelaxationProperties(spin_system)``.

    The class stores the theory selection and auxiliary parameters required
    for Redfield and phenomenological relaxation calculations.
    """

    # Store the default relaxation settings at class level.
    _antisymmetric: bool = False
    _dynamic_frequency_shift: bool = False
    _relative_error: float = 1e-6
    _sr2k: bool = False
    _tau_c: float | np.ndarray | None = None
    _theory: Literal["redfield", "phenomenological"] | None = None
    _thermalization: bool = False
    _T1: np.ndarray | None = None
    _T2: np.ndarray | None = None
    _molecule: Molecule | None = None

    def __init__(
        self,
        spin_system: SpinSystem,
    ) -> None:
        """
        Initialise the relaxation settings for a spin system.

        Usage: ``RelaxationProperties(spin_system)``.

        Parameters
        ----------
        spin_system : SpinSystem
            Spin system to which the relaxation settings belong.

        Returns
        -------
        None
            The instance is initialised in place.
        """

        # Store a reference to the parent spin system.
        self._spin_system = spin_system

    def _convert_relaxation_array(
        self,
        values: list | tuple | np.ndarray | str,
        name: str,
    ) -> np.ndarray:
        """
        Convert file or array-like relaxation data to a NumPy array.

        Usage: ``self._convert_relaxation_array(values, name)``.

        Parameters
        ----------
        values : list or tuple or ndarray or str
            Input relaxation data or a path to a text file.
        name : str
            Name of the relaxation quantity for error reporting.

        Returns
        -------
        values : ndarray
            One-dimensional NumPy array of floating-point values.
        """

        # Read relaxation data from a text file when a path is provided.
        if isinstance(values, str):
            values = read_array(values, data_type=float)

        # Convert array-like relaxation data to a NumPy array.
        elif isinstance(values, (list, tuple, np.ndarray)):
            values = arraylike_to_array(values)

        # Reject unsupported input types explicitly.
        else:
            raise TypeError(
                f"{name} should be a one-dimensional array or a string."
            )

        return values

    def _validate_positive_spin_array(
        self,
        values: np.ndarray,
        name: str,
    ) -> None:
        """
        Validate the shape and positivity of spin-resolved input data.

        Usage: ``self._validate_positive_spin_array(values, name)``.

        Parameters
        ----------
        values : ndarray
            Array to validate.
        name : str
            Name of the relaxation quantity for error reporting.

        Returns
        -------
        None
            The input is validated in place.
        """

        # Check that the array length matches the number of spins.
        if values.shape != self._spin_system.isotopes.shape:
            raise ValueError(
                f"Mismatch between the given {name} values and the "
                "number of spins in the system."
            )

        # Check that all relaxation values are strictly positive.
        if np.min(values) <= 0:
            raise ValueError(f"{name} cannot be zero or negative.")

    @property
    def antisymmetric(
        self,
    ) -> bool:
        """
        Return whether antisymmetric interaction tensors are included.

        This option applies to Redfield relaxation theory. The default value is
        ``False``.
        """

        return self._antisymmetric

    @antisymmetric.setter
    def antisymmetric(
        self,
        antisymmetric: bool,
    ) -> None:
        # Store the antisymmetric-interaction setting.
        self._antisymmetric = antisymmetric

        # Report the updated antisymmetric-interaction setting.
        status(
            "Antisymmetric part of the interaction tensors set to: "
            f"{self.antisymmetric}\n"
        )

    @property
    def dynamic_frequency_shift(
        self,
    ) -> bool:
        """
        Return whether the dynamic frequency shift is included.

        This option applies to Redfield relaxation theory and corresponds to the
        imaginary part of the relaxation superoperator. The default value is
        ``False``.
        """

        return self._dynamic_frequency_shift

    @dynamic_frequency_shift.setter
    def dynamic_frequency_shift(
        self,
        dynamic_frequency_shift: bool,
    ) -> None:
        # Store the dynamic-frequency-shift setting.
        self._dynamic_frequency_shift = dynamic_frequency_shift

        # Report the updated dynamic-frequency-shift setting.
        status(
            "Dynamic frequency shift set to: "
            f"{self.dynamic_frequency_shift}\n"
        )

    @property
    def relative_error(
        self,
    ) -> float:
        """
        Return the relative error used in Redfield calculations.

        This value acts as the convergence criterion for the Redfield integral.
        The default value is ``1e-6``.
        """

        return self._relative_error

    @relative_error.setter
    def relative_error(
        self,
        relative_error: float,
    ) -> None:
        # Store the Redfield relative-error threshold.
        self._relative_error = relative_error

        # Report the updated Redfield relative-error threshold.
        status(f"Relative error set to: {self.relative_error}\n")

    @property
    def sr2k(
        self,
    ) -> bool:
        """
        Return whether scalar relaxation of the second kind is included.

        The default value is ``False``.
        """

        return self._sr2k

    @sr2k.setter
    def sr2k(
        self,
        sr2k: bool,
    ) -> None:
        # Store the SR2K setting.
        self._sr2k = sr2k

        # Report the updated SR2K setting.
        status(f"SR2K set to: {self.sr2k}\n")

    @property
    def tau_c(
        self,
    ) -> float | np.ndarray | None:
        """
        Return the rotational correlation time or times.

        The value is ``None`` until it is assigned explicitly or through
        :meth:`auto_tau_c`.

        For isotropic rotational diffusion, a single value is used. Example::

            spin_system.relaxation.tau_c = 50e-12

        For anisotropic rotational diffusion, an array of three values is used,
        corresponding to the principal components of the diffusion tensor.
        Example::

            spin_system.relaxation.tau_c = [50e-12, 100e-12, 150e-12]
        """

        return self._tau_c

    @tau_c.setter
    def tau_c(
        self,
        tau_c: float | list[float] | tuple[float, ...] | np.ndarray,
    ) -> None:
        # Store a single isotropic correlation time.
        if isinstance(tau_c, (float, int)):
            self._tau_c = float(tau_c)

        # Store three anisotropic principal-axis correlation times.
        elif isinstance(tau_c, (list, tuple)) and len(tau_c) == 3:
            self._tau_c = np.array([float(value) for value in tau_c])

        # Store an anisotropic correlation-time array directly.
        elif isinstance(tau_c, np.ndarray) and tau_c.shape == (3,):
            self._tau_c = tau_c.astype(float)

        # Reject unsupported correlation-time input formats.
        else:
            raise ValueError(
                "tau_c must be either a single float for isotropic rotational "
                "diffusion or three values for anisotropic rotational diffusion."
            )

        # Report the updated rotational correlation time or times.
        status(f"Rotational correlation time(s) set to: {self.tau_c}\n")

    def auto_tau_c(
        self,
        T: float,
        eta: float,
        model: Literal["iso", "aniso"] = "iso",
        r: float | None = None,
        scaling_factor: float = 1.0,
    ) -> None:
        """
        Automatically assign rotational correlation time or times.

        Usage: ``auto_tau_c(T, eta, model="iso", r=None, scaling_factor=1.0)``.

        The correlation time is evaluated either from the
        Stokes-Einstein-Debye relation for isotropic rotational diffusion or
        from Perrin theory for anisotropic rotational diffusion.

        Parameters
        ----------
        T : float
            Temperature in kelvin.
        eta : float
            Viscosity of the solvent in pascal-seconds.
        model : {"iso", "aniso"}, default="iso"
            Rotational-diffusion model used in the calculation.
        r : float, default=None
            Effective hydrodynamic radius in metres. Required for the isotropic
            model.
        scaling_factor : float, default=1.0
            Scaling factor applied to the calculated correlation times.

        Returns
        -------
        None
            The calculated value is stored in ``tau_c``.

        Raises
        ------
        ValueError
            Raised if the required model-specific inputs are missing or if
            ``model`` is invalid.
        """

        # Use the isotropic Stokes-Einstein-Debye relation when requested.
        if model == "iso":
            if r is None:
                raise ValueError(
                    "Hydrodynamic radius 'r' must be provided for the isotropic "
                    "model."
                )
            self.tau_c = (
                rotational_correlation_time_SED(T, eta, r, 2)
                * scaling_factor
            )

        # Use Perrin theory for anisotropic rotational diffusion.
        elif model == "aniso":
            if self.molecule is None:
                raise ValueError(
                    "Molecule must be assigned to the 'molecule' attribute "
                    "before calling auto_tau_c with the anisotropic model."
                )
            self.tau_c = rotational_correlation_times_Perrin(
                self.molecule.masses,
                self.molecule.xyz,
                T,
                eta,
                2,
            ) * scaling_factor

        # Reject unsupported rotational-diffusion models.
        else:
            raise ValueError("Model must be either 'iso' or 'aniso'.")

    @property
    def theory(
        self,
    ) -> Literal["redfield", "phenomenological"] | None:
        """
        Return the selected relaxation theory.

        Supported values are ``"redfield"`` and ``"phenomenological"``.
        The value is ``None`` until a theory is selected explicitly.
        """

        return self._theory

    @theory.setter
    def theory(
        self,
        theory: Literal["redfield", "phenomenological"],
    ) -> None:
        # Validate the requested relaxation-theory label.
        if theory not in ["redfield", "phenomenological"]:
            raise ValueError(
                "Relaxation theory must be 'redfield' or 'phenomenological'."
            )

        # Store the selected relaxation theory.
        self._theory = theory

        # Report the updated relaxation-theory selection.
        status(f"Relaxation theory set to: {self.theory}\n")

    @property
    def thermalization(
        self,
    ) -> bool:
        """
        Return whether Levitt-di Bari thermalization is applied.

        The default value is ``False``.
        """

        return self._thermalization

    @thermalization.setter
    def thermalization(
        self,
        thermalization: bool,
    ) -> None:
        # Store the thermalization setting.
        self._thermalization = thermalization

        # Report the updated thermalization setting.
        status(f"Thermalization set to: {self.thermalization}\n")

    @property
    def T1(
        self,
    ) -> np.ndarray | None:
        """
        Return the longitudinal relaxation time constants for each spin.

        These values are used to construct the phenomenological relaxation
        superoperator. Two input types are supported:

        - If `ArrayLike`: a one-dimensional array of size ``N`` containing
          ``T1`` values.
        - If `str`: path to the file containing the ``T1`` values.

        The input is converted and stored as a NumPy array. The value is
        ``None`` until it is assigned.

        Examples::

            # Using array input
            spin_system.relaxation.T1 = np.array([5.5, 6.0, 2.7])

            # Using string input
            spin_system.relaxation.T1 = "/path/to/the/file/T1.txt"
        """

        return self._T1

    @T1.setter
    def T1(
        self,
        T1: list | tuple | np.ndarray | str,
    ) -> None:
        # Convert the T1 input to a validated NumPy array.
        T1 = self._convert_relaxation_array(T1, "T1")
        self._validate_positive_spin_array(T1, "T1")

        # Store the longitudinal relaxation times.
        self._T1 = T1

        # Report the updated longitudinal relaxation times.
        status(f"T1 set to: {self.T1}\n")

    @property
    def T2(
        self,
    ) -> np.ndarray | None:
        """
        Return the transverse relaxation time constants for each spin.

        These values are used to construct the phenomenological relaxation
        superoperator. Two input types are supported:

        - If `ArrayLike`: a one-dimensional array of size ``N`` containing
          ``T2`` values.
        - If `str`: path to the file containing the ``T2`` values.

        The input is converted and stored as a NumPy array. The value is
        ``None`` until it is assigned.

        Examples::

            # Using array input
            spin_system.relaxation.T2 = np.array([5.5, 6.0, 2.7])

            # Using string input
            spin_system.relaxation.T2 = "/path/to/the/file/T2.txt"
        """

        return self._T2

    @T2.setter
    def T2(
        self,
        T2: list | tuple | np.ndarray | str,
    ) -> None:
        # Convert the T2 input to a validated NumPy array.
        T2 = self._convert_relaxation_array(T2, "T2")
        self._validate_positive_spin_array(T2, "T2")

        # Store the transverse relaxation times.
        self._T2 = T2

        # Report the updated transverse relaxation times.
        status(f"T2 set to: {self.T2}\n")

    @property
    def R1(
        self,
    ) -> np.ndarray:
        """
        Return the longitudinal relaxation rates for each spin.

        These values are used to construct the phenomenological relaxation
        superoperator. Two input types are supported:

        - If `ArrayLike`: a one-dimensional array of size ``N`` containing
          ``R1`` values.
        - If `str`: path to the file containing the ``R1`` values.

        The input is stored indirectly through the relation ``R1 = 1 / T1``.

        Examples::

            # Using array input
            spin_system.relaxation.R1 = np.array([5.5, 6.0, 2.7])

            # Using string input
            spin_system.relaxation.R1 = "/path/to/the/file/R1.txt"
        """

        return 1 / self.T1

    @R1.setter
    def R1(
        self,
        R1: list | tuple | np.ndarray | str,
    ) -> None:
        # Convert the R1 input to a validated NumPy array.
        R1 = self._convert_relaxation_array(R1, "R1")
        self._validate_positive_spin_array(R1, "R1")

        # Store the corresponding longitudinal relaxation times.
        self._T1 = 1 / R1

        # Report the updated longitudinal relaxation rates.
        status(f"R1 set to: {self.R1}\n")

    @property
    def R2(
        self,
    ) -> np.ndarray:
        """
        Return the transverse relaxation rates for each spin.

        These values are used to construct the phenomenological relaxation
        superoperator. Two input types are supported:

        - If `ArrayLike`: a one-dimensional array of size ``N`` containing
          ``R2`` values.
        - If `str`: path to the file containing the ``R2`` values.

        The input is stored indirectly through the relation ``R2 = 1 / T2``.

        Examples::

            # Using array input
            spin_system.relaxation.R2 = np.array([5.5, 6.0, 2.7])

            # Using string input
            spin_system.relaxation.R2 = "/path/to/the/file/R2.txt"
        """

        return 1 / self.T2

    @R2.setter
    def R2(
        self,
        R2: list | tuple | np.ndarray | str,
    ) -> None:
        # Convert the R2 input to a validated NumPy array.
        R2 = self._convert_relaxation_array(R2, "R2")
        self._validate_positive_spin_array(R2, "R2")

        # Store the corresponding transverse relaxation times.
        self._T2 = 1 / R2

        # Report the updated transverse relaxation rates.
        status(f"R2 set to: {self.R2}\n")

    @property
    def molecule(
        self,
    ) -> Molecule | None:
        """
        Return the molecule associated with the spin system.

        The molecule is used to define the rotational principal axes. The
        value is ``None`` until a molecule is assigned explicitly.
        """

        return self._molecule

    @molecule.setter
    def molecule(
        self,
        molecule: Molecule,
    ) -> None:
        # Validate the assigned molecule object.
        if not isinstance(molecule, Molecule):
            raise ValueError("Invalid input type for molecule.")

        # Store the molecular structure used in relaxation calculations.
        self._molecule = molecule

        # Report that a molecule has been assigned.
        status("Molecule has been assigned.\n")