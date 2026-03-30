"""
Tests for relaxation models and relaxation superoperators.
"""

import os
import unittest

import numpy as np
import scipy.sparse as sp

import spinguin as sg
from spinguin._core._nmr_isotopes import ISOTOPES
from spinguin._core._relaxation import dd_constant


class TestRelaxation(unittest.TestCase):
    """
    Test dipolar constants and relaxation superoperator construction.
    """

    def _get_test_data_path(
        self,
        filename,
    ):
        """
        Return the absolute path to a shared test-data file.

        Parameters
        ----------
        filename : str
            Name of the requested test-data file.

        Returns
        -------
        str
            Absolute path to the requested file.
        """

        # Locate the shared test-data directory.
        test_data_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "test_data",
        )

        return os.path.join(test_data_dir, filename)

    def _assert_allclose(
        self,
        value,
        reference,
        rtol=1e-05,
        atol=1e-08,
    ):
        """
        Check that two arrays or sparse matrices agree numerically.

        Parameters
        ----------
        value : array-like or sparse matrix
            Tested value.
        reference : array-like or sparse matrix
            Reference value.
        rtol : float, optional
            Relative tolerance passed to `numpy.allclose`.
        atol : float, optional
            Absolute tolerance passed to `numpy.allclose`.

        Returns
        -------
        None
            The assertion is evaluated in place.
        """

        # Convert sparse matrices to dense arrays when necessary.
        if hasattr(value, "toarray"):
            value = value.toarray()
        if hasattr(reference, "toarray"):
            reference = reference.toarray()

        # Compare the tested and reference values.
        self.assertTrue(np.allclose(value, reference, rtol=rtol, atol=atol))

    def _build_redfield_spin_system(
        self,
    ):
        """
        Create the shared six-spin system used in Redfield tests.

        Returns
        -------
        SpinSystem
            Spin system with a built basis set and assigned interactions.
        """

        # Create the shared proton-nitrogen spin system.
        spin_system = sg.SpinSystem(["1H", "1H", "1H", "1H", "1H", "14N"])

        # Build the basis set used in the regression tests.
        spin_system.basis.max_spin_order = 3
        spin_system.basis.build()

        # Define the chemical shifts in ppm.
        spin_system.chemical_shifts = [8.56, 8.56, 7.47, 7.47, 7.88, 95.94]

        # Define the scalar couplings in Hz.
        spin_system.J_couplings = [
            [0, 0, 0, 0, 0, 0],
            [-1.04, 0, 0, 0, 0, 0],
            [4.85, 1.05, 0, 0, 0, 0],
            [1.05, 4.85, 0.71, 0, 0, 0],
            [1.24, 1.24, 7.55, 7.55, 0, 0],
            [8.16, 8.16, 0.87, 0.87, -0.19, 0],
        ]

        # Define the Cartesian coordinates of the nuclei.
        spin_system.xyz = [
            [2.0495335, 0.0000000, -1.4916842],
            [-2.0495335, 0.0000000, -1.4916842],
            [2.1458878, 0.0000000, 0.9846086],
            [-2.1458878, 0.0000000, 0.9846086],
            [0.0000000, 0.0000000, 2.2681296],
            [0.0000000, 0.0000000, -1.5987077],
        ]

        # Define the shielding tensors.
        shielding = np.zeros((6, 3, 3))
        shielding[5] = np.array(
            [
                [-406.20, 0.00, 0.00],
                [0.00, 299.44, 0.00],
                [0.00, 0.00, -181.07],
            ]
        )
        spin_system.shielding = shielding

        # Define the electric-field-gradient tensors.
        efg = np.zeros((6, 3, 3))
        efg[5] = np.array(
            [
                [0.3069, 0.0000, 0.0000],
                [0.0000, 0.7969, 0.0000],
                [0.0000, 0.0000, -1.1037],
            ]
        )
        spin_system.efg = efg

        return spin_system

    def test_dd_constant(self):
        """
        Test the dipole-dipole relaxation constant against tabulated values.
        """

        # Get gyromagnetic ratios in rad s^-1 T^-1.
        gamma_13c = 2 * np.pi * ISOTOPES["13C"][1] * 1e6
        gamma_1h = 2 * np.pi * ISOTOPES["1H"][1] * 1e6
        gamma_15n = 2 * np.pi * ISOTOPES["15N"][1] * 1e6

        # Define interatomic distances in metres.
        r_13c_13c = 0.153e-9
        r_13c_1h = 0.106e-9
        r_13c_15n = 0.147e-9

        # Calculate the DD constants and convert them to hertz.
        dd_13c_13c = (
            -dd_constant(gamma_13c, gamma_13c) / r_13c_13c**3 /
            (2 * np.pi)
        )
        dd_13c_1h = (
            -dd_constant(gamma_13c, gamma_1h) / r_13c_1h**3 /
            (2 * np.pi)
        )
        dd_13c_15n = (
            dd_constant(gamma_13c, gamma_15n) / r_13c_15n**3 /
            (2 * np.pi)
        )

        # Compare with the tabulated reference values in hertz.
        self._assert_allclose(2.12e3, dd_13c_13c, rtol=0.01)
        self._assert_allclose(25.44e3, dd_13c_1h, rtol=0.01)
        self._assert_allclose(0.97e3, dd_13c_15n, rtol=0.01)

    def test_ldb_thermalization(self):
        """
        Test that relaxation drives the system back to thermal equilibrium.
        """

        # Set the global simulation parameters.
        sg.parameters.default()
        sg.parameters.magnetic_field = 1
        sg.parameters.temperature = 273

        # Build the shared Redfield test system.
        spin_system = self._build_redfield_spin_system()

        # Define the relaxation model.
        spin_system.relaxation.theory = "redfield"
        spin_system.relaxation.tau_c = 50e-12
        spin_system.relaxation.thermalization = True
        spin_system.relaxation.sr2k = True

        # Define the propagation parameters.
        time_step = 2e-3
        nsteps = 50000

        # Build the Hamiltonian, relaxation superoperator, and Liouvillian.
        hamiltonian = sg.hamiltonian(spin_system)
        relaxation_superoperator = sg.relaxation(spin_system)
        liouvillian = sg.liouvillian(hamiltonian, relaxation_superoperator)

        # Create the thermal equilibrium state and invert the proton spins.
        state = sg.equilibrium_state(spin_system)
        pulse_operator = (
            "I(x,0) + I(x,1) + I(x,2) + I(x,3) + I(x,4) + I(x,5)"
        )
        pulse_180 = sg.pulse(spin_system, pulse_operator, 180)
        state = pulse_180 @ state

        # Restrict the evolution to the zero-quantum subspace.
        liouvillian, state = spin_system.basis.truncate_by_coherence(
            [0],
            liouvillian,
            state,
        )

        # Build the propagator and evolve the system.
        propagator = sg.propagator(liouvillian, time_step)
        for _ in range(nsteps):
            state = propagator @ state

        # Construct the equilibrium reference in the same basis.
        state_reference = sg.equilibrium_state(spin_system)

        # Verify that the final state matches thermal equilibrium.
        self._assert_allclose(state, state_reference)

    def test_relaxation_redfield_1(self):
        """
        Test the Redfield relaxation superoperator against reference data.
        """

        # Set the global simulation parameters.
        sg.parameters.default()
        sg.parameters.magnetic_field = 1

        # Load the previously calculated Redfield superoperator.
        relaxation_reference = sp.csc_array(
            sp.load_npz(self._get_test_data_path("relaxation.npz"))
        )

        # Build the shared Redfield test system.
        spin_system = self._build_redfield_spin_system()

        # Define the relaxation model.
        spin_system.relaxation.theory = "redfield"
        spin_system.relaxation.tau_c = 50e-12

        # Build the Redfield relaxation superoperator.
        relaxation_superoperator = sg.relaxation(spin_system)

        # Compare the calculated superoperator with the reference.
        self._assert_allclose(relaxation_superoperator, relaxation_reference)

        # Recalculate the same superoperator to exercise caching paths.
        relaxation_superoperator = sg.relaxation(spin_system)

        # Compare the repeated calculation with the reference.
        self._assert_allclose(relaxation_superoperator, relaxation_reference)

    def test_relaxation_redfield_2(self):
        """
        Test Redfield relaxation after zero-quantum basis truncation.
        """

        # Set the global simulation parameters.
        sg.parameters.default()
        sg.parameters.magnetic_field = 1

        # Build the shared Redfield test system.
        spin_system = self._build_redfield_spin_system()

        # Define the relaxation model.
        spin_system.relaxation.theory = "redfield"
        spin_system.relaxation.tau_c = 50e-12

        # Truncate the basis set to zero-quantum terms only.
        spin_system.basis.truncate_by_coherence([0])

        # Build the relaxation superoperator and verify that no error is raised.
        sg.relaxation(spin_system)

    def test_relaxation_redfield_3(self):
        """
        Test isotropic and equivalent anisotropic Redfield diffusion models.
        """

        # Set the global simulation parameters.
        sg.parameters.default()
        sg.parameters.magnetic_field = 1

        # Build the shared Redfield test system.
        spin_system = self._build_redfield_spin_system()

        # Define the Redfield relaxation model.
        spin_system.relaxation.theory = "redfield"

        # Build the superoperator using isotropic rotational diffusion.
        spin_system.relaxation.tau_c = 50e-12
        relaxation_isotropic = sg.relaxation(spin_system)

        # Build the superoperator using equivalent anisotropic diffusion.
        spin_system.relaxation.tau_c = [50e-12, 50e-12, 50e-12]
        spin_system.relaxation.molecule = sg.Molecule(
            spin_system.isotopes,
            spin_system.xyz,
        )
        relaxation_anisotropic = sg.relaxation(spin_system)

        # Verify that both diffusion models give the same result.
        self._assert_allclose(relaxation_isotropic, relaxation_anisotropic)

    def test_relaxation_phenomenological(self):
        """
        Test the phenomenological relaxation superoperator against reference data.
        """

        # Set the global simulation parameters.
        sg.parameters.default()

        # Load the previously calculated phenomenological superoperator.
        relaxation_reference = sp.csc_array(
            sp.load_npz(
                self._get_test_data_path("relaxation_phenomenological.npz")
            )
        )

        # Build the spin system for the phenomenological relaxation test.
        spin_system = sg.SpinSystem(["1H", "1H", "1H", "1H", "1H", "14N"])
        spin_system.basis.max_spin_order = 3
        spin_system.basis.build()

        # Define the phenomenological relaxation model.
        spin_system.relaxation.theory = "phenomenological"
        spin_system.relaxation.T1 = np.array([5, 5, 5, 5, 5, 0.001])
        spin_system.relaxation.T2 = np.array([5, 5, 5, 5, 5, 0.001])

        # Build the relaxation superoperator.
        relaxation_superoperator = sg.relaxation(spin_system)

        # Compare the calculated superoperator with the reference.
        self._assert_allclose(relaxation_superoperator, relaxation_reference)

        # Recalculate the same superoperator to exercise caching paths.
        relaxation_superoperator = sg.relaxation(spin_system)

        # Compare the repeated calculation with the reference.
        self._assert_allclose(relaxation_superoperator, relaxation_reference)

    def test_relaxation_sr2k(self):
        """
        Test the SR2K contribution against pre-computed reference data.
        """

        # Set the global simulation parameters.
        sg.parameters.default()
        sg.parameters.magnetic_field = 3e-6

        # Load the previously calculated SR2K superoperator.
        relaxation_reference = sp.csc_array(
            sp.load_npz(self._get_test_data_path("relaxation_sr2k.npz"))
        )

        # Build the spin system for the SR2K test.
        spin_system = sg.SpinSystem(["1H", "1H", "1H", "1H", "1H", "14N"])
        spin_system.basis.max_spin_order = 3
        spin_system.basis.build()
        spin_system.chemical_shifts = [8.56, 8.56, 7.47, 7.47, 7.88, 95.94]
        spin_system.J_couplings = [
            [0, 0, 0, 0, 0, 0],
            [-1.04, 0, 0, 0, 0, 0],
            [4.85, 1.05, 0, 0, 0, 0],
            [1.05, 4.85, 0.71, 0, 0, 0],
            [1.24, 1.24, 7.55, 7.55, 0, 0],
            [8.16, 8.16, 0.87, 0.87, -0.19, 0],
        ]

        # Define the phenomenological relaxation model with SR2K enabled.
        spin_system.relaxation.theory = "phenomenological"
        spin_system.relaxation.sr2k = "true"
        spin_system.relaxation.T1 = np.array([5, 5, 5, 5, 5, 0.001])
        spin_system.relaxation.T2 = np.array([5, 5, 5, 5, 5, 0.001])

        # Build the relaxation superoperator.
        relaxation_superoperator = sg.relaxation(spin_system)

        # Compare the calculated superoperator with the reference.
        self._assert_allclose(relaxation_superoperator, relaxation_reference)

        # Recalculate the same superoperator to exercise caching paths.
        relaxation_superoperator = sg.relaxation(spin_system)

        # Compare the repeated calculation with the reference.
        self._assert_allclose(relaxation_superoperator, relaxation_reference)