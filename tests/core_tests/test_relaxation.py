"""
Tests for relaxation models and relaxation superoperators.
"""

import unittest

import numpy as np
import scipy.sparse as sp

import spinguin as sg
from ._helpers import build_spin_system, test_data_path

class TestRelaxation(unittest.TestCase):
    """
    Test relaxation superoperator construction.
    """

    def _redfield_spin_system(self) -> sg.SpinSystem:
        """
        Create the spin system used in Redfield relaxation tests.

        Returns
        -------
        SpinSystem
            Spin system with a built basis set and assigned interactions.
        """

        # Make the spin system
        ss = build_spin_system(["1H", "1H", "14N"], 3)

        # Define the chemical shifts (in ppm)
        ss.chemical_shifts = [5, 6, 7]

        # Define scalar couplings (in Hz)
        ss.J_couplings = [
            [0,  0,  0],
            [5,  0,  0],
            [2,  10, 0]
        ]

        # Define Cartesian coordinates of nuclei
        ss.xyz = [
            [2.0000000, 0.0000000, 0.0000000],
            [0.0000000, 2.0000000, 0.0000000],
            [0.0000000, 0.0000000, 2.0000000],
        ]

        # Define the shielding tensors.
        shielding = np.zeros((3, 3, 3))
        shielding[2] = np.array([
            [-130.0, -150.0, -70.00],
            [-120.0,  90.00,  230.0],
            [-60.00,  230.0, -30.00]
        ])
        ss.shielding = shielding

        # Define the electric-field-gradient tensors.
        efg = np.zeros((3, 3, 3))
        efg[2] = np.array([
            [0.3000, 0.0000,  0.0000],
            [0.0000, 0.8000,  0.0000],
            [0.0000, 0.0000, -1.1000]
        ])
        ss.efg = efg

        return ss
    
    def _phenomenological_spin_system(self) -> sg.SpinSystem:
        """
        Create the spin system used in testing the phenomenological relaxation
        superoperator.

        Returns
        -------
        SpinSystem
            Spin system with a built basis set and assigned interactions.
        """
        # Make the spin system
        ss = build_spin_system(["1H", "1H", "14N"], 3)

        # Define the chemical shifts (in ppm)
        ss.chemical_shifts = [5, 6, 7]

        # Define scalar couplings (in Hz)
        ss.J_couplings = [
            [0,  0,  0],
            [5,  0,  0],
            [2,  10, 0]
        ]
        
        # Define the relaxation theory
        ss.relaxation.theory = "phenomenological"
        ss.relaxation.T1 = np.array([5, 5, 0.001])
        ss.relaxation.T2 = np.array([5, 5, 0.001])

        return ss

    def test_ldb_thermalization(self):
        """
        Test that relaxation drives the system back to thermal equilibrium.
        """

        # Set the global simulation parameters.
        sg.parameters.default()
        sg.parameters.magnetic_field = 1
        sg.parameters.temperature = 273

        # Build the Redfield test system.
        ss = self._redfield_spin_system()

        # Define the relaxation theory
        ss.relaxation.theory = "redfield"
        ss.relaxation.tau_c = 50e-12
        ss.relaxation.thermalization = True
        ss.relaxation.sr2k = True

        # Define the propagation parameters.
        time_step = 2e-3
        nsteps = 40000

        # Build the Liouvillian.
        H = sg.hamiltonian(ss)
        R = sg.relaxation(ss)
        L = sg.liouvillian(H, R)

        # Create the thermal equilibrium state.
        rho = sg.equilibrium_state(ss)

        # Invert the proton spins.
        op_string = "I(x,0) + I(x,1)"
        pul_180 = sg.pulse(ss, op_string, 180)
        rho = pul_180 @ rho

        # Restrict the evolution to the zero-quantum subspace.
        L, rho = ss.basis.truncate_by_coherence([0], L, rho)

        # Build the propagator and evolve the system.
        P = sg.propagator(L, time_step)
        for _ in range(nsteps):
            rho = P @ rho

        # Construct the equilibrium reference in the same basis.
        rho_ref = sg.equilibrium_state(ss)

        # Verify that the final state matches thermal equilibrium.
        self.assertTrue(np.allclose(rho, rho_ref))

    def test_relaxation_redfield_1(self):
        """
        Test the Redfield relaxation superoperator against reference data.
        """

        # Set the global simulation parameters.
        sg.parameters.default()
        sg.parameters.magnetic_field = 1
        sg.parameters.temperature = 293

        # Load the previously calculated Redfield superoperator.
        R_ref = sp.csc_array(sp.load_npz(test_data_path("relaxation.npz")))

        # Build the Redfield test system.
        ss = self._redfield_spin_system()

        # Define the relaxation model.
        ss.relaxation.theory = "redfield"
        ss.relaxation.antisymmetric = True
        ss.relaxation.dynamic_frequency_shift = True
        ss.relaxation.thermalization = True
        ss.relaxation.tau_c = 50e-12
        
        # Build the Redfield relaxation superoperator.
        R = sg.relaxation(ss)

        # Compare the calculated superoperator with the reference.
        self.assertTrue(np.allclose(R.toarray(), R_ref.toarray()))

        # Recalculate the same superoperator to exercise caching paths.
        R = sg.relaxation(ss)

        # Compare the repeated calculation with the reference.
        self.assertTrue(np.allclose(R.toarray(), R_ref.toarray()))

    def test_relaxation_redfield_2(self):
        """
        Test Redfield relaxation after zero-quantum basis truncation.
        """
        # Set the global parameters
        sg.parameters.default()
        sg.parameters.magnetic_field = 1
        sg.parameters.temperature = 293

        # Build the Redfield test system.
        ss = self._redfield_spin_system()

        # Define the relaxation model.
        ss.relaxation.theory = "redfield"
        ss.relaxation.tau_c = 50e-12
        ss.relaxation.thermalization = True
        ss.relaxation.antisymmetric = True
        ss.relaxation.dynamic_frequency_shift = True

        # Truncate the basis set to zero-quantum terms only.
        ss.basis.truncate_by_coherence([0])

        # Build the relaxation superoperator and verify that no error is raised.
        sg.relaxation(ss)

    def test_relaxation_redfield_3(self):
        """
        Test isotropic and equivalent anisotropic Redfield diffusion models.
        """
        # Set the global parameters
        sg.parameters.default()
        sg.parameters.magnetic_field = 1
        sg.parameters.temperature = 293

        # Build the Redfield test system.
        ss = self._redfield_spin_system()

        # Define the relaxation theory
        ss.relaxation.theory = "redfield"
        ss.relaxation.antisymmetric = True
        ss.relaxation.dynamic_frequency_shift = True
        ss.relaxation.thermalization = True
        
        # Build the superoperator using isotropic rotational diffusion.
        ss.relaxation.tau_c = 50e-12
        R_iso = sg.relaxation(ss)

        # Build the superoperator using equivalent anisotropic diffusion.
        ss.relaxation.tau_c = [50e-12, 50e-12, 50e-12]
        ss.relaxation.molecule = sg.Molecule(ss.isotopes, ss.xyz)
        R_aniso = sg.relaxation(ss)

        # Verify that both diffusion models give the same result.
        # NOTE: Absolute tolerance is increased slightly
        self.assertTrue(
            np.allclose(R_iso.toarray(), R_aniso.toarray(), atol=1e-7)
        )

    def test_relaxation_phenomenological(self):
        """
        Test the phenomenological relaxation superoperator against reference
        data.
        """

        # Set the global parameters
        sg.parameters.default()
        sg.parameters.magnetic_field = 9.4
        sg.parameters.temperature = 293

        # Load the previously calculated R for comparison
        R_ref = sp.csc_array(
            sp.load_npz(test_data_path('relaxation_phenomenological.npz'))
        )

        # Build the spin system for the phenomenological relaxation test.
        ss = self._phenomenological_spin_system()

        # Build the relaxation superoperator.
        R = sg.relaxation(ss)

        # Compare the calculated superoperator with the reference.
        self.assertTrue(np.allclose(R.toarray(), R_ref.toarray()))

        # Recalculate the same superoperator to exercise caching paths.
        R = sg.relaxation(ss)

        # Compare the repeated calculation with the reference.
        self.assertTrue(np.allclose(R.toarray(), R_ref.toarray()))

    def test_relaxation_sr2k(self):
        """
        Test the SR2K contribution against pre-computed reference data.
        """
        # Set the global simulation parameters.
        sg.parameters.default()
        sg.parameters.magnetic_field = 3e-6
        sg.parameters.temperature = 293

        # Load the previously calculated SR2K superoperator.
        R_ref = sp.csc_array(sp.load_npz(test_data_path('relaxation_sr2k.npz')))

        # Build the spin system for the SR2K test.
        ss = self._phenomenological_spin_system()

        # Enable SR2K
        ss.relaxation.sr2k = "true"
        
        # Build the relaxation superoperator.
        R = sg.relaxation(ss)

        # Compare the calculated superoperator with the reference.
        self.assertTrue(np.allclose(R.toarray(), R_ref.toarray()))
        
        # Recalculate the same superoperator to exercise caching paths.
        R = sg.relaxation(ss)
        
        # Compare the repeated calculation with the reference.
        self.assertTrue(np.allclose(R.toarray(), R_ref.toarray()))