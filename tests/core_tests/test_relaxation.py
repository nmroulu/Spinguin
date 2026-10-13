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

    def test_auto_tau_c_iso(self):
        """
        Test calculating the rotational correlation time using the isotropic
        model.
        """
        # Set the global simulation parameters.
        sg.parameters.default()
        sg.parameters.temperature = 293

        # Solvent viscosity
        eta = 0.59e-3
        
        # Create the spin system
        ss = sg.SpinSystem(test_data_path("isotopes_pyridine.txt"))

        # Error is raised if hydrodynamic radius or molecule is not specified
        with self.assertRaises(ValueError):
            ss.relaxation.auto_tau_c(eta, "iso")

        # Calculate correlation time using molecule structure
        ss.relaxation.molecule = sg.Molecule(
            isotopes = test_data_path("isotopes_pyridine.txt"),
            xyz = test_data_path("xyz_pyridine.txt")
        )
        ss.relaxation.auto_tau_c(eta, "iso")
        tau_c_1 = ss.relaxation.tau_c

        # Re-calculate correlation time using custom hydrodynamic radius
        r = 2.5
        ss.relaxation.auto_tau_c(eta, "iso", r)
        tau_c_2 = ss.relaxation.tau_c

        # Confirm that the values are not the same
        self.assertTrue(tau_c_1 != tau_c_2)

        # Confirm that the results are reasonable (< 30% off). The reference
        # uses the Bondi van der Waals volume of pyridine (r = 2.66 Å).
        tau_c_ref = 11.4e-12
        self.assertTrue(abs(tau_c_1 - tau_c_ref)/tau_c_ref < 0.3)
        self.assertTrue(abs(tau_c_2 - tau_c_ref)/tau_c_ref < 0.3)

    def test_auto_tau_c_aniso(self):
        """
        Test calculating the rotational correlation times using the
        anisotropic model.
        """
        # Set the global simulation parameters.
        sg.parameters.default()
        sg.parameters.temperature = 293

        # Solvent viscosity
        eta = 0.59e-3

        # Create a spin system whose molecule is a single carbon atom.
        ss = sg.SpinSystem(["13C"])
        ss.relaxation.molecule = sg.Molecule(["13C"], [[0, 0, 0]])

        # Calculate the correlation times of both models for the sphere.
        ss.relaxation.auto_tau_c(eta, "iso")
        tau_c_iso = ss.relaxation.tau_c
        ss.relaxation.auto_tau_c(eta, "aniso")
        tau_c_aniso = ss.relaxation.tau_c

        # Confirm that the Perrin model reduces to the isotropic model.
        self.assertTrue(np.allclose(tau_c_aniso, tau_c_iso, rtol=1e-6))

        # Confirm that the scaling factor multiplies all correlation times.
        ss.relaxation.auto_tau_c(eta, "aniso", scaling_factor=0.1)
        self.assertTrue(np.allclose(ss.relaxation.tau_c, 0.1*tau_c_aniso))

        # Create the pyridine spin system and its molecule.
        ss = sg.SpinSystem(test_data_path("isotopes_pyridine.txt"))
        ss.relaxation.molecule = sg.Molecule(
            isotopes = test_data_path("isotopes_pyridine.txt"),
            xyz = test_data_path("xyz_pyridine.txt")
        )

        # Calculate the anisotropic correlation times of pyridine.
        ss.relaxation.auto_tau_c(eta, "aniso")
        tau_c = ss.relaxation.tau_c

        # Confirm that rotation about the ring normal is the slowest.
        self.assertTrue(tau_c[0] < tau_c[1] < tau_c[2])

        # Confirm that the results are reasonable (< 30% off).
        tau_c_ref = np.array([12.1e-12, 13.0e-12, 15.6e-12])
        self.assertTrue(np.all(np.abs(tau_c - tau_c_ref)/tau_c_ref < 0.3))

    def test_relaxation_aniso_rotation_invariance(self):
        """
        Test that the anisotropic Redfield relaxation rate does not depend on
        the orientation of the molecule in the laboratory frame.
        """
        # Set the global simulation parameters.
        sg.parameters.default()
        sg.parameters.magnetic_field = 9.4
        sg.parameters.temperature = 293

        # Define an asymmetric three-spin molecule and its correlation times.
        isotopes = ["13C", "1H", "1H"]
        xyz = np.array([[0.0, 0.0, 0.0], [1.1, 0.0, 0.0], [-0.4, 1.0, 0.3]])
        tau_c = [10e-12, 20e-12, 40e-12]

        # Calculate the 13C z-magnetisation relaxation rate for random orientations.
        rng = np.random.default_rng(1)
        rates = []
        for _ in range(4):

            # Rotate the molecule by a random proper rotation.
            q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
            q *= np.sign(np.linalg.det(q))
            xyz_rotated = xyz @ q.T

            # Build the spin system with anisotropic rotational diffusion.
            ss = sg.SpinSystem(isotopes)
            ss.xyz = xyz_rotated
            ss.basis.build()
            ss.relaxation.theory = "redfield"
            ss.relaxation.tau_c = tau_c
            ss.relaxation.molecule = sg.Molecule(isotopes, xyz_rotated)

            # Project the relaxation superoperator onto the 13C magnetisation.
            R = sg.relaxation(ss).toarray()
            rho = np.ravel(sg.state(ss, "I(z, 0)"))
            rates.append(np.real(rho.conj() @ R @ rho / (rho.conj() @ rho)))

        # Confirm that all orientations give the same rate. The tolerance
        # allows for the grid sampling of the van der Waals solid.
        self.assertTrue(np.allclose(rates, rates[0], rtol=1e-3))
