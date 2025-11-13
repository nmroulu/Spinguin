import unittest
import numpy as np
import scipy.sparse as sp
import os
import spinguin as sg
from spinguin._core._relaxation import dd_constant
from spinguin._core._nmr_isotopes import ISOTOPES

class TestRelaxation(unittest.TestCase):

    def test_dd_constant(self):
        """
        Test the dipole-dipole (DD) relaxation constant calculation.
        Compares calculated values against tabulated values from the reference:
        Apperley, Harris & Hodgkinson: Solid-state NMR: Basic principles and
        practice.
        """

        # Get gyromagnetic ratios (gamma) in Hz/T
        y_13C = 2 * np.pi * ISOTOPES['13C'][1] * 1e6
        y_1H = 2 * np.pi * ISOTOPES['1H'][1] * 1e6
        y_15N = 2 * np.pi * ISOTOPES['15N'][1] * 1e6

        # Interatomic distances in meters
        r_13C_13C = 0.153e-9
        r_13C_1H = 0.106e-9
        r_13C_15N = 0.147e-9

        # Calculate DD constants and convert to Hz
        dd_13C_13C = -dd_constant(y_13C, y_13C) / r_13C_13C**3 / (2 * np.pi)
        dd_13C_1H = -dd_constant(y_13C, y_1H) / r_13C_1H**3 / (2 * np.pi)
        dd_13C_15N = dd_constant(y_13C, y_15N) / r_13C_15N**3 / (2 * np.pi)

        # Compare with tabulated values (in Hz)
        self.assertTrue(np.allclose(2.12e3, dd_13C_13C, rtol=0.01))
        self.assertTrue(np.allclose(25.44e3, dd_13C_1H, rtol=0.01))
        self.assertTrue(np.allclose(0.97e3, dd_13C_15N, rtol=0.01))

    def test_ldb_thermalization(self):
        """
        Test the thermalization of the relaxation superoperator. Simulates the
        evolution of a spin system under relaxation and compares the final state
        with the thermal equilibrium state.
        """
        # Set the global parameters
        sg.parameters.default()
        sg.parameters.magnetic_field = 1
        sg.parameters.temperature = 273

        # Make the spin system
        ss = sg.SpinSystem(["1H", "1H", "1H", "1H", "1H", "14N"])
        ss.basis.max_spin_order = 3
        ss.basis.build()

        # Define the chemical shifts (in ppm)
        ss.chemical_shifts = [8.56, 8.56, 7.47, 7.47, 7.88, 95.94]

        # Define scalar couplings (in Hz)
        ss.J_couplings = [
            [ 0,     0,      0,      0,      0,      0],
            [-1.04,  0,      0,      0,      0,      0],
            [ 4.85,  1.05,   0,      0,      0,      0],
            [ 1.05,  4.85,   0.71,   0,      0,      0],
            [ 1.24,  1.24,   7.55,   7.55,   0,      0],
            [ 8.16,  8.16,   0.87,   0.87,  -0.19,   0]
        ]

        # Define Cartesian coordinates of nuclei
        ss.xyz = [
            [ 2.0495335, 0.0000000, -1.4916842],
            [-2.0495335, 0.0000000, -1.4916842],
            [ 2.1458878, 0.0000000,  0.9846086],
            [-2.1458878, 0.0000000,  0.9846086],
            [ 0.0000000, 0.0000000,  2.2681296],
            [ 0.0000000, 0.0000000, -1.5987077]
        ]

        # Define shielding tensors
        shielding = np.zeros((6, 3, 3))
        shielding[5] = np.array([
            [-406.20, 0.00,   0.00],
            [ 0.00,   299.44, 0.00],
            [ 0.00,   0.00,  -181.07]
        ])
        ss.shielding = shielding

        # Define electric field gradient tensors
        efg = np.zeros((6, 3, 3))
        efg[5] = np.array([
            [0.3069, 0.0000,  0.0000],
            [0.0000, 0.7969,  0.0000],
            [0.0000, 0.0000, -1.1037]
        ])
        ss.efg = efg

        # Define the relaxation theory
        ss.relaxation.theory = "redfield"
        ss.relaxation.tau_c = 50e-12
        ss.relaxation.thermalization = True
        ss.relaxation.sr2k = True

        # Propagation parameters
        time_step = 2e-3
        nsteps = 50000

        # Get the Hamiltonian
        H = sg.hamiltonian(ss)

        # Get the relaxation superoperator
        R = sg.relaxation(ss)

        # Construct the Liouvillian
        L = sg.liouvillian(H, R)

        # Create the thermal equilibrium state
        rho = sg.equilibrium_state(ss)

        # Apply a 180-degree pulse (to all 1H)
        op_string = "I(x,0) + I(x,1) + I(x,2) + I(x,3) + I(x,4) + I(x,5)"
        pul_180 = sg.pulse(ss, op_string, 180)
        rho = pul_180 @ rho

        # Switch to the zero-quantum (ZQ) subspace
        L = ss.basis.truncate_by_coherence([0], L)

        # Get the propagator
        P = sg.propagator(L, time_step)
        
        # Simulate the evolution of the spin system
        for _ in range(nsteps):
            rho = P @ rho

        # Create an equilibrium state in the ZQ basis for reference
        rho_ref = sg.equilibrium_state(ss)

        # Verify that the final state matches the thermal equilibrium state
        self.assertTrue(np.allclose(rho, rho_ref))

    def test_relaxation_redfield_1(self):
        """
        Test that creates a relaxation superoperator using Redfield theory and
        compares that to a previously calculated value.
        """
        # Set the global parameters
        sg.parameters.default()
        sg.parameters.magnetic_field = 1

        # Load the previously calculated R for comparison
        test_dir = os.path.dirname(os.path.dirname(__file__))
        R_previous = sp.csc_array(sp.load_npz(os.path.join(
            test_dir, 'test_data', 'relaxation.npz')
        ))

        # Make the spin system
        ss = sg.SpinSystem(["1H", "1H", "1H", "1H", "1H", "14N"])

        # Create the basis set
        ss.basis.max_spin_order = 3
        ss.basis.build()

        # Define the chemical shifts (in ppm)
        ss.chemical_shifts = [8.56, 8.56, 7.47, 7.47, 7.88, 95.94]

        # Define scalar couplings (in Hz)
        ss.J_couplings = [
            [ 0,     0,      0,      0,      0,      0],
            [-1.04,  0,      0,      0,      0,      0],
            [ 4.85,  1.05,   0,      0,      0,      0],
            [ 1.05,  4.85,   0.71,   0,      0,      0],
            [ 1.24,  1.24,   7.55,   7.55,   0,      0],
            [ 8.16,  8.16,   0.87,   0.87,  -0.19,   0]
        ]

        # Define Cartesian coordinates of nuclei
        ss.xyz = [
            [ 2.0495335, 0.0000000, -1.4916842],
            [-2.0495335, 0.0000000, -1.4916842],
            [ 2.1458878, 0.0000000,  0.9846086],
            [-2.1458878, 0.0000000,  0.9846086],
            [ 0.0000000, 0.0000000,  2.2681296],
            [ 0.0000000, 0.0000000, -1.5987077]
        ]

        # Define shielding tensors
        shielding = np.zeros((6, 3, 3))
        shielding[5] = np.array([
            [-406.20, 0.00,   0.00],
            [ 0.00,   299.44, 0.00],
            [ 0.00,   0.00,  -181.07]
        ])
        ss.shielding = shielding

        # Define electric field gradient tensors
        efg = np.zeros((6, 3, 3))
        efg[5] = np.array([
            [0.3069, 0.0000,  0.0000],
            [0.0000, 0.7969,  0.0000],
            [0.0000, 0.0000, -1.1037]
        ])
        ss.efg = efg

        # Define the relaxation theory
        ss.relaxation.theory = "redfield"
        ss.relaxation.tau_c = 50e-12
        
        # Get the Redfield relaxation superoperator
        R = sg.relaxation(ss)

        # Compare with the reference
        self.assertTrue(np.allclose(R.toarray(), R_previous.toarray()))

        # Obtain R again (to check possible errors from caches etc.)
        R = sg.relaxation(ss)

        # Compare with the reference
        self.assertTrue(np.allclose(R.toarray(), R_previous.toarray()))

    def test_relaxation_redfield_2(self):
        """
        Test that creates a relaxation superoperator using Redfield theory when
        the basis set has been truncated such that some spin operators relevant
        for relaxation no longer exist in the basis set.
        """
        # Set the global parameters
        sg.parameters.default()
        sg.parameters.magnetic_field = 1

        # Make the spin system
        ss = sg.SpinSystem(["1H", "1H", "1H", "1H", "1H", "14N"])

        # Create the basis set
        ss.basis.max_spin_order = 3
        ss.basis.build()

        # Define the chemical shifts (in ppm)
        ss.chemical_shifts = [8.56, 8.56, 7.47, 7.47, 7.88, 95.94]

        # Define scalar couplings (in Hz)
        ss.J_couplings = [
            [ 0,     0,      0,      0,      0,      0],
            [-1.04,  0,      0,      0,      0,      0],
            [ 4.85,  1.05,   0,      0,      0,      0],
            [ 1.05,  4.85,   0.71,   0,      0,      0],
            [ 1.24,  1.24,   7.55,   7.55,   0,      0],
            [ 8.16,  8.16,   0.87,   0.87,  -0.19,   0]
        ]

        # Define Cartesian coordinates of nuclei
        ss.xyz = [
            [ 2.0495335, 0.0000000, -1.4916842],
            [-2.0495335, 0.0000000, -1.4916842],
            [ 2.1458878, 0.0000000,  0.9846086],
            [-2.1458878, 0.0000000,  0.9846086],
            [ 0.0000000, 0.0000000,  2.2681296],
            [ 0.0000000, 0.0000000, -1.5987077]
        ]

        # Define shielding tensors
        shielding = np.zeros((6, 3, 3))
        shielding[5] = np.array([
            [-406.20, 0.00,   0.00],
            [ 0.00,   299.44, 0.00],
            [ 0.00,   0.00,  -181.07]
        ])
        ss.shielding = shielding

        # Define electric field gradient tensors
        efg = np.zeros((6, 3, 3))
        efg[5] = np.array([
            [0.3069, 0.0000,  0.0000],
            [0.0000, 0.7969,  0.0000],
            [0.0000, 0.0000, -1.1037]
        ])
        ss.efg = efg

        # Define the relaxation theory
        ss.relaxation.theory = "redfield"
        ss.relaxation.tau_c = 50e-12

        # Truncate the basis set to include only zero-quantum terms
        ss.basis.truncate_by_coherence([0])

        # Build the relaxation superoperator (and see that no errors arise)
        sg.relaxation(ss)

    def test_relaxation_phenomenological(self):
        """
        Test that creates the phenomenological relaxation superoperator and
        makes a comparison to a previously computed result.
        """
        # Set the global parameters
        sg.parameters.default()

        # Load the previously calculated R for comparison
        test_dir = os.path.dirname(os.path.dirname(__file__))
        R_previous = sp.csc_array(sp.load_npz(os.path.join(
            test_dir, 'test_data', 'relaxation_phenomenological.npz')
        ))

        # Make the spin system
        ss = sg.SpinSystem(["1H", "1H", "1H", "1H", "1H", "14N"])

        # Create the basis set
        ss.basis.max_spin_order = 3
        ss.basis.build()
        
        # Define the relaxation theory
        ss.relaxation.theory = "phenomenological"
        ss.relaxation.T1 = np.array([5, 5, 5, 5, 5, 0.001])
        ss.relaxation.T2 = np.array([5, 5, 5, 5, 5, 0.001])

        # Obtain the relaxation superoperator
        R = sg.relaxation(ss)

        # Compare to the previous result
        self.assertTrue(np.allclose(R.toarray(), R_previous.toarray()))

        # Obtain R again (check for cache errors etc.)
        R = sg.relaxation(ss)

        # Compare to the previous result
        self.assertTrue(np.allclose(R.toarray(), R_previous.toarray()))

    def test_relaxation_sr2k(self):
        """
        Test that adds the contribution from SR2K to a phenomenological
        relaxation superoperator and compares the result to a pre-computed
        value.
        """
        # Set the global parameters
        sg.parameters.default()
        sg.parameters.magnetic_field = 3e-6

        # Load the previously calculated R for comparison
        test_dir = os.path.dirname(os.path.dirname(__file__))
        R_previous = sp.csc_array(sp.load_npz(
            os.path.join(test_dir, 'test_data', 'relaxation_sr2k.npz')
        ))

        # Make the spin system
        ss = sg.SpinSystem(["1H", "1H", "1H", "1H", "1H", "14N"])

        # Create the basis set
        ss.basis.max_spin_order = 3
        ss.basis.build()

        # Define the chemical shifts (in ppm)
        ss.chemical_shifts = [8.56, 8.56, 7.47, 7.47, 7.88, 95.94]

        # Define scalar couplings (in Hz)
        ss.J_couplings = [
            [ 0,     0,      0,      0,      0,      0],
            [-1.04,  0,      0,      0,      0,      0],
            [ 4.85,  1.05,   0,      0,      0,      0],
            [ 1.05,  4.85,   0.71,   0,      0,      0],
            [ 1.24,  1.24,   7.55,   7.55,   0,      0],
            [ 8.16,  8.16,   0.87,   0.87,  -0.19,   0]
        ]

        # Define the relaxation theory
        ss.relaxation.theory = "phenomenological"
        ss.relaxation.sr2k = "true"
        ss.relaxation.T1 = np.array([5, 5, 5, 5, 5, 0.001])
        ss.relaxation.T2 = np.array([5, 5, 5, 5, 5, 0.001])
        
        # Get the relaxation superoperator
        R = sg.relaxation(ss)

        # Compare
        self.assertTrue(np.allclose(R.toarray(), R_previous.toarray()))
        
        # Get the relaxation superoperator again (check for cache errors etc.)
        R = sg.relaxation(ss)
        
        # Compare
        self.assertTrue(np.allclose(R.toarray(), R_previous.toarray()))