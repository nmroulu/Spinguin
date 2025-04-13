import unittest
import numpy as np
from spinguin._spin_system import SpinSystem
from spinguin import _hamiltonian, _propagation, _relaxation, _states
from spinguin._nmr_isotopes import ISOTOPES
from spinguin._basis import truncate_basis_by_coherence, transform_to_truncated_basis

class TestRelaxation(unittest.TestCase):

    def test_dd_constant(self):
        """
        Test the dipole-dipole (DD) relaxation constant calculation.
        Compares calculated values against tabulated values from the reference:
        Apperley, Harris & Hodgkinson: Solid-state NMR: Basic principles and practice.
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
        dd_13C_13C = -_relaxation.dd_constant(y_13C, y_13C) / r_13C_13C**3 / (2 * np.pi)
        dd_13C_1H = -_relaxation.dd_constant(y_13C, y_1H) / r_13C_1H**3 / (2 * np.pi)
        dd_13C_15N = _relaxation.dd_constant(y_13C, y_15N) / r_13C_15N**3 / (2 * np.pi)

        # Compare with tabulated values (in Hz)
        self.assertTrue(np.allclose(2.12e3, dd_13C_13C, rtol=0.01))
        self.assertTrue(np.allclose(25.44e3, dd_13C_1H, rtol=0.01))
        self.assertTrue(np.allclose(0.97e3, dd_13C_15N, rtol=0.01))

    def test_thermalization(self):
        """
        Test the thermalization process of a spin system.
        Simulates the evolution of a spin system under relaxation and compares
        the final state with the thermal equilibrium state.
        """

        # Define the spin system
        isotopes = np.array(['1H', '1H', '1H', '1H', '1H', '14N'])
        chemical_shifts = np.array([8.56, 8.56, 7.47, 7.47, 7.88, 95.94])
        J_couplings = np.array([  # Scalar coupling matrix
            [ 0,     0,      0,      0,      0,      0],
            [-1.04,  0,      0,      0,      0,      0],
            [ 4.85,  1.05,   0,      0,      0,      0],
            [ 1.05,  4.85,   0.71,   0,      0,      0],
            [ 1.24,  1.24,   7.55,   7.55,   0,      0],
            [ 8.16,  8.16,   0.87,   0.87,  -0.19,   0]
        ])
        xyz = np.array([  # Cartesian coordinates of nuclei
            [ 2.0495335, 0.0000000, -1.4916842],
            [-2.0495335, 0.0000000, -1.4916842],
            [ 2.1458878, 0.0000000,  0.9846086],
            [-2.1458878, 0.0000000,  0.9846086],
            [ 0.0000000, 0.0000000,  2.2681296],
            [ 0.0000000, 0.0000000, -1.5987077]
        ])
        shielding = np.zeros((6, 3, 3))  # Shielding tensors
        shielding[5] = np.array([
            [-406.20, 0.00,   0.00],
            [ 0.00,   299.44, 0.00],
            [ 0.00,   0.00,  -181.07]
        ])
        efg = np.zeros((6, 3, 3))  # Electric field gradient tensors
        efg[5] = np.array([
            [0.3069, 0.0000,  0.0000],
            [0.0000, 0.7969,  0.0000],
            [0.0000, 0.0000, -1.1037]
        ])
        spin_system = SpinSystem(isotopes, chemical_shifts, J_couplings, xyz, shielding, efg, max_spin_order=3)

        # Simulation parameters
        field = 1  # Magnetic field strength in Tesla
        tau_c = 50e-12  # Correlation time in seconds
        temp = 273  # Temperature in Kelvin
        time_step = 2e-3  # Time step in seconds
        nsteps = 50000  # Number of simulation steps

        # Get the Hamiltonian and relaxation superoperator
        H = _hamiltonian.hamiltonian(spin_system, field)
        R = _relaxation.relaxation(spin_system, H, field, tau_c)

        # Create the thermal equilibrium state
        rho = _states.equilibrium_state(spin_system, temp, field)

        # Apply a 180-degree pulse
        pul_180 = _propagation.pulse(spin_system, "I(x,0) + I(x,1) + I(x,2) + I(x,3) + I(x,4) + I(x,5)", angle=180)
        rho = pul_180 @ rho

        # Switch to the zero-quantum (ZQ) subspace
        ZQ_map = truncate_basis_by_coherence(spin_system, [0])
        H, R, rho = transform_to_truncated_basis(ZQ_map, H, R, rho)

        # Thermalize the relaxation superoperator
        R = _relaxation.ldb_thermalization(spin_system, R, field, temp)

        # Get the propagator
        P = _propagation.propagator(time_step, H, R)
        
        # Simulate the evolution of the spin system
        for _ in range(nsteps):
            rho = P @ rho  # Propagate the density matrix

        # Verify that the final state matches the thermal equilibrium state
        self.assertTrue(np.allclose(rho, _states.equilibrium_state(spin_system, temp, field)))