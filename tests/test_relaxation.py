import unittest
import numpy as np
from hyppy.spin_system import SpinSystem
from hyppy import nmr_isotopes, relaxation, hamiltonian, states, propagation
from hyppy.basis import ZQ_basis, ZQ_filter

class TestRelaxation(unittest.TestCase):

    def test_dd_constant(self):

        # Get gammas in Hz/T
        y_13C = 2*np.pi * nmr_isotopes.ISOTOPES['13C'][1] * 1e6
        y_1H = 2*np.pi * nmr_isotopes.ISOTOPES['1H'][1] * 1e6
        y_15N = 2*np.pi * nmr_isotopes.ISOTOPES['15N'][1] * 1e6

        # Test against tabulated values (in Hz in the book)
        r_13C_13C = 0.153e-9
        r_13C_1H = 0.106e-9
        r_13C_15N = 0.147e-9
        dd_13C_13C = -relaxation.dd_constant(y_13C, y_13C) / r_13C_13C**3 / (2*np.pi)
        dd_13C_1H = -relaxation.dd_constant(y_13C, y_1H) / r_13C_1H**3 / (2*np.pi)
        dd_13C_15N = relaxation.dd_constant(y_13C, y_15N) / r_13C_15N**3 / (2*np.pi)

        # Compare with: Apperley, Harris & Hodgkinson: Solid-state NMR: Basic principles and practice
        self.assertTrue(np.allclose(2.12e3, dd_13C_13C, rtol=0.01))
        self.assertTrue(np.allclose(25.44e3, dd_13C_1H, rtol=0.01))
        self.assertTrue(np.allclose(0.97e3, dd_13C_15N, rtol=0.01))

    def test_thermalization(self):

        # Example system
        isotopes = np.array(['1H', '1H', '1H', '1H', '1H', '14N'])
        chemical_shifts = np.array([8.56, 8.56, 7.47, 7.47, 7.88, 95.94])
        scalar_couplings = np.array([\
            [ 0,     0,      0,      0,      0,      0],
            [-1.04,  0,      0,      0,      0,      0],
            [ 4.85,  1.05,   0,      0,      0,      0],
            [ 1.05,  4.85,   0.71,   0,      0,      0],
            [ 1.24,  1.24,   7.55,   7.55,   0,      0],
            [ 8.16,  8.16,   0.87,   0.87,  -0.19,   0]
        ])
        xyz = np.array([\
            [ 2.0495335, 0.0000000, -1.4916842],
            [-2.0495335, 0.0000000, -1.4916842],
            [ 2.1458878, 0.0000000,  0.9846086],
            [-2.1458878, 0.0000000,  0.9846086],
            [ 0.0000000, 0.0000000,  2.2681296],
            [ 0.0000000, 0.0000000, -1.5987077]
        ])
        shielding = np.zeros((6, 3, 3))
        shielding[5] = np.array([\
            [-406.20, 0.00,   0.00],
            [ 0.00,   299.44, 0.00],
            [ 0.00,   0.00,  -181.07]
        ])
        efg = np.zeros((6, 3, 3))
        efg[5] = np.array([\
            [0.3069, 0.0000,  0.0000],
            [0.0000, 0.7969,  0.0000],
            [0.0000, 0.0000, -1.1037]
        ])
        spin_system = SpinSystem(isotopes, chemical_shifts, scalar_couplings, xyz, shielding, efg, max_spin_order=3)
        field = 1
        tau_c = 50e-12
        temp = 273
        time_step = 2e-3
        nsteps = 50000

        # Get the Hamiltonian
        H = hamiltonian.hamiltonian(spin_system, field)
        R = relaxation.relaxation(spin_system, H, field, tau_c)

        # Create the thermal equilibrium
        rho = states.thermal_equilibrium(spin_system, temp, field)

        # Apply 180-degree pulse
        pul_180 = propagation.pulse(spin_system, 'I_x', [0, 1, 2, 3, 4, 5], angle=180)
        rho = pul_180 @ rho

        # Switch to ZQ subspace
        ZQ_basis(spin_system)
        H = ZQ_filter(spin_system, H)
        R = ZQ_filter(spin_system, R)
        rho = ZQ_filter(spin_system, rho)

        # Thermalize R
        R = relaxation.thermalize(spin_system, R, field, temp)

        # Get the propagator
        P = propagation.propagator(time_step, H, R)
        
        # Simulation
        for _ in range(nsteps):

            # Propagate
            rho = P @ rho

        # Should result in thermal equilibrium
        self.assertTrue(np.allclose(rho, states.thermal_equilibrium(spin_system, temp, field)))