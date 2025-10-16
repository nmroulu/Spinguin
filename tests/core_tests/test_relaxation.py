import unittest
import numpy as np
import scipy.sparse as sp
import os
from spinguin._core.hamiltonian import sop_H
from spinguin._core._relaxation import dd_constant, relaxation
from spinguin._core.propagation import sop_pulse, propagator
from spinguin._core.nmr_isotopes import ISOTOPES
from spinguin._core.basis import make_basis, truncate_basis_by_coherence
from spinguin._core.states import equilibrium_state, state_to_truncated_basis
from spinguin._core._superoperators import sop_to_truncated_basis
import spinguin as sg

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

        # Make the spin system
        spins = np.array([1/2, 1/2, 1/2, 1/2, 1/2, 1])
        max_spin_order = 3
        basis = make_basis(spins, max_spin_order)

        # Get the gyromagnetic ratios (in rad/s/T)
        y_1H = 2*np.pi * ISOTOPES['1H'][1] * 1e6
        y_14N = 2*np.pi * ISOTOPES['14N'][1] * 1e6
        gammas = np.array([y_1H, y_1H, y_1H, y_1H, y_1H, y_14N])

        # Define the chemical shifts (in ppm)
        chemical_shifts = np.array([8.56, 8.56, 7.47, 7.47, 7.88, 95.94])

        # Define scalar couplings (in Hz)
        J_couplings = np.array([
            [ 0,     0,      0,      0,      0,      0],
            [-1.04,  0,      0,      0,      0,      0],
            [ 4.85,  1.05,   0,      0,      0,      0],
            [ 1.05,  4.85,   0.71,   0,      0,      0],
            [ 1.24,  1.24,   7.55,   7.55,   0,      0],
            [ 8.16,  8.16,   0.87,   0.87,  -0.19,   0]
        ])

        # Define Cartesian coordinates of nuclei
        xyz = np.array([
            [ 2.0495335, 0.0000000, -1.4916842],
            [-2.0495335, 0.0000000, -1.4916842],
            [ 2.1458878, 0.0000000,  0.9846086],
            [-2.1458878, 0.0000000,  0.9846086],
            [ 0.0000000, 0.0000000,  2.2681296],
            [ 0.0000000, 0.0000000, -1.5987077]
        ])

        # Define shielding tensors
        shielding = np.zeros((6, 3, 3))
        shielding[5] = np.array([
            [-406.20, 0.00,   0.00],
            [ 0.00,   299.44, 0.00],
            [ 0.00,   0.00,  -181.07]
        ])

        # Define electric field gradient tensors
        efg = np.zeros((6, 3, 3))
        efg[5] = np.array([
            [0.3069, 0.0000,  0.0000],
            [0.0000, 0.7969,  0.0000],
            [0.0000, 0.0000, -1.1037]
        ])

        # Magnetic field (in T)
        B = 1

        # Temperature (in K)
        T = 273

        # Set the environment
        sg.parameters.magnetic_field = B
        sg.parameters.temperature = T

        # Create a SpinSystem
        spin_system = sg.SpinSystem(["1H", "1H", "1H", "1H", "1H", "14N"])
        spin_system.basis.max_spin_order = 3
        spin_system.basis.build()

        # Assign parameters
        spin_system.chemical_shifts = chemical_shifts
        spin_system.J_couplings = J_couplings
        spin_system.xyz = xyz
        spin_system.shielding = shielding
        spin_system.efg = efg

        # Set the relaxation theory
        spin_system.relaxation.theory = "redfield"
        spin_system.relaxation.tau_c = 50e-12
        spin_system.relaxation.sr2k = True
        spin_system.relaxation.thermalization = True

        # Propagation parameters
        time_step = 2e-3    # Time step in seconds
        nsteps = 50000      # Number of simulation steps

        # Get the Hamiltonian
        H = sop_H(
            basis = basis,
            gammas = gammas,
            spins = spins,
            chemical_shifts = chemical_shifts,
            J_couplings = J_couplings,
            B = B,
            side = "comm",
            sparse = True,
            zero_value = 1e-12,
            interactions = ["zeeman", "chemical_shift", "J_coupling"]
        )
        
        # Get the relaxation superoperator
        R = relaxation(spin_system)
        
        # Construct the total Liouvillian
        L = -1j*H - R

        # Create the thermal equilibrium state
        H_left = sop_H(
            basis = basis,
            gammas = gammas,
            spins = spins,
            chemical_shifts = chemical_shifts,
            J_couplings = J_couplings,
            B = B,
            side = "left",
            sparse = True,
            zero_value = 1e-12,
            interactions = ["zeeman", "chemical_shift", "J_coupling"]
        )
        rho = equilibrium_state(basis, spins, H_left, T, sparse=False,
                                zero_value=1e-18)

        # Apply a 180-degree pulse (to all 1H)
        op_string = "I(x,0) + I(x,1) + I(x,2) + I(x,3) + I(x,4) + I(x,5)"
        pul_180 = sop_pulse(basis, spins, op_string, angle=180, sparse=True,
                            zero_value=1e-18)
        rho = pul_180 @ rho

        # Switch to the zero-quantum (ZQ) subspace
        ZQ_basis, ZQ_map = truncate_basis_by_coherence(basis,
                                                       coherence_orders=[0])
        L = sop_to_truncated_basis(ZQ_map, L)
        rho = state_to_truncated_basis(ZQ_map, rho)

        # Get the propagator
        P = propagator(L, time_step)
        
        # Simulate the evolution of the spin system
        for _ in range(nsteps):
            rho = P @ rho

        # Create an equilibrium state in the ZQ basis for reference
        H_left_ZQ = sop_H(
            basis = ZQ_basis,
            gammas = gammas,
            spins = spins,
            chemical_shifts = chemical_shifts,
            J_couplings = J_couplings,
            B = B,
            side = "left",
            sparse = True,
            zero_value = 1e-12,
            interactions = ["zeeman", "chemical_shift", "J_coupling"]
        )
        rho_ref = equilibrium_state(ZQ_basis, spins, H_left_ZQ, T, sparse=False,
                                    zero_value=1e-18)

        # Verify that the final state matches the thermal equilibrium state
        self.assertTrue(np.allclose(rho, rho_ref))

    def test_sop_R_redfield(self):
        """
        Test that creates a relaxation superoperator using Redfield theory and
        compares that to a previously calculated value.
        """

        # Make the spin system
        spins = np.array([1/2, 1/2, 1/2, 1/2, 1/2, 1])
        max_spin_order = 3
        basis = make_basis(spins, max_spin_order)

        # Get the gyromagnetic ratios (in rad/s/T)
        y_1H = 2*np.pi * ISOTOPES['1H'][1] * 1e6
        y_14N = 2*np.pi * ISOTOPES['14N'][1] * 1e6
        gammas = np.array([y_1H, y_1H, y_1H, y_1H, y_1H, y_14N])

        # Define the chemical shifts (in ppm)
        chemical_shifts = np.array([8.56, 8.56, 7.47, 7.47, 7.88, 95.94])

        # Define scalar couplings (in Hz)
        J_couplings = np.array([
            [ 0,     0,      0,      0,      0,      0],
            [-1.04,  0,      0,      0,      0,      0],
            [ 4.85,  1.05,   0,      0,      0,      0],
            [ 1.05,  4.85,   0.71,   0,      0,      0],
            [ 1.24,  1.24,   7.55,   7.55,   0,      0],
            [ 8.16,  8.16,   0.87,   0.87,  -0.19,   0]
        ])

        # Define Cartesian coordinates of nuclei
        xyz = np.array([
            [ 2.0495335, 0.0000000, -1.4916842],
            [-2.0495335, 0.0000000, -1.4916842],
            [ 2.1458878, 0.0000000,  0.9846086],
            [-2.1458878, 0.0000000,  0.9846086],
            [ 0.0000000, 0.0000000,  2.2681296],
            [ 0.0000000, 0.0000000, -1.5987077]
        ])

        # Define shielding tensors
        shielding = np.zeros((6, 3, 3))
        shielding[5] = np.array([
            [-406.20, 0.00,   0.00],
            [ 0.00,   299.44, 0.00],
            [ 0.00,   0.00,  -181.07]
        ])

        # Define electric field gradient tensors
        efg = np.zeros((6, 3, 3))
        efg[5] = np.array([
            [0.3069, 0.0000,  0.0000],
            [0.0000, 0.7969,  0.0000],
            [0.0000, 0.0000, -1.1037]
        ])

        # Set the environment
        sg.parameters.magnetic_field = 1

        # Create a SpinSystem
        spin_system = sg.SpinSystem(["1H", "1H", "1H", "1H", "1H", "14N"])
        spin_system.basis.max_spin_order = 3
        spin_system.basis.build()

        # Assign parameters
        spin_system.chemical_shifts = chemical_shifts
        spin_system.J_couplings = J_couplings
        spin_system.xyz = xyz
        spin_system.shielding = shielding
        spin_system.efg = efg

        # Set the relaxation theory
        spin_system.relaxation.theory = "redfield"
        spin_system.relaxation.tau_c = 50e-12
        spin_system.relaxation.sr2k = False
        spin_system.relaxation.thermalization = False
        
        # Get the Redfield relaxation superoperator
        R = relaxation(spin_system)
        
        # Obtain R again (to check possible errors from caches etc.)
        R = relaxation(spin_system)

        # Load the previously calculated R for comparison
        test_dir = os.path.dirname(os.path.dirname(__file__))
        R_previous = sp.load_npz(os.path.join(
            test_dir, 'test_data', 'relaxation.npz'))

        # Assert that the generated Hamiltonian matches the reference
        self.assertTrue(np.allclose(R.toarray(), R_previous.toarray()))

        # Obtain R again after basis truncation that causes some operators to be
        # zero
        basis, _ = truncate_basis_by_coherence(basis, [0])
        R = relaxation(spin_system)

    def test_sop_R_phenomenological(self):
        """
        Test that creates the phenomenological relaxation superoperator and
        makes a comparison to a previously computed result.
        """

        # Make the spin system
        spin_system = sg.SpinSystem(["1H", "1H", "1H", "1H", "1H", "14N"])
        spin_system.basis.max_spin_order = 3
        spin_system.basis.build()
        
        # Set the relaxation theory
        spin_system.relaxation.theory = "phenomenological"
        spin_system.relaxation.T1 = np.array([5, 5, 5, 5, 5, 1e-3])
        spin_system.relaxation.T2 = np.array([5, 5, 5, 5, 5, 1e-3])

        # Get the relaxation superoperator
        R = relaxation(spin_system)

        # Obtain R again (check for cache errors etc.)
        R = relaxation(spin_system)

        # Load the previously calculated R for comparison
        test_dir = os.path.dirname(os.path.dirname(__file__))
        R_previous = sp.load_npz(os.path.join(
            test_dir, 'test_data', 'relaxation_phenomenological.npz'))

        # Assert that the generated Hamiltonian matches the reference
        self.assertTrue(np.allclose(R.toarray(), R_previous.toarray()))

    def test_sop_R_sr2k(self):
        """
        Test that adds the contribution from SR2K to a phenomenological
        relaxation superoperator and compares the result to a pre-computed
        value.
        """

        # Make the spin system
        spin_system = sg.SpinSystem(["1H", "1H", "1H", "1H", "1H", "14N"])
        spin_system.basis.max_spin_order = 3
        spin_system.basis.build()

        # Define the chemical shifts (in ppm)
        spin_system.chemical_shifts = [8.56, 8.56, 7.47, 7.47, 7.88, 95.94]

        # Define scalar couplings (in Hz)
        spin_system.J_couplings = [
            [ 0,     0,      0,      0,      0,      0],
            [-1.04,  0,      0,      0,      0,      0],
            [ 4.85,  1.05,   0,      0,      0,      0],
            [ 1.05,  4.85,   0.71,   0,      0,      0],
            [ 1.24,  1.24,   7.55,   7.55,   0,      0],
            [ 8.16,  8.16,   0.87,   0.87,  -0.19,   0]
        ]

        # Magnetic field (in T)
        sg.parameters.magnetic_field = 3e-6

        # Set the relaxation theory
        spin_system.relaxation.theory = "phenomenological"
        spin_system.relaxation.T1 = np.array([5, 5, 5, 5, 5, 1e-3])
        spin_system.relaxation.T2 = np.array([5, 5, 5, 5, 5, 1e-3])
        spin_system.relaxation.sr2k = True
        
        # Get the relaxation superoperator
        R = relaxation(spin_system)

        # Get the relaxation superoperator again (to check cache problems)
        R = relaxation(spin_system)

        # Load the previously calculated R for comparison
        test_dir = os.path.dirname(os.path.dirname(__file__))
        R_previous = sp.load_npz(
            os.path.join(test_dir, 'test_data', 'relaxation_sr2k.npz'))

        # Assert that the generated Hamiltonian matches the reference
        self.assertTrue(np.allclose(R.toarray(), R_previous.toarray()))