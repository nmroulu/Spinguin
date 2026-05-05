"""
Tests for rotating-frame transformations and propagators.
"""

import unittest

import numpy as np

import spinguin as sg
from ._helpers import build_spin_system

class TestRotframe(unittest.TestCase):
    """
    Test rotating-frame operations against direct reference propagation.
    """

    def test_rotating_frame(self):
        """
        Test the rotating frame implementation against a numerically exact
        solution in a simple simulation.
        """
        # Set the global parameters
        sg.parameters.default()
        sg.parameters.magnetic_field = 1.0
        sg.parameters.temperature = 295

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

        # Define the relaxation theory
        ss.relaxation.theory = "redfield"
        ss.relaxation.tau_c = 50e-12
        
        # Build the Liouvillian
        H = sg.hamiltonian(ss)
        R = sg.relaxation(ss)
        L = sg.liouvillian(H, R)

        # Build the initial state
        rho = sg.equilibrium_state(ss)

        # Excite the proton spins.
        indices = np.where(ss.isotopes == "1H")[0]
        op_pulse = "+".join(f"I(y,{i})" for i in indices)
        Px = sg.pulse(ss, op_pulse, 90)
        rho = Px @ rho

        # Define the acquisition parameters.
        dt = 1e-3
        npoints = 200

        # Define the rotating-frame carrier frequencies in ppm.
        cH = 8
        cN = -200

        # Build the reference propagator in the rotating frame.
        P_ref = sg.propagator(L, dt)
        P_ref = sg.propagator_to_rotframe(ss, P_ref, dt, {"1H": cH, "14N": cN})

        # Build the propagator from the rotating-frame Liouvillian.
        L_rot = sg.rotating_frame(ss, L, ["1H", "14N"], [cH, cN])
        P_rot = sg.propagator(L_rot, dt)

        # Build the measurement operator for the proton signal.
        op_measure = "+".join(f"I(-,{i})" for i in indices)

        # Allocate arrays for the reference and rotating-frame FIDs.
        fid_ref = np.zeros(npoints, dtype=complex)
        fid_rot = np.zeros(npoints, dtype=complex)

        # Propagate the states and record both FIDs.
        rho_ref = rho
        rho_rot = rho
        for step in range(npoints):
            fid_ref[step] = sg.measure(ss, rho_ref, op_measure)
            fid_rot[step] = sg.measure(ss, rho_rot, op_measure)
            rho_ref = P_ref @ rho_ref
            rho_rot = P_rot @ rho_rot

        # Verify that both rotating-frame treatments give identical FIDs.
        self.assertTrue(np.allclose(fid_ref, fid_rot))