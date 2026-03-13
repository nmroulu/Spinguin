import unittest
import numpy as np
import spinguin as sg

class TestRotframe(unittest.TestCase):

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
        ss.relaxation.thermalization = True
        ss.relaxation.tau_c = 50e-12
        
        # Get the Liouvillian
        H = sg.hamiltonian(ss)
        R = sg.relaxation(ss)
        L = sg.liouvillian(H, R)

        # Obtain the equilibrium state
        rho = sg.equilibrium_state(ss)

        # Apply pulse for protons
        indices = np.where(ss.isotopes == "1H")[0]
        op_pulse = "+".join(f"I(y,{i})" for i in indices)
        Px = sg.pulse(ss, op_pulse, 90)
        rho = Px @ rho

        # Dwell time and number of points
        dt = 1e-3
        npoints = 200

        # Center frequencies for 1H and 14N (in ppm)
        cH = 8
        cN = -200

        # Construct the numerically exact propagator in the rotating frame
        P_ref = sg.propagator(L, dt)
        P_ref = sg.propagator_to_rotframe(ss, P_ref, dt, {"1H": cH, "14N": cN})

        # Calculate first the rotating frame Liouvillian and then the propagator
        L_rot = sg.rotating_frame(ss, L, ["1H", "14N"], [cH, cN])
        P_rot = sg.propagator(L_rot, dt)

        # Construct the operator to be measured
        op_measure = "+".join(f"I(-,{i})" for i in indices)

        # Initialize an array for storing results
        fid_ref = np.zeros(npoints, dtype=complex)
        fid_rot = np.zeros(npoints, dtype=complex)

        # Perform the time evolution
        rho_ref = rho
        rho_rot = rho
        for step in range(npoints):
            fid_ref[step] = sg.measure(ss, rho_ref, op_measure)
            fid_rot[step] = sg.measure(ss, rho_rot, op_measure)
            rho_ref = P_ref @ rho_ref
            rho_rot = P_rot @ rho_rot

        # Check that the FIDs match
        self.assertTrue(np.allclose(fid_ref, fid_rot))