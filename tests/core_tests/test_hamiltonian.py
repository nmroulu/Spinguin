"""
Tests for Hamiltonian construction in the Spinguin core.
"""

import unittest

import numpy as np
import spinguin as sg
from ._helpers import build_spin_system, test_data_path
from scipy.sparse import csc_array, load_npz


class TestHamiltonian(unittest.TestCase):

    def test_hamiltonian(self):
        """
        Test the Hamiltonian generation against a previously calculated result.
        """

        # Set the global parameters used in the reference calculation.
        sg.parameters.default()
        sg.parameters.magnetic_field = 7e-3
        
        # Build the test spin system.
        ss = build_spin_system(["1H", "1H", "14N"], 3)

        # Define the chemical shifts in ppm.
        ss.chemical_shifts = [4, 6, 150]

        # Define the scalar couplings in Hz.
        ss.J_couplings = [
            [0,  0,  0],
            [8,  0,  0],
            [2,  3,  0],
        ]

        # Load the previously calculated Hamiltonian for comparison.
        H_previous = csc_array(load_npz(test_data_path("hamiltonian.npz")))
        
        # Generate the sparse Hamiltonian and compare it with the reference.
        sg.parameters.sparse_superoperator = True
        H_comm = sg.hamiltonian(ss)
        self.assertTrue(np.allclose(H_comm.toarray(), H_previous.toarray()))

        # Generate the dense Hamiltonian and compare it with the reference.
        sg.parameters.sparse_superoperator = False
        H_comm = sg.hamiltonian(ss)
        self.assertTrue(np.allclose(H_comm, H_previous.toarray()))

        # Repeat the calculation to check cache reuse and consistency.
        sg.parameters.sparse_superoperator = True
        H_comm = sg.hamiltonian(ss)
        self.assertTrue(np.allclose(H_comm.toarray(), H_previous.toarray()))
        sg.parameters.sparse_superoperator = False
        H_comm = sg.hamiltonian(ss)
        self.assertTrue(np.allclose(H_comm, H_previous.toarray()))

        # Build the left and right actions separately and recombine them.
        sg.parameters.sparse_superoperator = True
        H_left = sg.hamiltonian(ss, side='left')
        H_right = sg.hamiltonian(ss, side='right')
        H_comm = H_left - H_right
        self.assertTrue(np.allclose(H_comm.toarray(), H_previous.toarray()))
        sg.parameters.sparse_superoperator = False
        H_left = sg.hamiltonian(ss, side='left')
        H_right = sg.hamiltonian(ss, side='right')
        H_comm = H_left - H_right
        self.assertTrue(np.allclose(H_comm, H_previous.toarray()))

        # Build each interaction contribution separately and sum them.
        sg.parameters.sparse_superoperator = True
        H_comm_Z = sg.hamiltonian(ss, ["zeeman"])
        H_comm_CS = sg.hamiltonian(ss, ["chemical_shift"])
        H_comm_J = sg.hamiltonian(ss, ["J_coupling"])
        H_comm = H_comm_Z + H_comm_CS + H_comm_J
        self.assertTrue(np.allclose(H_comm.toarray(), H_previous.toarray()))
        sg.parameters.sparse_superoperator = False
        H_comm_Z = sg.hamiltonian(ss, ["zeeman"])
        H_comm_CS = sg.hamiltonian(ss, ["chemical_shift"])
        H_comm_J = sg.hamiltonian(ss, ["J_coupling"])
        H_comm = H_comm_Z + H_comm_CS + H_comm_J
        self.assertTrue(np.allclose(H_comm, H_previous.toarray()))

    def test_rotating_frame_irradiation_hamiltonian(self):
        """
        Test the rotating-frame irradiation Hamiltonian against hard pulses,
        analytic rotations and analytic off-resonance nutation, and test that
        only the requested isotope is irradiated.
        """

        # Set the global parameters used in the test.
        sg.parameters.default()
        sg.parameters.magnetic_field = 1.0
        sg.parameters.temperature = 295

        # Amplitude of the linearly polarised irradiation field in T.
        B1 = 2e-6

        # Test with dense and sparse backends.
        for sparse in [False, True]:
            sg.parameters.sparse_superoperator = sparse

            # Test isotopes with both signs of the gyromagnetic ratio.
            for isotope in ["1H", "13C", "129Xe"]:

                # Build a single-spin system and its basis states.
                ss = build_spin_system([isotope], 1)
                rho_x = sg.state(ss, "I(x,0)")
                rho_y = sg.state(ss, "I(y,0)")
                rho_z = sg.state(ss, "I(z,0)")

                # Transform the Zeeman Liouvillian to the on-resonance frame.
                L = sg.liouvillian(sg.hamiltonian(ss))
                L = sg.rotating_frame(ss, L, [isotope], 0)

                # Calculate the nutation frequency and the 90-degree time.
                w1 = 1 / 2 * B1 * abs(sg.gamma(isotope, "rad/s"))
                t_90 = np.pi / 2 / w1

                # Irradiation with phases 0 and 90 must equal hard pulses.
                for phase, operator in [(0, "I(x,0)"), (90, "I(y,0)")]:
                    H_rf = sg.rotating_frame_irradiation_hamiltonian(
                        ss, isotope, B1, phase
                    )
                    P = sg.propagator(L - 1j * H_rf, t_90)
                    pul = sg.pulse(ss, operator, angle=90)
                    self.assertTrue(np.allclose(P @ rho_z, pul @ rho_z))

                # Irradiation with an arbitrary phase and angle must rotate the
                # z-magnetisation about the corresponding transverse axis.
                phase = 37
                angle = 2 / 3 * np.pi
                H_rf = sg.rotating_frame_irradiation_hamiltonian(
                    ss, isotope, B1, phase
                )
                P = sg.propagator(L - 1j * H_rf, angle / w1)
                phi = phase * np.pi / 180
                rho_ref = (
                    np.cos(angle) * rho_z
                    + np.sin(angle) * (np.sin(phi)*rho_x - np.cos(phi)*rho_y)
                )
                self.assertTrue(np.allclose(P @ rho_z, rho_ref))

                # Off-resonance irradiation must nutate the z-magnetisation
                # about the tilted effective field.
                delta = 1.0
                ss.chemical_shifts = [delta]
                L = sg.liouvillian(sg.hamiltonian(ss))
                L = sg.rotating_frame(ss, L, [isotope], 0)
                H_rf = sg.rotating_frame_irradiation_hamiltonian(
                    ss, isotope, B1
                )
                offset = (
                    sg.resonance_frequency(isotope, delta=delta, unit="rad/s")
                    - sg.resonance_frequency(isotope, unit="rad/s")
                )
                w_eff = np.sqrt(offset**2 + w1**2)
                Mz_0 = sg.measure(ss, rho_z, "I(z,0)")
                for t in [0.3 / w_eff, 2.0 / w_eff, 7.0 / w_eff]:
                    P = sg.propagator(L - 1j * H_rf, t)
                    Mz = sg.measure(ss, P @ rho_z, "I(z,0)") / Mz_0
                    Mz_ref = (offset**2 + w1**2 * np.cos(w_eff*t)) / w_eff**2
                    self.assertAlmostEqual(Mz.real, Mz_ref, places=6)

            # Build a heteronuclear two-spin system.
            ss = build_spin_system(["1H", "13C"], 2)
            rho_z_H = sg.state(ss, "I(z,0)")
            rho_z_C = sg.state(ss, "I(z,1)")

            # Irradiate only 1H for a 90-degree nutation of 1H.
            L = sg.liouvillian(sg.hamiltonian(ss))
            L = sg.rotating_frame(ss, L, ["1H", "13C"], [0, 0])
            H_rf = sg.rotating_frame_irradiation_hamiltonian(ss, "1H", B1, 90)
            w1 = 1 / 2 * B1 * abs(sg.gamma("1H", "rad/s"))
            P = sg.propagator(L - 1j * H_rf, np.pi / 2 / w1)

            # The 1H magnetisation is rotated and the 13C one is unaffected.
            pul = sg.pulse(ss, "I(y,0)", angle=90)
            self.assertTrue(np.allclose(P @ rho_z_H, pul @ rho_z_H))
            self.assertTrue(np.allclose(P @ rho_z_C, rho_z_C))


    def test_irradiation_steady_state(self):
        """
        Test the steady-state longitudinal magnetisation under continuous-wave
        irradiation with thermalised phenomenological relaxation against the
        steady-state solution of the Bloch equations.
        """

        # Set the global parameters used in the test.
        sg.parameters.default()
        sg.parameters.magnetic_field = 1.0
        sg.parameters.temperature = 295

        # Define the relaxation times in s and the irradiation amplitude in T.
        T1 = 2.0
        T2 = 0.5
        B1 = 2e-7

        # Test with dense and sparse backends.
        for sparse in [False, True]:
            sg.parameters.sparse_superoperator = sparse

            # Test isotopes with both signs of the gyromagnetic ratio.
            for isotope in ["1H", "129Xe"]:

                # Build a single-spin system with phenomenological relaxation.
                ss = build_spin_system([isotope], 1)
                ss.relaxation.theory = "phenomenological"
                ss.relaxation.T1 = [T1]
                ss.relaxation.T2 = [T2]

                # Calculate the nutation frequency of the irradiation.
                w1 = 1 / 2 * B1 * abs(sg.gamma(isotope, "rad/s"))

                # Compare the steady state on and off resonance.
                for delta in [0.0, 0.05, 0.3]:

                    # Build the irradiated Liouvillian in the rotating frame.
                    ss.chemical_shifts = [delta]
                    L = sg.liouvillian(sg.hamiltonian(ss), sg.relaxation(ss))
                    L = sg.rotating_frame(ss, L, [isotope], 0)
                    H_rf = sg.rotating_frame_irradiation_hamiltonian(
                        ss, isotope, B1
                    )
                    L = L - 1j * H_rf

                    # Propagate the thermal equilibrium to the steady state.
                    rho_eq = sg.equilibrium_state(ss)
                    P = sg.propagator(L, 40 * T1)
                    Mz = (
                        sg.measure(ss, P @ rho_eq, "I(z)")
                        / sg.measure(ss, rho_eq, "I(z)")
                    )

                    # Calculate the steady state of the Bloch equations.
                    offset = (
                        sg.resonance_frequency(isotope, delta=delta, unit="rad/s")
                        - sg.resonance_frequency(isotope, unit="rad/s")
                    )
                    Mz_ref = (
                        (1 + offset**2 * T2**2)
                        / (1 + offset**2 * T2**2 + w1**2 * T1 * T2)
                    )
                    self.assertAlmostEqual(Mz.real, Mz_ref, places=6)
