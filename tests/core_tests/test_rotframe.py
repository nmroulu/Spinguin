"""
Tests for rotating-frame transformations and propagators.
"""

import unittest

import numpy as np

import spinguin as sg


class TestRotframe(unittest.TestCase):
    """
    Test rotating-frame operations against direct reference propagation.
    """

    def _assert_allclose(
        self,
        value,
        reference,
        rtol=1e-05,
        atol=1e-08,
    ):
        """
        Assert that two numerical arrays are equal within tolerance.

        Parameters
        ----------
        value : array_like
            Tested numerical data.
        reference : array_like
            Reference numerical data.
        rtol : float, optional
            Relative tolerance passed to `numpy.allclose`.
        atol : float, optional
            Absolute tolerance passed to `numpy.allclose`.

        Returns
        -------
        None
            The assertion is evaluated in place.
        """

        # Compare the tested data with the reference data.
        self.assertTrue(np.allclose(value, reference, rtol=rtol, atol=atol))

    def _build_spin_system(
        self,
    ):
        """
        Create the shared six-spin system used in the rotating-frame test.

        Returns
        -------
        SpinSystem
            Spin system with basis, interactions, shielding, and EFG tensors.
        """

        # Create the proton-nitrogen spin system.
        spin_system = sg.SpinSystem(["1H", "1H", "1H", "1H", "1H", "14N"])

        # Build the basis set used in the simulation.
        spin_system.basis.max_spin_order = 3
        spin_system.basis.build()

        # Define the chemical shifts in ppm.
        spin_system.chemical_shifts = [8.56, 8.56, 7.47, 7.47, 7.88, 95.94]

        # Define the scalar couplings in hertz.
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

    def test_rotating_frame(self):
        """
        Test the rotating-frame implementation against direct propagation.
        """

        # Set the global simulation parameters.
        sg.parameters.default()
        sg.parameters.magnetic_field = 1.0
        sg.parameters.temperature = 295

        # Build the shared spin system.
        spin_system = self._build_spin_system()

        # Define the Redfield relaxation model.
        spin_system.relaxation.theory = "redfield"
        spin_system.relaxation.thermalization = True
        spin_system.relaxation.tau_c = 50e-12

        # Build the Liouvillian used in both simulations.
        hamiltonian = sg.hamiltonian(spin_system)
        relaxation_superoperator = sg.relaxation(spin_system)
        liouvillian = sg.liouvillian(hamiltonian, relaxation_superoperator)

        # Build the initial state and excite the proton spins.
        state = sg.equilibrium_state(spin_system)
        proton_indices = np.where(spin_system.isotopes == "1H")[0]
        pulse_operator = "+".join(f"I(y,{index})" for index in proton_indices)
        pulse_x = sg.pulse(spin_system, pulse_operator, 90)
        state = pulse_x @ state

        # Define the acquisition parameters.
        dwell_time = 1e-3
        npoints = 200

        # Define the rotating-frame carrier frequencies in ppm.
        proton_centre = 8
        nitrogen_centre = -200

        # Build the reference propagator in the rotating frame.
        propagator_reference = sg.propagator(liouvillian, dwell_time)
        propagator_reference = sg.propagator_to_rotframe(
            spin_system,
            propagator_reference,
            dwell_time,
            {"1H": proton_centre, "14N": nitrogen_centre},
        )

        # Build the propagator from the rotating-frame Liouvillian.
        liouvillian_rotating = sg.rotating_frame(
            spin_system,
            liouvillian,
            ["1H", "14N"],
            [proton_centre, nitrogen_centre],
        )
        propagator_rotating = sg.propagator(liouvillian_rotating, dwell_time)

        # Build the measurement operator for the proton signal.
        measure_operator = "+".join(
            f"I(-,{index})" for index in proton_indices
        )

        # Allocate arrays for the reference and rotating-frame FIDs.
        fid_reference = np.zeros(npoints, dtype=complex)
        fid_rotating = np.zeros(npoints, dtype=complex)

        # Propagate the states and record both FIDs.
        state_reference = state
        state_rotating = state
        for step in range(npoints):
            fid_reference[step] = sg.measure(
                spin_system,
                state_reference,
                measure_operator,
            )
            fid_rotating[step] = sg.measure(
                spin_system,
                state_rotating,
                measure_operator,
            )
            state_reference = propagator_reference @ state_reference
            state_rotating = propagator_rotating @ state_rotating

        # Verify that both rotating-frame treatments give identical FIDs.
        self._assert_allclose(fid_reference, fid_rotating)