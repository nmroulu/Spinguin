"""
Tests for state propagation utilities.
"""

import unittest

import numpy as np

import spinguin as sg


class TestPropagation(unittest.TestCase):
    """
    Test pulse-generated state propagation.
    """

    def _build_spin_system(
        self,
    ):
        """
        Create the spin system used in the pulse propagation test.

        Returns
        -------
        SpinSystem
            Spin system with a built basis set.
        """

        # Create the spin system used in the test.
        spin_system = sg.SpinSystem(["1H", "1H", "14N"])

        # Build the basis set required by the pulse calculations.
        spin_system.basis.max_spin_order = 2
        spin_system.basis.build()

        return spin_system

    def _assert_allclose(
        self,
        value,
        reference,
    ):
        """
        Check that two propagated states are numerically equal.

        Parameters
        ----------
        value : array-like
            Tested propagated state.
        reference : array-like
            Reference state.

        Returns
        -------
        None
            The assertion is evaluated in place.
        """

        # Compare the propagated state with its reference value.
        self.assertTrue(np.allclose(value, reference))

    def test_pulse(self):
        """
        Test the behavior of pulses on spin states.
        """

        # Reset parameters to defaults.
        sg.parameters.default()

        # Build the spin system used in the test.
        spin_system = self._build_spin_system()

        # Create the initial basis states used in the pulse checks.
        rho_x = sg.state(spin_system, "I(x,0)")
        rho_y = sg.state(spin_system, "I(y,0)")
        rho_z = sg.state(spin_system, "I(z,0)")
        rho_xz = sg.state(spin_system, "I(x,0)*I(z,1)")
        rho_yz = sg.state(spin_system, "I(y,0)*I(z,1)")

        # Create the pulse propagators in dense format.
        sg.parameters.sparse_superoperator = False
        pul_90_x_dense = sg.pulse(spin_system, "I(x,0)", angle=90)
        pul_90_y_dense = sg.pulse(spin_system, "I(y,0)", angle=90)
        pul_90_z_dense = sg.pulse(spin_system, "I(z,0)", angle=90)
        pul_180_x_dense = sg.pulse(spin_system, "I(x,0)", angle=180)
        pul_180_y_dense = sg.pulse(spin_system, "I(y,0)", angle=180)
        pul_180_z_dense = sg.pulse(spin_system, "I(z,0)", angle=180)
        with self.assertWarns(Warning):
            pul_180_zz_dense = sg.pulse(
                spin_system,
                "I(z,0)*I(z,1)",
                angle=180,
            )

        # Create the pulse propagators in sparse format.
        sg.parameters.sparse_superoperator = True
        pul_90_x_sparse = sg.pulse(spin_system, "I(x,0)", angle=90)
        pul_90_y_sparse = sg.pulse(spin_system, "I(y,0)", angle=90)
        pul_90_z_sparse = sg.pulse(spin_system, "I(z,0)", angle=90)
        pul_180_x_sparse = sg.pulse(spin_system, "I(x,0)", angle=180)
        pul_180_y_sparse = sg.pulse(spin_system, "I(y,0)", angle=180)
        pul_180_z_sparse = sg.pulse(spin_system, "I(z,0)", angle=180)
        with self.assertWarns(Warning):
            pul_180_zz_sparse = sg.pulse(
                spin_system,
                "I(z,0)*I(z,1)",
                angle=180,
            )

        # Verify the propagated dense states.
        self._assert_allclose(-rho_y, pul_90_x_dense @ rho_z)
        self._assert_allclose(rho_x, pul_90_y_dense @ rho_z)
        self._assert_allclose(rho_z, pul_90_z_dense @ rho_z)
        self._assert_allclose(-rho_z, pul_180_x_dense @ rho_z)
        self._assert_allclose(-rho_z, pul_180_y_dense @ rho_z)
        self._assert_allclose(rho_z, pul_180_z_dense @ rho_z)
        self._assert_allclose(2 * rho_yz, pul_180_zz_dense @ rho_x)
        self._assert_allclose(-rho_x, pul_180_zz_dense @ (2 * rho_yz))
        self._assert_allclose(-2 * rho_yz, pul_180_zz_dense @ (-rho_x))
        self._assert_allclose(rho_x, pul_180_zz_dense @ (-2 * rho_yz))
        self._assert_allclose(-2 * rho_xz, pul_180_zz_dense @ rho_y)
        self._assert_allclose(-rho_y, pul_180_zz_dense @ (-2 * rho_xz))
        self._assert_allclose(2 * rho_xz, pul_180_zz_dense @ (-rho_y))
        self._assert_allclose(rho_y, pul_180_zz_dense @ (2 * rho_xz))

        # Verify the propagated sparse states.
        self._assert_allclose(-rho_y, pul_90_x_sparse @ rho_z)
        self._assert_allclose(rho_x, pul_90_y_sparse @ rho_z)
        self._assert_allclose(rho_z, pul_90_z_sparse @ rho_z)
        self._assert_allclose(-rho_z, pul_180_x_sparse @ rho_z)
        self._assert_allclose(-rho_z, pul_180_y_sparse @ rho_z)
        self._assert_allclose(rho_z, pul_180_z_sparse @ rho_z)
        self._assert_allclose(2 * rho_yz, pul_180_zz_sparse @ rho_x)
        self._assert_allclose(-rho_x, pul_180_zz_sparse @ (2 * rho_yz))
        self._assert_allclose(-2 * rho_yz, pul_180_zz_sparse @ (-rho_x))
        self._assert_allclose(rho_x, pul_180_zz_sparse @ (-2 * rho_yz))
        self._assert_allclose(-2 * rho_xz, pul_180_zz_sparse @ rho_y)
        self._assert_allclose(-rho_y, pul_180_zz_sparse @ (-2 * rho_xz))
        self._assert_allclose(2 * rho_xz, pul_180_zz_sparse @ (-rho_y))
        self._assert_allclose(rho_y, pul_180_zz_sparse @ (2 * rho_xz))