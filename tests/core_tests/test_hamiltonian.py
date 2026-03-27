"""
Tests for Hamiltonian construction in the Spinguin core.
"""

import os
import unittest

import numpy as np
import spinguin as sg
from scipy.sparse import csc_array, load_npz


class TestHamiltonian(unittest.TestCase):
    """
    Test Hamiltonian generation against reference data.
    """

    def _get_test_data_path(
        self,
        filename,
    ):
        """
        Return the absolute path to a file in the test-data directory.

        Parameters
        ----------
        filename : str
            Name of the requested test-data file.

        Returns
        -------
        str
            Absolute path to the requested file.
        """

        # Locate the shared directory that stores the reference test data.
        test_data_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "test_data",
        )

        return os.path.join(test_data_dir, filename)

    def _build_spin_system(
        self,
    ):
        """
        Create the spin system used in the Hamiltonian regression test.

        Returns
        -------
        SpinSystem
            Spin system with a built basis set and assigned interactions.
        """

        # Create the spin system used in the regression test.
        spin_system = sg.SpinSystem(
            ["1H", "1H", "1H", "1H", "1H", "1H", "1H", "14N"]
        )

        # Construct the basis set used in the reference calculation.
        spin_system.basis.max_spin_order = 3
        spin_system.basis.build()

        # Define the chemical shifts in ppm.
        spin_system.chemical_shifts = [
            -22.7,
            -22.7,
            8.34,
            8.34,
            7.12,
            7.12,
            7.77,
            43.60,
        ]

        # Define the scalar couplings in Hz.
        spin_system.J_couplings = [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [-6.53, 0, 0, 0, 0, 0, 0, 0],
            [0.00, 1.66, 0, 0, 0, 0, 0, 0],
            [1.40, 0.00, -0.06, 0, 0, 0, 0, 0],
            [-0.09, 0.35, 6.03, 0.14, 0, 0, 0, 0],
            [0.38, -0.13, 0.09, 5.93, 0.06, 0, 0, 0],
            [0.01, 0.03, 1.12, -0.02, 7.75, -0.01, 0, 0],
            [-0.30, 15.91, 4.47, 0.04, 1.79, 0, -0.46, 0],
        ]

        return spin_system

    def _assert_matches_reference(
        self,
        operator,
        reference,
    ):
        """
        Check that a generated Hamiltonian matches the reference matrix.

        Parameters
        ----------
        operator : sparse matrix
            Generated Hamiltonian superoperator.
        reference : sparse matrix
            Reference Hamiltonian superoperator.

        Returns
        -------
        None
            The assertion is evaluated in place.
        """

        # Compare the generated and reference operators in dense form.
        self.assertTrue(np.allclose(operator.toarray(), reference.toarray()))

    # TODO: Tests for dense: too memory-consuming
    def test_hamiltonian(self):
        """
        Test the Hamiltonian generation against a previously calculated result.
        """

        # Set the global parameters used in the reference calculation.
        sg.parameters.default()
        sg.parameters.magnetic_field = 7e-3

        # Build the spin system used in the regression test.
        spin_system = self._build_spin_system()

        # Load the previously calculated Hamiltonian for comparison.
        hamiltonian_previous = csc_array(
            load_npz(self._get_test_data_path("hamiltonian.npz"))
        )

        # Generate the sparse Hamiltonian and compare it with the reference.
        sg.parameters.sparse_superoperator = True
        hamiltonian_commutator = sg.hamiltonian(spin_system)
        self._assert_matches_reference(
            hamiltonian_commutator,
            hamiltonian_previous,
        )
        # sg.parameters.sparse_superoperator = False
        # hamiltonian_commutator = sg.hamiltonian(spin_system)
        # self.assertTrue(
        #     np.allclose(hamiltonian_commutator, hamiltonian_previous.toarray())
        # )

        # Repeat the calculation to check cache reuse and consistency.
        sg.parameters.sparse_superoperator = True
        hamiltonian_commutator = sg.hamiltonian(spin_system)
        self._assert_matches_reference(
            hamiltonian_commutator,
            hamiltonian_previous,
        )
        # sg.parameters.sparse_superoperator = False
        # hamiltonian_commutator = sg.hamiltonian(spin_system)
        # self.assertTrue(
        #     np.allclose(hamiltonian_commutator, hamiltonian_previous.toarray())
        # )

        # Build the left and right actions separately and recombine them.
        sg.parameters.sparse_superoperator = True
        hamiltonian_left = sg.hamiltonian(spin_system, side="left")
        hamiltonian_right = sg.hamiltonian(spin_system, side="right")
        hamiltonian_commutator = hamiltonian_left - hamiltonian_right
        self._assert_matches_reference(
            hamiltonian_commutator,
            hamiltonian_previous,
        )
        # sg.parameters.sparse_superoperator = False
        # hamiltonian_left = sg.hamiltonian(spin_system, side="left")
        # hamiltonian_right = sg.hamiltonian(spin_system, side="right")
        # hamiltonian_commutator = hamiltonian_left - hamiltonian_right
        # self.assertTrue(
        #     np.allclose(hamiltonian_commutator, hamiltonian_previous.toarray())
        # )

        # Build each interaction contribution separately and sum them.
        sg.parameters.sparse_superoperator = True
        hamiltonian_zeeman = sg.hamiltonian(spin_system, ["zeeman"])
        hamiltonian_chemical_shift = sg.hamiltonian(
            spin_system,
            ["chemical_shift"],
        )
        hamiltonian_j_coupling = sg.hamiltonian(spin_system, ["J_coupling"])
        hamiltonian_commutator = (
            hamiltonian_zeeman
            + hamiltonian_chemical_shift
            + hamiltonian_j_coupling
        )
        self._assert_matches_reference(
            hamiltonian_commutator,
            hamiltonian_previous,
        )
        # sg.parameters.sparse_superoperator = False
        # hamiltonian_zeeman = sg.hamiltonian(spin_system, ["zeeman"])
        # hamiltonian_chemical_shift = sg.hamiltonian(
        #     spin_system,
        #     ["chemical_shift"],
        # )
        # hamiltonian_j_coupling = sg.hamiltonian(spin_system, ["J_coupling"])
        # hamiltonian_commutator = (
        #     hamiltonian_zeeman
        #     + hamiltonian_chemical_shift
        #     + hamiltonian_j_coupling
        # )
        # self.assertTrue(
        #     np.allclose(hamiltonian_commutator, hamiltonian_previous.toarray())
        # )