"""
Tests for superoperator construction, representation, and basis truncation.
"""

from typing import Literal

import unittest

import numpy as np

import spinguin as sg
from spinguin._core._la import cartesian_tensor_to_spherical_tensor
from spinguin._core._superoperators import structure_coefficients


class TestSuperoperators(unittest.TestCase):
    """
    Test superoperator generation against reference constructions.
    """

    def _assert_allclose(
        self,
        value,
        reference,
        rtol=1e-05,
        atol=1e-08,
    ):
        """
        Assert that two arrays or sparse matrices agree numerically.
        """

        # Convert sparse matrices to dense arrays when necessary.
        if hasattr(value, "toarray"):
            value = value.toarray()
        if hasattr(reference, "toarray"):
            reference = reference.toarray()

        # Compare the tested and reference values.
        self.assertTrue(np.allclose(value, reference, rtol=rtol, atol=atol))

    def _build_superoperators_from_basis(
        self,
        spin_system,
    ):
        """
        Build superoperators for every basis operator definition.
        """

        # Construct all superoperators from the current basis.
        return [
            sg.superoperator(spin_system, operator_definition)
            for operator_definition in spin_system.basis.basis
        ]

    def _build_operator_strings(
        self,
        labels,
    ):
        """
        Generate three-spin operator strings for the given labels.
        """

        # Build all three-spin product-operator strings.
        operator_strings = []
        for label_i in labels:
            operator_i = "E" if label_i == "E" else f"I({label_i}, 0)"
            for label_j in labels:
                operator_j = "E" if label_j == "E" else f"I({label_j}, 1)"
                for label_k in labels:
                    operator_k = "E" if label_k == "E" else f"I({label_k}, 2)"
                    operator_strings.append(
                        f"{operator_i} * {operator_j} * {operator_k}"
                    )

        return operator_strings

    def _build_manual_side_superoperator(
        self,
        spin_system,
        operator_definition,
        side,
    ):
        """
        Build a left or right superoperator by explicit matrix algebra.
        """

        # Initialise the reference superoperator matrix.
        superoperator_reference = np.zeros(
            (spin_system.basis.dim, spin_system.basis.dim),
            dtype=complex,
        )

        # Construct the operator that defines the superoperator.
        operator_i = sg.operator(spin_system, operator_definition)

        # Loop over the operator bras.
        for bra_index, bra_definition in enumerate(spin_system.basis.basis):
            operator_j = sg.operator(spin_system, bra_definition)

            # Loop over the operator kets.
            for ket_index, ket_definition in enumerate(spin_system.basis.basis):
                operator_k = sg.operator(spin_system, ket_definition)

                # Calculate the operator-space normalisation factor.
                norm = np.sqrt(
                    (operator_j.conj().T @ operator_j).trace() *
                    (operator_k.conj().T @ operator_k).trace()
                )

                # Calculate the requested matrix element.
                if side == "left":
                    element = (operator_j.conj().T @ operator_i @ operator_k).trace()
                else:
                    element = (operator_j.conj().T @ operator_k @ operator_i).trace()

                superoperator_reference[bra_index, ket_index] = element / norm

        return superoperator_reference

    def test_superoperator_1(self):
        """
        Test that the operator sparsity does not influence the output.
        """

        # Reset parameters to defaults.
        sg.parameters.default()

        # Create and build the example spin system.
        spin_system = sg.SpinSystem(["1H", "14N", "23Na"])
        spin_system.basis.max_spin_order = 2
        spin_system.basis.build()

        # Build all basis superoperators using dense operators.
        sg.parameters.sparse_operator = False
        superoperators_dense = self._build_superoperators_from_basis(spin_system)

        # Clear the cache before rebuilding with sparse operators.
        sg.clear_cache()

        # Build all basis superoperators using sparse operators.
        sg.parameters.sparse_operator = True
        superoperators_sparse = self._build_superoperators_from_basis(spin_system)

        # Compare the dense and sparse results.
        for superoperator_dense, superoperator_sparse in zip(
            superoperators_dense,
            superoperators_sparse,
        ):
            self._assert_allclose(superoperator_dense, superoperator_sparse)

    def test_superoperator_2(self):
        """
        Test that the superoperator sparsity setting works as intended.
        """

        # Reset parameters to defaults.
        sg.parameters.default()

        # Create and build the example spin system.
        spin_system = sg.SpinSystem(["1H", "14N", "23Na"])
        spin_system.basis.max_spin_order = 2
        spin_system.basis.build()

        # Build dense superoperators.
        sg.parameters.sparse_superoperator = False
        superoperators_dense = self._build_superoperators_from_basis(spin_system)

        # Build sparse superoperators.
        sg.parameters.sparse_superoperator = True
        superoperators_sparse = self._build_superoperators_from_basis(spin_system)

        # Compare the dense and sparse representations.
        for superoperator_dense, superoperator_sparse in zip(
            superoperators_dense,
            superoperators_sparse,
        ):
            self._assert_allclose(superoperator_dense, superoperator_sparse)

    def test_superoperator_3(self):
        """
        Test that the superoperator created using the structure coefficients
        matches the explicit matrix-algebra construction.
        """

        # Reset the parameters to defaults.
        sg.parameters.default()

        # Create and build the test spin system.
        spin_system = sg.SpinSystem(["1H", "14N"])
        spin_system.basis.max_spin_order = 2
        spin_system.basis.build()

        # Test all product operators from the basis set.
        for operator_definition in spin_system.basis.basis:

            # Construct the reference left and right superoperators explicitly.
            superoperator_left_reference = self._build_manual_side_superoperator(
                spin_system,
                operator_definition,
                "left",
            )
            superoperator_right_reference = self._build_manual_side_superoperator(
                spin_system,
                operator_definition,
                "right",
            )

            # Build the same superoperators using structure coefficients.
            superoperator_left = sg.superoperator(
                spin_system,
                operator_definition,
                "left",
            )
            superoperator_right = sg.superoperator(
                spin_system,
                operator_definition,
                "right",
            )

            # Compare the reference and inbuilt constructions.
            self._assert_allclose(
                superoperator_left,
                superoperator_left_reference,
            )
            self._assert_allclose(
                superoperator_right,
                superoperator_right_reference,
            )

    def test_superoperator_4(self):
        """
        Test superoperator construction for truncated basis sets against the
        reference method.
        """

        # Reset parameters to defaults.
        sg.parameters.default()

        # Define the test spin systems.
        test_systems = [
            sg.SpinSystem(["1H"]),
            sg.SpinSystem(["1H", "14N"]),
        ]

        # Test both spin systems.
        for spin_system in test_systems:

            # Test every possible maximum spin order.
            for max_spin_order in range(1, spin_system.nspins + 1):

                # Build the corresponding basis set.
                spin_system.basis.max_spin_order = max_spin_order
                spin_system.basis.build()

                # Test every basis operator definition.
                for operator_definition in spin_system.basis.basis:

                    # Build the reference superoperators.
                    superoperator_left_reference = sop_prod_ref(
                        operator_definition,
                        spin_system.basis.basis,
                        spin_system.spins,
                        "left",
                    )
                    superoperator_right_reference = sop_prod_ref(
                        operator_definition,
                        spin_system.basis.basis,
                        spin_system.spins,
                        "right",
                    )
                    superoperator_comm_reference = sop_prod_ref(
                        operator_definition,
                        spin_system.basis.basis,
                        spin_system.spins,
                        "comm",
                    )

                    # Build the superoperators using the public API.
                    superoperator_left = sg.superoperator(
                        spin_system,
                        operator_definition,
                        "left",
                    )
                    superoperator_right = sg.superoperator(
                        spin_system,
                        operator_definition,
                        "right",
                    )
                    superoperator_comm = sg.superoperator(
                        spin_system,
                        operator_definition,
                        "comm",
                    )

                    # Compare the inbuilt and reference constructions.
                    self._assert_allclose(
                        superoperator_left,
                        superoperator_left_reference,
                    )
                    self._assert_allclose(
                        superoperator_right,
                        superoperator_right_reference,
                    )
                    self._assert_allclose(
                        superoperator_comm,
                        superoperator_comm_reference,
                    )

    def test_superoperator_5(self):
        """
        Test caching behaviour of the superoperator function when the basis
        changes.
        """

        # Reset parameters to defaults.
        sg.parameters.default()

        # Create the example spin system.
        spin_system = sg.SpinSystem(["1H", "1H", "1H"])

        # Define the operator whose superoperator is tested.
        operator_definition = np.array([2, 0, 0])

        # Construct the full basis and superoperator.
        spin_system.basis.max_spin_order = 3
        spin_system.basis.build()
        superoperator_full = sg.superoperator(
            spin_system,
            operator_definition,
            "comm",
        )

        # Construct the truncated basis and corresponding superoperator.
        spin_system.basis.max_spin_order = 2
        spin_system.basis.build()
        superoperator_truncated = sg.superoperator(
            spin_system,
            operator_definition,
            "comm",
        )

        # The resulting shapes should be different.
        self.assertNotEqual(
            superoperator_full.shape,
            superoperator_truncated.shape,
        )

    def test_superoperator_6(self):
        """
        Test creating the same superoperator from string and array inputs.
        """

        # Reset to default parameters.
        sg.parameters.default()

        # Create and build the example spin system.
        spin_system = sg.SpinSystem(["1H", "1H"])
        spin_system.basis.max_spin_order = 2
        spin_system.basis.build()

        # Compare string and array inputs for several superoperators.
        self._assert_allclose(
            sg.superoperator(spin_system, "I(z,0)", "left"),
            sg.superoperator(spin_system, [2, 0], "left"),
        )
        self._assert_allclose(
            sg.superoperator(spin_system, "I(z,0)", "right"),
            sg.superoperator(spin_system, [2, 0], "right"),
        )
        self._assert_allclose(
            sg.superoperator(spin_system, "I(z,0)", "comm"),
            sg.superoperator(spin_system, [2, 0], "comm"),
        )
        self._assert_allclose(
            sg.superoperator(spin_system, "I(z,0) + I(z,1)", "comm"),
            sg.superoperator(spin_system, [2, 0], "comm") +
            sg.superoperator(spin_system, [0, 2], "comm"),
        )
        self._assert_allclose(
            sg.superoperator(spin_system, "I(+,0) * I(-,1)", "comm"),
            -2 * sg.superoperator(spin_system, [1, 3], "comm"),
        )
        self._assert_allclose(
            sg.superoperator(spin_system, "I(x,0) + I(x,1)", "comm"),
            (
                -1 / np.sqrt(2) * sg.superoperator(spin_system, [1, 0], "comm") +
                1 / np.sqrt(2) * sg.superoperator(spin_system, [3, 0], "comm") -
                1 / np.sqrt(2) * sg.superoperator(spin_system, [0, 1], "comm") +
                1 / np.sqrt(2) * sg.superoperator(spin_system, [0, 3], "comm")
            ),
        )

    def test_sop_T_coupled(self):
        """
        Test Cartesian and coupled-spherical constructions of a two-spin term.
        """

        # Set parameters.
        sg.parameters.default()
        sg.parameters.sparse_superoperator = False

        # Create and build the example spin system.
        spin_system = sg.SpinSystem(["1H", "1H"])
        spin_system.basis.max_spin_order = 2
        spin_system.basis.build()

        # Generate a reproducible Cartesian interaction tensor.
        interaction_tensor = np.random.default_rng(0).random((3, 3))

        # Define the Cartesian operator labels.
        cartesian_labels = np.array(["x", "y", "z"])

        # Perform the Cartesian contraction explicitly.
        left_hand_side = np.zeros(
            (spin_system.basis.dim, spin_system.basis.dim),
            dtype=complex,
        )
        for row_index in range(interaction_tensor.shape[0]):
            for column_index in range(interaction_tensor.shape[1]):
                left_hand_side += (
                    interaction_tensor[row_index, column_index] *
                    sg.superoperator(
                        spin_system,
                        f"I({cartesian_labels[row_index]},0)"
                        f"*I({cartesian_labels[column_index]},1)",
                    )
                )

        # Convert the interaction tensor to spherical components.
        spherical_tensor = cartesian_tensor_to_spherical_tensor(
            interaction_tensor
        )

        # Perform the same contraction using coupled spherical tensors.
        right_hand_side = np.zeros(
            (spin_system.basis.dim, spin_system.basis.dim),
            dtype=complex,
        )
        for rank in range(0, 3):
            for projection in range(-rank, rank + 1):
                right_hand_side += (
                    (-1) ** projection *
                    spherical_tensor[(rank, projection)] *
                    sg.sop_T_coupled(
                        spin_system,
                        rank,
                        -projection,
                        0,
                        1,
                    )
                )

        # Both constructions should give the same result.
        self._assert_allclose(left_hand_side, right_hand_side)

    def test_sop_to_truncated_basis(self):
        """
        Test the transformation of superoperators to a truncated basis.
        """

        # Reset to defaults.
        sg.parameters.default()

        # Create the example spin system.
        spin_system = sg.SpinSystem(["1H", "1H", "1H", "1H", "14N"])

        # Build the original basis set.
        spin_system.basis.max_spin_order = 3
        spin_system.basis.build()

        # Define the operator labels to test.
        operator_labels = ["E", "x", "y", "z", "+", "-"]

        # Build the superoperators in the original basis.
        operator_strings = self._build_operator_strings(operator_labels)
        superoperators_original = [
            sg.superoperator(spin_system, operator_string)
            for operator_string in operator_strings
        ]

        # Truncate the basis set and the original superoperators.
        superoperators_truncated = spin_system.basis.truncate_by_coherence(
            [-2, 0, 2],
            *superoperators_original,
        )

        # Rebuild the same superoperators directly in the truncated basis.
        superoperators_truncated_reference = [
            sg.superoperator(spin_system, operator_string)
            for operator_string in operator_strings
        ]

        # Compare the transformed and directly rebuilt superoperators.
        for superoperator_truncated, superoperator_reference in zip(
            superoperators_truncated,
            superoperators_truncated_reference,
        ):
            self._assert_allclose(
                superoperator_truncated,
                superoperator_reference,
            )


def sop_prod_ref(
    op_def: np.ndarray,
    basis: np.ndarray,
    spins: np.ndarray,
    side: Literal["comm", "left", "right"],
) -> np.ndarray:
    """
    Calculate a reference superoperator using structure coefficients directly.

    NOTE:
    This implementation is very slow and should be used for testing purposes
    only.

    Parameters
    ----------
    op_def : ndarray
        Specifies the product operator to be generated. For example,
        input `(0, 2, 0, 1)` will generate `E*T_10*E*T_11`. The indices are
        given by `N = l^2 + l - q`, where `l` is the rank and `q` is the
        projection.
    basis : ndarray
        A two-dimensional array where each row contains integers that represent
        a Kronecker product of single-spin irreducible spherical tensors.
    spins : ndarray
        A sequence of floats describing the spin quantum numbers of the spin
        system.
    side : {'comm', 'left', 'right'}
        Specifies the type of superoperator.

    Returns
    -------
    sop : ndarray
        Superoperator defined by `op_def`.
    """

    # Build commutation superoperators as left minus right multiplication.
    if side == "comm":
        return sop_prod_ref(op_def, basis, spins, "left") - sop_prod_ref(
            op_def,
            basis,
            spins,
            "right",
        )

    # Obtain the basis dimension and number of spins.
    dim = basis.shape[0]
    nspins = spins.shape[0]

    # Initialise the superoperator.
    sop = np.zeros((dim, dim), dtype=complex)

    # Loop over each matrix row.
    for row_index in range(dim):

        # Loop over each matrix column.
        for column_index in range(dim):

            # Initialise the matrix element.
            element = 1

            # Loop over the spins.
            for spin_index in range(nspins):

                # Get the single-spin operator indices.
                i_ind = op_def[spin_index]
                j_ind = basis[row_index, spin_index]
                k_ind = basis[column_index, spin_index]

                # Get the structure coefficients for the current spin.
                coefficients = structure_coefficients(spins[spin_index], side)

                # Add the current single-spin contribution to the product.
                element = element * coefficients[i_ind, j_ind, k_ind]

            # Store the completed matrix element.
            sop[row_index, column_index] = element

    return sop