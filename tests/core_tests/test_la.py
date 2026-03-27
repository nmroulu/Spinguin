"""
Tests for linear-algebra utilities used by the Spinguin core.
"""

import math
import unittest

import numpy as np
import spinguin as sg
from scipy.sparse import csc_array, random_array
from scipy.sparse.linalg import expm

from spinguin._core import _la
from spinguin._core._hide_prints import HidePrints


class TestLinearAlgebra(unittest.TestCase):
    """
    Test the internal linear-algebra helper functions.
    """

    def _to_dense(
        self,
        array,
    ):
        """
        Return a dense representation of an array when needed.

        Parameters
        ----------
        array : array-like or sparse matrix
            Array to convert.

        Returns
        -------
        ndarray
            Dense representation of the input array.
        """

        # Convert sparse matrices to dense arrays when necessary.
        if hasattr(array, "toarray"):
            return array.toarray()

        return array

    def _assert_allclose(
        self,
        actual,
        reference,
    ):
        """
        Check that two arrays agree within numerical tolerance.

        Parameters
        ----------
        actual : array-like or sparse matrix
            Tested array.
        reference : array-like or sparse matrix
            Reference array.

        Returns
        -------
        None
            The assertion is evaluated in place.
        """

        # Compare the tested and reference arrays in dense form.
        self.assertTrue(
            np.allclose(self._to_dense(actual), self._to_dense(reference))
        )

    def test_isvector(self):
        """
        Test recognising row and column vectors with valid and invalid shapes.
        """

        # Create row vectors, column vectors, and non-vector arrays.
        row1 = np.array([[1, 0, 0]])
        row2 = csc_array([[0, 1, 0, 7]])
        col1 = row1.T
        col2 = row2.T
        arr1 = np.array(
            [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ]
        )
        arr2 = np.array([[[1], [2], [3]]])
        arr3 = np.array([1, 0, 0])

        # Check vectors with the correct orientation.
        self.assertTrue(_la.isvector(row1, "row"))
        self.assertTrue(_la.isvector(row2, "row"))
        self.assertTrue(_la.isvector(col1, "col"))
        self.assertTrue(_la.isvector(col2, "col"))

        # Check vectors with the incorrect orientation.
        self.assertFalse(_la.isvector(row1, "col"))
        self.assertFalse(_la.isvector(row2, "col"))
        self.assertFalse(_la.isvector(col1, "row"))
        self.assertFalse(_la.isvector(col2, "row"))

        # Check two-dimensional arrays that are not vectors.
        self.assertFalse(_la.isvector(arr1, "col"))
        self.assertFalse(_la.isvector(arr1, "row"))

        # Check arrays with unsupported shapes.
        with self.assertRaises(ValueError):
            _la.isvector(arr2, "col")
        with self.assertRaises(ValueError):
            _la.isvector(arr2, "row")
        with self.assertRaises(ValueError):
            _la.isvector(arr3, "col")
        with self.assertRaises(ValueError):
            _la.isvector(arr3, "row")

    def test_norm_1(self):
        """
        Test the row-wise and column-wise 1-norm implementations.
        """

        # Create a dense test matrix.
        matrix = np.random.rand(3, 3)

        # Compare dense results with NumPy reference values.
        self.assertAlmostEqual(
            _la.norm_1(matrix, "row"),
            np.linalg.norm(matrix, ord=np.inf),
        )
        self.assertAlmostEqual(
            _la.norm_1(matrix, "col"),
            np.linalg.norm(matrix, ord=1),
        )

        # Compare sparse results with the same NumPy reference values.
        self.assertAlmostEqual(
            _la.norm_1(csc_array(matrix), "row"),
            np.linalg.norm(matrix, ord=np.inf),
        )
        self.assertAlmostEqual(
            _la.norm_1(csc_array(matrix), "col"),
            np.linalg.norm(matrix, ord=1),
        )

    def test_expm(self):
        """
        Test matrix exponentiation for dense and sparse matrices.
        """

        # Create a dense test matrix with moderately large entries.
        matrix = np.array(
            [
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9],
            ]
        )

        # Compare the dense result with SciPy.
        self._assert_allclose(_la.expm(matrix, 1e-32), expm(matrix))

        # Compare the sparse result with SciPy.
        matrix_sparse = csc_array(matrix)
        self._assert_allclose(_la.expm(matrix_sparse, 1e-32), expm(matrix_sparse))

    def test_eliminate_small(self):
        """
        Test removal of matrix elements below a threshold.
        """

        # Create dense and sparse versions of the same test matrix.
        matrix_sparse = csc_array(
            [
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9],
            ]
        )
        matrix_dense = matrix_sparse.toarray()

        # Remove elements smaller than the requested threshold.
        _la.eliminate_small(matrix_sparse, zero_value=5)
        _la.eliminate_small(matrix_dense, zero_value=5)

        # Define the expected truncated matrix.
        matrix_reference = np.array(
            [
                [0, 0, 0],
                [0, 5, 6],
                [7, 8, 9],
            ]
        )

        # Compare the truncated matrices with the reference result.
        self.assertTrue(np.array_equal(matrix_sparse.toarray(), matrix_reference))
        self.assertTrue(np.array_equal(matrix_dense, matrix_reference))
        self.assertEqual(matrix_sparse.nnz, 5)

    def test_sparse_bytes(self):
        """
        Test serialising and deserialising sparse matrices.
        """

        # Create a large sparse test matrix.
        matrix = random_array((1000, 1000), density=0.5, format="csc")

        # Convert the sparse matrix to bytes and back.
        matrix_bytes = _la.sparse_to_bytes(matrix)
        matrix_recovered = _la.bytes_to_sparse(matrix_bytes)

        # Compare the original and recovered matrices.
        self._assert_allclose(matrix, matrix_recovered)

    def test_comm(self):
        """
        Test simple commutator identities.
        """

        # Create a dense test matrix.
        matrix = np.random.rand(3, 3)

        # Check that a matrix commutes with itself.
        self._assert_allclose(_la.comm(matrix, matrix), np.zeros_like(matrix))

        # Check that every matrix commutes with the identity.
        self._assert_allclose(_la.comm(matrix, np.eye(3)), np.zeros_like(matrix))

    def test_find_common_rows(self):
        """
        Test locating identical rows in two arrays.
        """

        # Create two arrays with two rows in common.
        array_a = np.array(
            [
                [0, 0, 0],
                [0, 0, 1],
                [0, 1, 0],
                [1, 0, 0],
            ]
        )
        array_b = np.array(
            [
                [0, 0, 0],
                [4, 5, 6],
                [1, 0, 0],
            ]
        )

        # Find the indices of the matching rows.
        indices_a, indices_b = _la.find_common_rows(array_a, array_b)

        # Check that the indexed rows are identical.
        self.assertTrue(np.array_equal(array_a[indices_a], array_b[indices_b]))

    def test_auxiliary_matrix_expm(self):
        """
        Test the auxiliary-matrix exponential for dense and sparse inputs.
        """

        # Define the block matrices in sparse format.
        matrix_a_sparse = csc_array(
            [
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9],
            ]
        ) * 1e-3
        matrix_b_sparse = csc_array(
            [
                [4, 5, 6],
                [7, 8, 9],
                [1, 2, 3],
            ]
        ) * 1e-3
        matrix_c_sparse = csc_array(
            [
                [7, 8, 9],
                [1, 2, 3],
                [4, 5, 6],
            ]
        ) * 1e-3

        # Create dense versions of the same matrices.
        matrix_a_dense = matrix_a_sparse.toarray()
        matrix_b_dense = matrix_b_sparse.toarray()
        matrix_c_dense = matrix_c_sparse.toarray()

        # Define the integration time.
        time_interval = 1

        # Compute the auxiliary-matrix exponential in both formalisms.
        expm_aux_sparse = _la.auxiliary_matrix_expm(
            matrix_a_sparse,
            matrix_b_sparse,
            matrix_c_sparse,
            time_interval,
            zero_value=1e-18,
        )
        expm_aux_dense = _la.auxiliary_matrix_expm(
            matrix_a_dense,
            matrix_b_dense,
            matrix_c_dense,
            time_interval,
            zero_value=1e-18,
        )

        # Extract the sparse block matrices.
        top_left_sparse = expm_aux_sparse[
            : matrix_a_sparse.shape[0],
            : matrix_a_sparse.shape[1],
        ]
        top_right_sparse = expm_aux_sparse[
            : matrix_a_sparse.shape[0],
            matrix_a_sparse.shape[1] :,
        ]
        bottom_left_sparse = expm_aux_sparse[
            matrix_a_sparse.shape[0] :,
            : matrix_a_sparse.shape[1],
        ]
        bottom_right_sparse = expm_aux_sparse[
            matrix_a_sparse.shape[0] :,
            matrix_a_sparse.shape[1] :,
        ]

        # Extract the dense block matrices.
        top_left_dense = expm_aux_dense[
            : matrix_a_dense.shape[0],
            : matrix_a_dense.shape[1],
        ]
        top_right_dense = expm_aux_dense[
            : matrix_a_dense.shape[0],
            matrix_a_dense.shape[1] :,
        ]
        bottom_left_dense = expm_aux_dense[
            matrix_a_dense.shape[0] :,
            : matrix_a_dense.shape[1],
        ]
        bottom_right_dense = expm_aux_dense[
            matrix_a_dense.shape[0] :,
            matrix_a_dense.shape[1] :,
        ]

        # Compute the reference block matrices directly.
        with HidePrints():
            top_left_reference = _la.expm(
                matrix_a_dense * time_interval,
                zero_value=1e-18,
            )
            top_right_reference = np.zeros(matrix_a_dense.shape, dtype=complex)
            for time_point in np.linspace(0, time_interval, 1000):
                top_right_reference += (
                    _la.expm(-matrix_a_dense * time_point, zero_value=1e-18)
                    @ matrix_b_dense
                    @ _la.expm(matrix_c_dense * time_point, zero_value=1e-18)
                    * (1 / 1000)
                )
            top_right_reference = (
                _la.expm(matrix_a_dense * time_interval, zero_value=1e-18)
                @ top_right_reference
            )
            bottom_left_reference = np.zeros_like(bottom_left_dense)
            bottom_right_reference = _la.expm(
                matrix_c_dense * time_interval,
                zero_value=1e-18,
            )

        # Verify the sparse block matrices.
        self._assert_allclose(top_left_sparse, top_left_reference)
        self._assert_allclose(top_right_sparse, top_right_reference)
        self._assert_allclose(bottom_left_sparse, bottom_left_reference)
        self._assert_allclose(bottom_right_sparse, bottom_right_reference)

        # Verify the dense block matrices.
        self._assert_allclose(top_left_dense, top_left_reference)
        self._assert_allclose(top_right_dense, top_right_reference)
        self._assert_allclose(bottom_left_dense, bottom_left_reference)
        self._assert_allclose(bottom_right_dense, bottom_right_reference)

    def test_angle_between_vectors(self):
        """
        Test angles between simple reference vectors.
        """

        # Define test vectors with known mutual angles.
        v1 = np.array([1, 0])
        v2 = np.array([0, 1])
        v3 = np.array([1, 1])

        # Compare the computed angles with analytical values.
        self.assertAlmostEqual(_la.angle_between_vectors(v1, v1), 0)
        self.assertAlmostEqual(_la.angle_between_vectors(v1, v2), np.pi / 2)
        self.assertAlmostEqual(_la.angle_between_vectors(v1, -v1), np.pi)
        self.assertAlmostEqual(_la.angle_between_vectors(v1, v3), np.pi / 4)

    def test_decompose_matrix(self):
        """
        Test isotropic, antisymmetric, and symmetric matrix decomposition.
        """

        # Create a dense test matrix.
        matrix = np.random.rand(3, 3)

        # Decompose the matrix into its standard parts.
        isotropic, antisymmetric, symmetric = _la.decompose_matrix(matrix)

        # Check the defining properties of the decomposition.
        self.assertTrue(np.allclose(matrix, isotropic + antisymmetric + symmetric))
        self.assertTrue(np.allclose(antisymmetric, -antisymmetric.T))
        self.assertTrue(np.allclose(symmetric, symmetric.T))

    def test_principal_axis_system(self):
        """
        Test diagonalisation in the principal-axis system.
        """

        # Create a dense test matrix and its symmetric component.
        matrix = np.random.rand(3, 3)
        _, _, matrix_symmetric = _la.decompose_matrix(matrix)

        # Determine the principal-axis system representation.
        eigenvalues, eigenvectors, tensor_pas = _la.principal_axis_system(matrix)

        # Check that the eigenvectors diagonalise the symmetric part.
        self.assertTrue(np.allclose(
            matrix_symmetric,
            np.linalg.inv(eigenvectors) @ np.diag(eigenvalues) @ eigenvectors,
        ))

        # Check that the original tensor can be reconstructed.
        self.assertTrue(np.allclose(
            matrix,
            np.linalg.inv(eigenvectors) @ tensor_pas @ eigenvectors,
        ))

    def test_cartesian_tensor_to_spherical_tensor(self):
        """
        Test projection from Cartesian to spherical tensor components.
        """

        # Create a dense Cartesian tensor.
        tensor = np.random.rand(3, 3)

        # Project the spherical tensor components and compare them.
        for l in range(0, 3):
            for q in range(-l, l + 1):
                self.assertAlmostEqual(
                    (tensor @ spherical_tensor(l, q)).trace(),
                    _la.cartesian_tensor_to_spherical_tensor(tensor)[(l, q)],
                )

    def test_vector_to_spherical_tensor(self):
        """
        Test projection from Cartesian vectors to spherical tensors.
        """

        # Create a dense Cartesian vector.
        vector = np.random.rand(3)

        # Project the spherical tensor components and compare them.
        for q in range(-1, 2):
            self.assertAlmostEqual(
                np.inner(spherical_vector(1, q), vector),
                _la.vector_to_spherical_tensor(vector)[(1, q)],
            )

    def test_cartesian_to_spherical_tensor_conventions(self):
        """
        Test the Cartesian and spherical tensor conventions for bilinear terms.
        """

        # Set the global parameters.
        sg.parameters.default()
        sg.parameters.sparse_operator = False

        # Create a random Cartesian interaction tensor.
        tensor = np.random.rand(3, 3)

        # Create the single-spin unit operator.
        unit_operator = sg.op_E(1 / 2)

        # Build the spin operators for the first spin.
        ix = np.kron(sg.op_Sx(1 / 2), unit_operator)
        iy = np.kron(sg.op_Sy(1 / 2), unit_operator)
        iz = np.kron(sg.op_Sz(1 / 2), unit_operator)

        # Build the spin operators for the second spin.
        sx = np.kron(unit_operator, sg.op_Sx(1 / 2))
        sy = np.kron(unit_operator, sg.op_Sy(1 / 2))
        sz = np.kron(unit_operator, sg.op_Sz(1 / 2))

        # Construct the Cartesian spin vectors.
        spin_i = np.array([[ix, iy, iz]], dtype=complex)
        spin_s = np.array([[sx], [sy], [sz]], dtype=complex)

        # Evaluate the Cartesian contraction explicitly.
        cartesian_result = np.zeros_like(ix)
        for index_i in range(tensor.shape[0]):
            for index_s in range(tensor.shape[1]):
                cartesian_result += (
                    tensor[index_i, index_s]
                    * spin_i[0, index_i]
                    @ spin_s[index_s, 0]
                )

        # Convert the tensor to spherical components.
        spherical_components = _la.cartesian_tensor_to_spherical_tensor(tensor)

        # Evaluate the same contraction in the spherical convention.
        spherical_result = np.zeros_like(ix)
        for l in range(0, 3):
            for q in range(-l, l + 1):
                spherical_result += (
                    (-1) ** q
                    * spherical_components[(l, q)]
                    * sg.op_T_coupled(l, -q, 1, 1 / 2, 1, 1 / 2)
                )

        # Check that both conventions give the same result.
        self.assertTrue(np.allclose(cartesian_result, spherical_result))

    def test_CG_coeff(self):
        """
        Test Clebsch-Gordan coefficients against known values.
        """

        # Compare against tabulated coefficient values.
        self.assertAlmostEqual(_la.CG_coeff(1 / 2, 1 / 2, 1 / 2, 1 / 2, 1, 1), 1)
        self.assertAlmostEqual(
            _la.CG_coeff(1 / 2, -1 / 2, 1 / 2, -1 / 2, 1, -1),
            1,
        )
        self.assertAlmostEqual(
            _la.CG_coeff(1 / 2, 1 / 2, 1 / 2, -1 / 2, 1, 0),
            math.sqrt(1 / 2),
        )
        self.assertAlmostEqual(
            _la.CG_coeff(1 / 2, 1 / 2, 1 / 2, -1 / 2, 0, 0),
            math.sqrt(1 / 2),
        )
        self.assertAlmostEqual(
            _la.CG_coeff(1 / 2, -1 / 2, 1 / 2, 1 / 2, 1, 0),
            math.sqrt(1 / 2),
        )
        self.assertAlmostEqual(
            _la.CG_coeff(1 / 2, -1 / 2, 1 / 2, 1 / 2, 0, 0),
            -math.sqrt(1 / 2),
        )
        self.assertAlmostEqual(_la.CG_coeff(1, 1, 1 / 2, 1 / 2, 3 / 2, 3 / 2), 1)
        self.assertAlmostEqual(
            _la.CG_coeff(1, 1, 1 / 2, -1 / 2, 3 / 2, 1 / 2),
            math.sqrt(1 / 3),
        )
        self.assertAlmostEqual(
            _la.CG_coeff(1, 1, 1 / 2, -1 / 2, 1 / 2, 1 / 2),
            math.sqrt(2 / 3),
        )
        self.assertAlmostEqual(
            _la.CG_coeff(1, 0, 1 / 2, 1 / 2, 3 / 2, 1 / 2),
            math.sqrt(2 / 3),
        )
        self.assertAlmostEqual(
            _la.CG_coeff(1, 0, 1 / 2, 1 / 2, 1 / 2, 1 / 2),
            -math.sqrt(1 / 3),
        )

    def test_custom_dot(self):
        """
        Test the custom sparse matrix product against SciPy.
        """

        # Create two sparse matrices with compatible dimensions.
        matrix_a = random_array((200, 300), density=0.2, format="csc", dtype=complex)
        matrix_b = random_array((300, 400), density=0.2, format="csc", dtype=complex)

        # Compare the general sparse product with SciPy.
        matrix_scipy = matrix_a @ matrix_b
        matrix_custom = _la.custom_dot(matrix_a, matrix_b, zero_value=1e-18)
        self._assert_allclose(matrix_scipy, matrix_custom)

        # Compare an empty-left product with SciPy.
        matrix_a_empty = csc_array((200, 300))
        matrix_scipy = matrix_a_empty @ matrix_b
        matrix_custom = _la.custom_dot(matrix_a_empty, matrix_b, zero_value=1e-18)
        self._assert_allclose(matrix_scipy, matrix_custom)

        # Compare an empty-right product with SciPy.
        matrix_b_empty = csc_array((300, 400))
        matrix_scipy = matrix_a @ matrix_b_empty
        matrix_custom = _la.custom_dot(matrix_a, matrix_b_empty, zero_value=1e-18)
        self._assert_allclose(matrix_scipy, matrix_custom)

        # Compare an empty-by-empty product with SciPy.
        matrix_scipy = matrix_a_empty @ matrix_b_empty
        matrix_custom = _la.custom_dot(
            matrix_a_empty,
            matrix_b_empty,
            zero_value=1e-18,
        )
        self._assert_allclose(matrix_scipy, matrix_custom)

        # Compare a product that yields an empty result.
        matrix_a = csc_array(
            [
                [0, 1],
                [0, 0],
            ]
        )
        matrix_scipy = matrix_a @ matrix_a
        matrix_custom = _la.custom_dot(matrix_a, matrix_a, zero_value=1e-18)
        self._assert_allclose(matrix_scipy, matrix_custom)

        # Reuse the same matrix repeatedly to exercise cached code paths.
        matrix_a = random_array((200, 200), density=0.2, format="csc", dtype=complex)
        for _ in range(10):
            matrix_a = _la.custom_dot(matrix_a, matrix_a, zero_value=1e-32)

        # Compare several index and data type combinations with SciPy.
        dtypes_indices = [np.int32, np.int64]
        dtypes_values = [np.int32, np.int64, np.float64, np.complex128]
        for dtype_a_indices in dtypes_indices:
            for dtype_b_indices in dtypes_indices:
                for dtype_a_values in dtypes_values:
                    for dtype_b_values in dtypes_values:
                        with self.subTest(
                            dtype_a_indices=dtype_a_indices,
                            dtype_b_indices=dtype_b_indices,
                            dtype_a_values=dtype_a_values,
                            dtype_b_values=dtype_b_values,
                        ):
                            matrix_a = random_array((200, 200), density=0.2).tocsc()
                            matrix_b = random_array((200, 200), density=0.2).tocsc()
                            matrix_a.data = matrix_a.data.astype(dtype_a_values)
                            matrix_a.indices = matrix_a.indices.astype(
                                dtype_a_indices
                            )
                            matrix_a.indptr = matrix_a.indptr.astype(
                                dtype_a_indices
                            )
                            matrix_b.data = matrix_b.data.astype(dtype_b_values)
                            matrix_b.indices = matrix_b.indices.astype(
                                dtype_b_indices
                            )
                            matrix_b.indptr = matrix_b.indptr.astype(
                                dtype_b_indices
                            )
                            matrix_custom = _la.custom_dot(
                                matrix_a,
                                matrix_b,
                                zero_value=1e-18,
                            )
                            matrix_scipy = matrix_a @ matrix_b
                            self._assert_allclose(matrix_scipy, matrix_custom)

    def test_expm_vec_taylor(self):
        """
        Test the Taylor-based action of the matrix exponential on a vector.
        """

        # Create a dense test matrix.
        matrix = np.array(
            [
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9],
            ]
        )

        # Create a dense column vector.
        vector = np.array([[1], [2], [3]])

        # Compute the exponential action for all dense and sparse combinations.
        expm_vector_dd = _la.expm_vec_taylor(matrix, vector, 1e-18)
        expm_vector_ds = _la.expm_vec_taylor(matrix, csc_array(vector), 1e-18)
        expm_vector_sd = _la.expm_vec_taylor(csc_array(matrix), vector, 1e-18)
        expm_vector_ss = _la.expm_vec_taylor(
            csc_array(matrix),
            csc_array(vector),
            1e-18,
        )

        # Compute the SciPy reference result.
        expm_vector_reference = expm(matrix) @ vector

        # Compare all implementations with the reference result.
        self._assert_allclose(expm_vector_dd, expm_vector_reference)
        self._assert_allclose(expm_vector_ds, expm_vector_reference)
        self._assert_allclose(expm_vector_sd, expm_vector_reference)
        self._assert_allclose(expm_vector_ss, expm_vector_reference)

    def test_expm_vec(self):
        """
        Test the action of the matrix exponential on a vector.
        """

        # Create a dense test matrix.
        matrix = np.array(
            [
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9],
            ]
        )

        # Create a dense column vector.
        vector = np.array([[1], [2], [3]])

        # Compute the exponential action for all dense and sparse combinations.
        expm_vector_dd = _la.expm_vec(matrix, vector, 1e-18)
        expm_vector_ds = _la.expm_vec(matrix, csc_array(vector), 1e-18)
        expm_vector_sd = _la.expm_vec(csc_array(matrix), vector, 1e-18)
        expm_vector_ss = _la.expm_vec(csc_array(matrix), csc_array(vector), 1e-18)

        # Compute the SciPy reference result.
        expm_vector_reference = expm(matrix) @ vector

        # Compare all implementations with the reference result.
        self._assert_allclose(expm_vector_dd, expm_vector_reference)
        self._assert_allclose(expm_vector_ds, expm_vector_reference)
        self._assert_allclose(expm_vector_sd, expm_vector_reference)
        self._assert_allclose(expm_vector_ss, expm_vector_reference)

def spherical_tensor(l, q):
    """
    Construct a spherical tensor in the Cartesian basis for testing.

    The tensor is obtained by combining covariant spherical basis vectors.

    Recipe described in Man: Cartesian and Spherical Tensors in NMR Hamiltonians
    https://doi.org/10.1002/cmr.a.21289
    """

    # Initialise the Cartesian tensor.
    tensor_lq = np.zeros((3, 3), dtype=complex)

    # Couple the spherical basis vectors to rank `l` and projection `q`.
    for q1 in range(-1, 2):
        for q2 in range(-1, 2):
            tensor_lq += _la.CG_coeff(1, q1, 1, q2, l, q) * np.outer(
                spherical_vector(1, q1),
                spherical_vector(1, q2),
            )

    return tensor_lq

def spherical_vector(l, q):
    """
    Construct a covariant spherical vector in the Cartesian basis.
    """

    # Define the Cartesian basis vectors.
    e_x = np.array([1, 0, 0])
    e_y = np.array([0, 1, 0])
    e_z = np.array([0, 0, 1])

    # Construct the requested spherical basis vector.
    if l == 1 and q == 1:
        vector = -1 / np.sqrt(2) * (e_x + 1j * e_y)
    elif l == 1 and q == 0:
        vector = e_z
    elif l == 1 and q == -1:
        vector = 1 / np.sqrt(2) * (e_x - 1j * e_y)

    return vector