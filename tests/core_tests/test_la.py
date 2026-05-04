"""
Tests for linear-algebra utilities used by the Spinguin core.
"""

import math
import unittest
from copy import deepcopy

import numpy as np
import spinguin as sg
from scipy.sparse import csc_array, random_array
from scipy.sparse.linalg import expm

from spinguin._core import _la
from ._helpers import spherical_tensor, spherical_vector

class TestLinearAlgebra(unittest.TestCase):
    """
    Test the internal linear-algebra helper functions.
    """

    def test_isvector(self):
        """
        Test recognising row and column vectors with valid and invalid shapes.
        """

        # Create row vectors, column vectors, and non-vector arrays.
        row1 = np.array([[1, 0, 0]])
        row2 = csc_array([[0, 1, 0, 7]])
        col1 = row1.T
        col2 = row2.T
        arr1 = np.eye(3)
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
        self.assertRaises(ValueError, _la.isvector, arr2, 'col')
        self.assertRaises(ValueError, _la.isvector, arr2, 'row')
        self.assertRaises(ValueError, _la.isvector, arr3, 'col')
        self.assertRaises(ValueError, _la.isvector, arr3, 'row')

    def test_norm_1(self):
        """
        Test the row-wise and column-wise 1-norm implementations.
        """

        # Create a dense test matrix.
        A = np.array([
            [1.2, -0.6,  0.9],
            [0.1,  1.1, -1.7],
            [1.8, -0.6, -0.7]
        ])

        # Compare dense results with NumPy reference values.
        self.assertAlmostEqual(
            _la.norm_1(A, 'row'),
            np.linalg.norm(A, ord=np.inf)
        )
        self.assertAlmostEqual(
            _la.norm_1(A, 'col'),
            np.linalg.norm(A, ord=1)
        )

        # Compare sparse results with the same NumPy reference values.
        self.assertAlmostEqual(
            _la.norm_1(csc_array(A), 'row'),
            np.linalg.norm(A, ord=np.inf)
        )
        self.assertAlmostEqual(
            _la.norm_1(csc_array(A), 'col'),
            np.linalg.norm(A, ord=1)
        )

    def test_expm(self):
        """
        Test matrix exponentiation for dense and sparse matrices.
        """

        # Create a dense test matrix with moderately large entries.
        A = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ])

        # Compare the dense result with SciPy.
        self.assertTrue(np.allclose(_la.expm(A, 1e-32), expm(A)))

        # Compare the sparse result with SciPy.
        A = csc_array(A)
        self.assertTrue(
            np.allclose(_la.expm(A, 1e-32).toarray(), expm(A).toarray())
        )

    def test_eliminate_small(self):
        """
        Test removal of matrix elements below a threshold.
        """

        # Create sparse and dense versions of the same test matrix.
        A_sp = csc_array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ])
        A_np = A_sp.toarray()
        
        # Remove elements smaller than the requested threshold.
        _la.eliminate_small(A_sp, zero_value=5)
        _la.eliminate_small(A_np, zero_value=5)

        # Define the expected truncated matrix.
        B = np.array([
            [0, 0, 0],
            [0, 5, 6],
            [7, 8, 9]
        ])

        # Compare the truncated matrices with the reference result.
        self.assertTrue(np.array_equal(A_sp.toarray(), B))
        self.assertTrue(np.array_equal(A_np, B))
        self.assertEqual(A_sp.nnz, 5)

    def test_sparse_bytes(self):
        """
        Test serialising and deserialising sparse matrices.
        """

        # Create a sparse test matrix.
        A = csc_array([
            [0, 9, 0],
            [1, 0, 2],
            [0, 8, 1]
        ])

        # Convert the sparse matrix to bytes and back.
        A_bytes = _la.sparse_to_bytes(A)
        B = _la.bytes_to_sparse(A_bytes)

        # Compare the original and recovered matrices.
        self.assertTrue(np.allclose(A.toarray(), B.toarray()))

    def test_comm(self):
        """
        Test simple commutator identities.
        """

        # Create a dense test matrix.
        A = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ])

        # Check that a matrix commutes with itself.
        self.assertTrue(np.allclose(_la.comm(A, A), np.zeros_like(A)))

        # Check that every matrix commutes with the identity.
        self.assertTrue(np.allclose(_la.comm(A, np.eye(3)), np.zeros_like(A)))

    def test_find_common_rows(self):
        """
        Test locating identical rows in two arrays.
        """

        # Create two arrays with two rows in common.
        A = np.array([
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [1, 0, 0]
        ])
        B = np.array([
            [0, 0, 0],
            [4, 5, 6],
            [1, 0, 0]
        ])
        
        # Find the indices of the matching rows.
        A_ind, B_ind = _la.find_common_rows(A, B)

        # Check that the indexed rows are identical.
        self.assertTrue(np.array_equal(A[A_ind], B[B_ind]))

    def test_auxiliary_matrix_expm(self):
        """
        Test the auxiliary-matrix exponential for dense and sparse inputs.
        """

        # Define the block matrices in sparse format.
        A_sp = csc_array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ]) * 1e-3
        B_sp = csc_array([
            [4, 5, 6],
            [7, 8, 9],
            [1, 2, 3]
        ]) * 1e-3
        C_sp = csc_array([
            [7, 8, 9],
            [1, 2, 3],
            [4, 5, 6]
        ]) * 1e-3
        
        # Create dense versions of the same matrices.
        A_dn = A_sp.toarray()
        B_dn = B_sp.toarray()
        C_dn = C_sp.toarray()

        # Define the integration time.
        T = 1
        
        # Compute the auxiliary-matrix exponential in both formalisms.
        expm_aux_sp = \
            _la.auxiliary_matrix_expm(A_sp, B_sp, C_sp, T, zero_value=1e-18)
        expm_aux_dn = \
            _la.auxiliary_matrix_expm(A_dn, B_dn, C_dn, T, zero_value=1e-18)

        # Extract the sparse block matrices.
        top_l_sp = expm_aux_sp[:A_sp.shape[0], :A_sp.shape[1]]
        top_r_sp = expm_aux_sp[:A_sp.shape[0], A_sp.shape[1]:]
        bot_l_sp = expm_aux_sp[A_sp.shape[0]:, :A_sp.shape[1]]
        bot_r_sp = expm_aux_sp[A_sp.shape[0]:, A_sp.shape[1]:]

        # Extract the dense block matrices.
        top_l_dn = expm_aux_dn[:A_dn.shape[0], :A_dn.shape[1]]
        top_r_dn = expm_aux_dn[:A_dn.shape[0], A_dn.shape[1]:]
        bot_l_dn = expm_aux_dn[A_dn.shape[0]:, :A_dn.shape[1]]
        bot_r_dn = expm_aux_dn[A_dn.shape[0]:, A_dn.shape[1]:]

        # Compute the reference block matrices directly.
        top_l_ref = _la.expm(A_dn*T, zero_value=1e-18)
        top_r_ref = np.zeros(A_dn.shape, dtype=complex)
        for t in np.linspace(0, T, 1000):
            top_r_ref += (
                _la.expm(-A_dn*t, zero_value=1e-18) @ B_dn @
                _la.expm(C_dn*t, zero_value=1e-18) * (1/1000)
            )
        top_r_ref = _la.expm(A_dn*T, zero_value=1e-18) @ top_r_ref
        bot_l_ref = np.zeros_like(bot_l_dn)
        bot_r_ref = _la.expm(C_dn*T, zero_value=1e-18)

        # Verify the sparse block matrices.
        self.assertTrue(np.allclose(top_l_sp.toarray(), top_l_ref))
        self.assertTrue(np.allclose(top_r_sp.toarray(), top_r_ref))
        self.assertTrue(np.allclose(bot_l_sp.toarray(), bot_l_ref))
        self.assertTrue(np.allclose(bot_r_sp.toarray(), bot_r_ref))

        # Verify the dense block matrices.
        self.assertTrue(np.allclose(top_l_dn, top_l_ref))
        self.assertTrue(np.allclose(top_r_dn, top_r_ref))
        self.assertTrue(np.allclose(bot_l_dn, bot_l_ref))
        self.assertTrue(np.allclose(bot_r_dn, bot_r_ref))

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
        self.assertAlmostEqual(_la.angle_between_vectors(v1, v2), np.pi/2)
        self.assertAlmostEqual(_la.angle_between_vectors(v1, -v1), np.pi)
        self.assertAlmostEqual(_la.angle_between_vectors(v1, v3), np.pi/4)

    def test_decompose_matrix(self):
        """
        Test isotropic, antisymmetric, and symmetric matrix decomposition.
        """

        # Create a dense test matrix.
        A = np.random.rand(3, 3)

        # Decompose the matrix into its standard parts.
        iso, asym, sym = _la.decompose_matrix(A)

        # Check the defining properties of the decomposition.
        self.assertTrue(np.allclose(A, iso+asym+sym))
        self.assertTrue(np.allclose(asym, -asym.T))
        self.assertTrue(np.allclose(sym, sym.T))

    def test_principal_axis_system(self):
        """
        Test diagonalisation in the principal-axis system.
        """

        # Create a dense test matrix and its symmetric component.
        A = np.random.rand(3, 3)
        _, _, A_sym = _la.decompose_matrix(A)

        # Determine the principal-axis system representation.
        eigenvalues, eigenvectors, tensor_PAS = _la.principal_axis_system(A)

        # Check that the eigenvectors diagonalise the symmetric part.
        self.assertTrue(np.allclose(
            A_sym, 
            np.linalg.inv(eigenvectors) @ np.diag(eigenvalues) @ eigenvectors
        ))

        # Check that the original tensor can be reconstructed.
        self.assertTrue(np.allclose(
            A,
            np.linalg.inv(eigenvectors) @ tensor_PAS @ eigenvectors
        ))

    def test_cartesian_tensor_to_spherical_tensor(self):
        """
        Test projection from Cartesian to spherical tensor components.
        """

        # Create a dense Cartesian tensor.
        tensor = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ])

        # Project the spherical tensor components and compare them.
        for l in range(0, 3):
            for q in range(-l, l+1):
                self.assertAlmostEqual(
                    (tensor @ spherical_tensor(l, q)).trace(),
                    _la.cartesian_tensor_to_spherical_tensor(tensor)[(l, q)]
                )

    def test_vector_to_spherical_tensor(self):
        """
        Test projection from Cartesian vectors to spherical tensors.
        """

        # Create a random vector
        vector = np.array([1, 2, 3])

        # Project the spherical tensor components and compare them.
        for q in range(-1, 2):
            self.assertAlmostEqual(
                np.inner(spherical_vector(1, q), vector),
                _la.vector_to_spherical_tensor(vector)[(1, q)]
            )

    def test_cartesian_to_spherical_tensor_conventions(self):
        """
        Test the Cartesian and spherical tensor conventions for bilinear terms.
        """

        # Set the global parameters.
        sg.parameters.default()
        sg.parameters.sparse_operator = False

        # Create a Cartesian interaction tensor.
        A = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ])

        # Create the single-spin unit operator.
        E = sg.op_E(1/2)

        # Build the spin operators for the first spin.
        Ix = np.kron(sg.op_Sx(1/2), E)
        Iy = np.kron(sg.op_Sy(1/2), E)
        Iz = np.kron(sg.op_Sz(1/2), E)

        # Build the spin operators for the second spin.
        Sx = np.kron(E, sg.op_Sx(1/2))
        Sy = np.kron(E, sg.op_Sy(1/2))
        Sz = np.kron(E, sg.op_Sz(1/2))

        # Construct the Cartesian spin vectors.
        I = np.array([[Ix, Iy, Iz]], dtype=complex)
        S = np.array([[Sx], [Sy], [Sz]], dtype=complex)

        # Evaluate the Cartesian contraction explicitly.
        left = np.zeros_like(Ix)
        for i in range(A.shape[0]):
            for s in range(A.shape[1]):
                left += A[i, s] * I[0, i] @ S[s, 0]

        # Convert the tensor to spherical components.
        A = _la.cartesian_tensor_to_spherical_tensor(A)

        # Evaluate the same contraction in the spherical convention.
        right = np.zeros_like(Ix)
        for l in range(0, 3):
            for q in range(-l, l+1):
                right += (
                    (-1)**(q) * A[(l, q)] * 
                    sg.op_T_coupled(l, -q, 1, 1/2, 1, 1/2)
                )

        # Check that both conventions give the same result.
        self.assertTrue(np.allclose(left, right))

    def test_CG_coeff(self):
        """
        Test Clebsch-Gordan coefficients against known values.
        """

        # Compare against tabulated coefficient values.
        self.assertAlmostEqual(_la.CG_coeff(1/2, 1/2, 1/2, 1/2, 1, 1), 1)
        self.assertAlmostEqual(_la.CG_coeff(1/2, -1/2, 1/2, -1/2, 1, -1), 1)
        self.assertAlmostEqual(
            _la.CG_coeff(1/2, 1/2, 1/2, -1/2, 1, 0),
            math.sqrt(1/2)
        )
        self.assertAlmostEqual(
            _la.CG_coeff(1/2, 1/2, 1/2, -1/2, 0, 0),
            math.sqrt(1/2)
        )
        self.assertAlmostEqual(
            _la.CG_coeff(1/2, -1/2, 1/2, 1/2, 1, 0),
            math.sqrt(1/2)
        )
        self.assertAlmostEqual(
            _la.CG_coeff(1/2, -1/2, 1/2, 1/2, 0, 0),
            -math.sqrt(1/2)
        )
        self.assertAlmostEqual(_la.CG_coeff(1, 1, 1/2, 1/2, 3/2, 3/2), 1)
        self.assertAlmostEqual(
            _la.CG_coeff(1, 1, 1/2, -1/2, 3/2, 1/2),
            math.sqrt(1/3)
        )
        self.assertAlmostEqual(
            _la.CG_coeff(1, 1, 1/2, -1/2, 1/2, 1/2),
            math.sqrt(2/3)
        )
        self.assertAlmostEqual(
            _la.CG_coeff(1, 0, 1/2, 1/2, 3/2, 1/2),
            math.sqrt(2/3)
        )
        self.assertAlmostEqual(
            _la.CG_coeff(1, 0, 1/2, 1/2, 1/2, 1/2),
            -math.sqrt(1/3)
        )

    def test_custom_dot(self):
        """
        Test the custom sparse matrix product against SciPy.
        """

        # Create two sparse matrices with compatible dimensions.
        A = csc_array([
            [1 + 1j, 0,      3 + 3j],
            [4 + 4j, 5 + 5j, 0     ],
        ], dtype=complex)
        B = csc_array([
            [0,      2 + 2j, 0,      0],
            [0,      0,      6 + 6j, 0],
            [7 + 7j, 0,      9 + 9j, 0],
        ], dtype=complex)

        # Compare the general sparse product with SciPy.
        C_SciPy = A @ B
        C_custom = _la.custom_dot(A, B, zero_value=1e-18)
        self.assertTrue(np.allclose(C_SciPy.toarray(), C_custom.toarray()))

        # Compare an empty-left product with SciPy.
        A_empty = csc_array(A.shape)
        C_SciPy = A_empty @ B
        C_custom = _la.custom_dot(A_empty, B, zero_value=1e-18)
        self.assertTrue(np.allclose(C_SciPy.toarray(), C_custom.toarray()))

        # Compare an empty-right product with SciPy.
        B_empty = csc_array(B.shape)
        C_SciPy = A @ B_empty
        C_custom = _la.custom_dot(A, B_empty, zero_value=1e-18)
        self.assertTrue(np.allclose(C_SciPy.toarray(), C_custom.toarray()))

        # Compare an empty-by-empty product with SciPy.
        C_SciPy = A_empty @ B_empty
        C_custom = _la.custom_dot(A_empty, B_empty, zero_value=1e-18)
        self.assertTrue(np.allclose(C_SciPy.toarray(), C_custom.toarray()))

        # Compare several index and data type combinations with SciPy.
        dtypes_I = [np.int32, np.int64]
        dtypes_T = [np.int32, np.int64, np.float64, np.complex128]
        for dtype_AI in dtypes_I:
            for dtype_BI in dtypes_I:
                for dtype_AT in dtypes_T:
                    for dtype_BT in dtypes_T:
                        A_curr = deepcopy(A)
                        B_curr = deepcopy(B)
                        A_curr.data = A_curr.data.astype(dtype_AT)
                        A_curr.indices = A_curr.indices.astype(dtype_AI)
                        A_curr.indptr = A_curr.indptr.astype(dtype_AI)
                        B_curr.data = B_curr.data.astype(dtype_BT)
                        B_curr.indices = B_curr.indices.astype(dtype_BI)
                        B_curr.indptr = B_curr.indptr.astype(dtype_BI)
                        C_custom = _la.custom_dot(A, B, zero_value=1e-18)
                        C_SciPy = A @ B
                        self.assertTrue(
                            np.allclose(C_SciPy.toarray(), C_custom.toarray())
                        )
                        
        # Compare a product that yields an empty result.
        A = csc_array([
            [0, 1],
            [0, 0]
        ])
        C_SciPy = A @ A
        C_custom = _la.custom_dot(A, A, zero_value=1e-18)
        self.assertTrue(np.allclose(C_SciPy.toarray(), C_custom.toarray()))
                        
    def test_expm_vec_taylor(self):
        """
        Test the Taylor-based action of the matrix exponential on a vector.
        """

        # Create a dense test matrix.
        A = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ])

        # Create a dense column vector.
        v = np.array([[1], [2], [3]])

        # Compute the exponential action for all dense and sparse combinations.
        eAv_dd = _la.expm_vec_taylor(A, v, 1e-18)
        eAv_ds = _la.expm_vec_taylor(A, csc_array(v), 1e-18)
        eAv_sd = _la.expm_vec_taylor(csc_array(A), v, 1e-18)
        eAv_ss = _la.expm_vec_taylor(csc_array(A), csc_array(v), 1e-18)

        # Compute the SciPy reference result.
        eAv_ref = expm(A) @ v

        # Compare all implementations with the reference result.
        self.assertTrue(np.allclose(eAv_dd, eAv_ref))
        self.assertTrue(np.allclose(eAv_ds, eAv_ref))
        self.assertTrue(np.allclose(eAv_sd, eAv_ref))
        self.assertTrue(np.allclose(eAv_ss.toarray(), eAv_ref))
                        
    def test_expm_vec(self):
        """
        Test the action of the matrix exponential on a vector.
        """

        # Create a dense test matrix.
        A = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ])

        # Create a dense column vector.
        v = np.array([[1], [2], [3]])

        # Compute the exponential action for all dense and sparse combinations.
        eAv_dd = _la.expm_vec(A, v, 1e-18)
        eAv_ds = _la.expm_vec(A, csc_array(v), 1e-18)
        eAv_sd = _la.expm_vec(csc_array(A), v, 1e-18)
        eAv_ss = _la.expm_vec(csc_array(A), csc_array(v), 1e-18)

        # Compute the SciPy reference result.
        eAv_ref = expm(A) @ v

        # Compare all implementations with the reference result.
        self.assertTrue(np.allclose(eAv_dd, eAv_ref))
        self.assertTrue(np.allclose(eAv_ds, eAv_ref))
        self.assertTrue(np.allclose(eAv_sd, eAv_ref))
        self.assertTrue(np.allclose(eAv_ss.toarray(), eAv_ref))