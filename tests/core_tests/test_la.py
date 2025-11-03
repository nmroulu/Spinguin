import unittest
import numpy as np
import math
from scipy.sparse.linalg import expm
from scipy.sparse import csc_array, random_array
from spinguin.la import _la
from spinguin.utils import HidePrints
import spinguin as sg

class TestLinearAlgebraMethods(unittest.TestCase):

    def test_isvector(self):

        # Create column vectors, row vectors, and arrays
        row1 = np.array([[1, 0, 0]])
        row2 = csc_array([[0, 1, 0, 7]])
        col1 = row1.T
        col2 = row2.T
        arr1 = np.array([[1, 0, 0],
                         [0, 1, 0],
                         [0, 0, 1]])
        arr2 = np.array([[[1], [2], [3]]])
        arr3 = np.array([1, 0, 0])

        # Check vectors with the correct order
        self.assertEqual(_la.isvector(row1, 'row'), True)
        self.assertEqual(_la.isvector(row2, 'row'), True)
        self.assertEqual(_la.isvector(col1, 'col'), True)
        self.assertEqual(_la.isvector(col2, 'col'), True)

        # Check vectors with the incorrect order
        self.assertEqual(_la.isvector(row1, 'col'), False)
        self.assertEqual(_la.isvector(row2, 'col'), False)
        self.assertEqual(_la.isvector(col1, 'row'), False)
        self.assertEqual(_la.isvector(col2, 'row'), False)

        # Check other 2D arrays
        self.assertEqual(_la.isvector(arr1, 'col'), False)
        self.assertEqual(_la.isvector(arr1, 'row'), False)

        # Check arrays with incorrect shapes
        self.assertRaises(ValueError, _la.isvector, arr2, 'col')
        self.assertRaises(ValueError, _la.isvector, arr2, 'row')
        self.assertRaises(ValueError, _la.isvector, arr3, 'col')
        self.assertRaises(ValueError, _la.isvector, arr3, 'row')

    def test_norm_1(self):

        # Create a 3x3 array
        A = np.random.rand(3, 3)

        # Test using NumPy arrays against the value given by NumPy
        self.assertAlmostEqual(_la.norm_1(A, 'row'),
                               np.linalg.norm(A, ord=np.inf))
        self.assertAlmostEqual(_la.norm_1(A, 'col'), np.linalg.norm(A, ord=1))

        # Test using sparse arrays against the value given by NumPy
        self.assertAlmostEqual(_la.norm_1(csc_array(A), 'row'),
                               np.linalg.norm(A, ord=np.inf))
        self.assertAlmostEqual(_la.norm_1(csc_array(A), 'col'),
                               np.linalg.norm(A, ord=1))

    def test_expm(self):

        # Create a 3x3 array with large numbers
        A = np.array([[1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 9]])

        # Compare against the value given by SciPy
        self.assertTrue(np.allclose(_la.expm(A, 1e-32), expm(A)))

        # Perform the same test with a SciPy sparse array
        A = csc_array(A)

        # Compare against the value given by SciPy
        self.assertTrue(np.allclose(_la.expm(A, 1e-32).toarray(),
                                    expm(A).toarray()))

    def test_eliminate_small(self):

        # Create a test array
        A_sp = csc_array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ])
        A_np = A_sp.toarray()
        
        # Eliminate small values from the arrays
        _la.eliminate_small(A_sp, zero_value=5)
        _la.eliminate_small(A_np, zero_value=5)

        # Create a comparison array
        B = np.array([[0, 0, 0],
                      [0, 5, 6],
                      [7, 8, 9]])

        # Compare and check the number of non-zeros
        self.assertTrue(np.array_equal(A_sp.toarray(), B))
        self.assertTrue(np.array_equal(A_np, B))
        self.assertEqual(A_sp.nnz, 5)

    def test_sparse_bytes(self):

        # Create a large random array
        A = random_array((1000, 1000), density=0.5, format='csc')

        # Convert to byte representation
        A_bytes = _la.sparse_to_bytes(A)
        
        # Convert back to sparse representation
        B = _la.bytes_to_sparse(A_bytes)

        # Compare the arrays
        self.assertTrue(np.allclose(A.toarray(), B.toarray()))

    def test_comm(self):

        # Create a random array
        A = np.random.rand(3, 3)

        # Commutator with itself should be zero
        self.assertTrue(np.allclose(_la.comm(A, A), np.zeros_like(A)))

        # Commutator with the identity array should be zero
        self.assertTrue(np.allclose(_la.comm(A, np.eye(3)), np.zeros_like(A)))

    def test_find_common_rows(self):

        # Create test arrays
        A = np.array([[0, 0, 0],
                      [0, 0, 1],
                      [0, 1, 0],
                      [1, 0, 0]])
        B = np.array([[0, 0, 0],
                      [4, 5, 6],
                      [1, 0, 0]])
        
        # Find indices that should return common rows
        A_ind, B_ind = _la.find_common_rows(A, B)

        # Check that the arrays are equal
        self.assertTrue(np.array_equal(A[A_ind], B[B_ind]))

    def test_auxiliary_matrix_expm(self):

        # Create the auxiliary matrix components in sparse format
        A_sp = csc_array([[1, 2, 3],
                          [4, 5, 6],
                          [7, 8, 9]]) * 1e-3
        B_sp = csc_array([[4, 5, 6],
                          [7, 8, 9],
                          [1, 2, 3]]) * 1e-3
        C_sp = csc_array([[7, 8, 9],
                          [1, 2, 3],
                          [4, 5, 6]]) * 1e-3
        
        # Get same components in dense format
        A_dn = A_sp.toarray()
        B_dn = B_sp.toarray()
        C_dn = C_sp.toarray()

        # Set integration time
        T = 1
        
        # Compute the auxiliary matrix exponential
        expm_aux_sp = _la.auxiliary_matrix_expm(
            A_sp, B_sp, C_sp, T, zero_value=1e-18)
        expm_aux_dn = _la.auxiliary_matrix_expm(
            A_dn, B_dn, C_dn, T, zero_value=1e-18)

        # Extract the sparse components
        top_l_sp = expm_aux_sp[:A_sp.shape[0], :A_sp.shape[1]]
        top_r_sp = expm_aux_sp[:A_sp.shape[0], A_sp.shape[1]:]
        bot_l_sp = expm_aux_sp[A_sp.shape[0]:, :A_sp.shape[1]]
        bot_r_sp = expm_aux_sp[A_sp.shape[0]:, A_sp.shape[1]:]

        # Extract the dense components
        top_l_dn = expm_aux_dn[:A_dn.shape[0], :A_dn.shape[1]]
        top_r_dn = expm_aux_dn[:A_dn.shape[0], A_dn.shape[1]:]
        bot_l_dn = expm_aux_dn[A_dn.shape[0]:, :A_dn.shape[1]]
        bot_r_dn = expm_aux_dn[A_dn.shape[0]:, A_dn.shape[1]:]

        # Compute the components manually for reference
        with HidePrints():
            top_l_ref = _la.expm(A_dn*T, zero_value=1e-18)
            top_r_ref = np.zeros(A_dn.shape, dtype=complex)
            for t in np.linspace(0, T, 1000):
                top_r_ref += _la.expm(-A_dn*t, zero_value=1e-18) @ B_dn @ \
                             _la.expm(C_dn*t, zero_value=1e-18) * (1/1000)
            top_r_ref = (_la.expm(A_dn*T, zero_value=1e-18) @ top_r_ref)
            bot_l_ref = np.zeros_like(bot_l_dn)
            bot_r_ref = _la.expm(C_dn*T, zero_value=1e-18)

        # Verify the sparse components
        self.assertTrue(np.allclose(top_l_sp.toarray(), top_l_ref))
        self.assertTrue(np.allclose(top_r_sp.toarray(), top_r_ref))
        self.assertTrue(np.allclose(bot_l_sp.toarray(), bot_l_ref))
        self.assertTrue(np.allclose(bot_r_sp.toarray(), bot_r_ref))

        # Verify the dense components
        self.assertTrue(np.allclose(top_l_dn, top_l_ref))
        self.assertTrue(np.allclose(top_r_dn, top_r_ref))
        self.assertTrue(np.allclose(bot_l_dn, bot_l_ref))
        self.assertTrue(np.allclose(bot_r_dn, bot_r_ref))

    def test_angle_between_vectors(self):

        # Create test vectors
        v1 = np.array([1, 0])
        v2 = np.array([0, 1])
        v3 = np.array([1, 1])

        # Compare with known results
        self.assertAlmostEqual(_la.angle_between_vectors(v1, v1), 0)
        self.assertAlmostEqual(_la.angle_between_vectors(v1, v2), np.pi/2)
        self.assertAlmostEqual(_la.angle_between_vectors(v1, -v1), np.pi)
        self.assertAlmostEqual(_la.angle_between_vectors(v1, v3), np.pi/4)

    def test_decompose_matrix(self):

        # Generate a test array
        A = np.random.rand(3, 3)

        # Decompose the matrix
        iso, asym, sym = _la.decompose_matrix(A)

        # Use the matrix properties for checking
        self.assertTrue(np.allclose(A, iso+asym+sym))
        self.assertTrue(np.allclose(asym, -asym.T))
        self.assertTrue(np.allclose(sym, sym.T))

    def test_principal_axis_system(self):

        # Generate a test array
        A = np.random.rand(3, 3)
        _, _, A_sym = _la.decompose_matrix(A)

        eigenvalues, eigenvectors, tensor_PAS = _la.principal_axis_system(A)

        # Check that the eigenvectors diagonalize the symmetric part
        self.assertTrue(np.allclose(
            A_sym, 
            np.linalg.inv(eigenvectors) @ np.diag(eigenvalues) @ eigenvectors))

        # Check that the original tensor can be reconstructed
        self.assertTrue(np.allclose(
            A, np.linalg.inv(eigenvectors) @ tensor_PAS @ eigenvectors))

    def test_cartesian_tensor_to_spherical_tensor(self):

        # Create a random tensor
        tensor = np.random.rand(3, 3)

        # Project the components of the spherical tensors (double outer product
        # convention) and compare
        for l in range(0, 3):
            for q in range(-l, l+1):
                self.assertAlmostEqual(
                    (tensor @ spherical_tensor(l, q)).trace(),
                    _la.cartesian_tensor_to_spherical_tensor(tensor)[(l, q)])

    def test_vector_to_spherical_tensor(self):

        # Create a random vector
        vector = np.random.rand(3)

        # Project the components of the spherical tensors and compare
        for q in range(-1, 2):
            self.assertAlmostEqual(
                np.inner(spherical_vector(1, q), vector),
                _la.vector_to_spherical_tensor(vector)[(1, q)])

    def test_cartesian_to_spherical_tensor_conventions(self):

        # Create a random Cartesian interaction tensor
        A = np.random.rand(3, 3)

        # Single-spin unit operator
        sg.config.sparse_operator = False
        E = sg.op_E(1/2)

        # Spin operators for I
        Ix = np.kron(sg.op_Sx(1/2), E)
        Iy = np.kron(sg.op_Sy(1/2), E)
        Iz = np.kron(sg.op_Sz(1/2), E)

        # Spin operators for S
        Sx = np.kron(E, sg.op_Sx(1/2))
        Sy = np.kron(E, sg.op_Sy(1/2))
        Sz = np.kron(E, sg.op_Sz(1/2))

        # Construct the Cartesian spin vectors
        I = np.array([[Ix, Iy, Iz]], dtype=complex)
        S = np.array([[Sx],
                      [Sy],
                      [Sz]], dtype=complex)

        # Perform the dot product manually
        left = np.zeros_like(Ix)
        for i in range(A.shape[0]):
            for s in range(A.shape[1]):
                left += A[i, s] * I[0, i] @ S[s, 0]

        # Convert A to spherical tensors
        A = _la.cartesian_tensor_to_spherical_tensor(A)

        # Use spherical tensors
        right = np.zeros_like(Ix)
        for l in range(0, 3):
            for q in range(-l, l+1):
                right += (-1)**(q) * A[(l, q)] * \
                         sg.op_T_coupled(l, -q, 1, 1/2, 1, 1/2)

        # Both conventions should give the same result
        self.assertTrue(np.allclose(left, right))

    def test_CG_coeff(self):

        # Test against known values
        self.assertAlmostEqual(_la.CG_coeff(1/2, 1/2, 1/2, 1/2, 1, 1), 1)
        self.assertAlmostEqual(_la.CG_coeff(1/2, -1/2, 1/2, -1/2, 1, -1), 1)
        self.assertAlmostEqual(_la.CG_coeff(1/2, 1/2, 1/2, -1/2, 1, 0),
                               math.sqrt(1/2))
        self.assertAlmostEqual(_la.CG_coeff(1/2, 1/2, 1/2, -1/2, 0, 0),
                               math.sqrt(1/2))
        self.assertAlmostEqual(_la.CG_coeff(1/2, -1/2, 1/2, 1/2, 1, 0),
                               math.sqrt(1/2))
        self.assertAlmostEqual(_la.CG_coeff(1/2, -1/2, 1/2, 1/2, 0, 0),
                               -math.sqrt(1/2))
        self.assertAlmostEqual(_la.CG_coeff(1, 1, 1/2, 1/2, 3/2, 3/2), 1)
        self.assertAlmostEqual(_la.CG_coeff(1, 1, 1/2, -1/2, 3/2, 1/2),
                               math.sqrt(1/3))
        self.assertAlmostEqual(_la.CG_coeff(1, 1, 1/2, -1/2, 1/2, 1/2),
                               math.sqrt(2/3))
        self.assertAlmostEqual(_la.CG_coeff(1, 0, 1/2, 1/2, 3/2, 1/2),
                               math.sqrt(2/3))
        self.assertAlmostEqual(_la.CG_coeff(1, 0, 1/2, 1/2, 1/2, 1/2),
                               -math.sqrt(1/3))

    def test_custom_dot(self):

        # Create two random arrays
        A = random_array((200, 300), density=0.2, format='csc', dtype=complex)
        B = random_array((300, 400), density=0.2, format='csc', dtype=complex)

        # Compare against SciPy
        C_SciPy = A @ B
        C_custom = _la.custom_dot(A, B, zero_value=1e-18)
        self.assertTrue(np.allclose(C_SciPy.toarray(), C_custom.toarray()))

        # Test empty @ non-empty
        A_empty = csc_array((200, 300))
        C_SciPy = A_empty @ B
        C_custom = _la.custom_dot(A_empty, B, zero_value=1e-18)
        self.assertTrue(np.allclose(C_SciPy.toarray(), C_custom.toarray()))

        # Test non-empty @ empty
        B_empty = csc_array((300, 400))
        C_SciPy = A @ B_empty
        C_custom = _la.custom_dot(A, B_empty, zero_value=1e-18)
        self.assertTrue(np.allclose(C_SciPy.toarray(), C_custom.toarray()))

        # Test empty @ empty
        C_SciPy = A_empty @ B_empty
        C_custom = _la.custom_dot(A_empty, B_empty, zero_value=1e-18)
        self.assertTrue(np.allclose(C_SciPy.toarray(), C_custom.toarray()))

        # Test with non-empty arrays where result will be empty
        A = csc_array([
            [0, 1],
            [0, 0]])
        C_SciPy = A @ A
        C_custom = _la.custom_dot(A, A, zero_value=1e-18)
        self.assertTrue(np.allclose(C_SciPy.toarray(), C_custom.toarray()))

        # Test repeatedly using same array
        A = random_array((200, 200), density=0.2, format="csc", dtype=complex)
        for _ in range(10):
            A = _la.custom_dot(A, A, zero_value=1e-32)

        # Test varying dtypes
        dtypes_I = [np.int32, np.int64]
        dtypes_T = [np.int32, np.int64, np.float64, np.complex128]
        for dtype_AI in dtypes_I:
            for dtype_BI in dtypes_I:
                for dtype_AT in dtypes_T:
                    for dtype_BT in dtypes_T:
                        A = random_array((200, 200), density=0.2).tocsc()
                        B = random_array((200, 200), density=0.2).tocsc()
                        A.data = A.data.astype(dtype_AT)
                        A.indices = A.indices.astype(dtype_AI)
                        A.indptr = A.indptr.astype(dtype_AI)
                        B.data = B.data.astype(dtype_BT)
                        B.indices = B.indices.astype(dtype_BI)
                        B.indptr = B.indptr.astype(dtype_BI)
                        C_custom = _la.custom_dot(A, B, zero_value=1e-18)
                        C_SciPy = A @ B
                        self.assertTrue(np.allclose(C_SciPy.toarray(),
                                                    C_custom.toarray()))
                        
    def test_expm_vec_taylor(self):

        # Create a 3x3 array with large numbers
        A = np.array([[1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 9]])

        # Create a column vector
        v = np.array([[1], [2], [3]])

        # Calculate the action of matrix exponential to the vector using
        # sparse and dense matrices and vectors
        eAv_dd = _la.expm_vec_taylor(A, v, 1e-18)
        eAv_ds = _la.expm_vec_taylor(A, csc_array(v), 1e-18)
        eAv_sd = _la.expm_vec_taylor(csc_array(A), v, 1e-18)
        eAv_ss = _la.expm_vec_taylor(csc_array(A), csc_array(v), 1e-18)

        # Calculate reference
        eAv_ref = expm(A) @ v

        # Should be equal
        self.assertTrue(np.allclose(eAv_dd, eAv_ref))
        self.assertTrue(np.allclose(eAv_ds, eAv_ref))
        self.assertTrue(np.allclose(eAv_sd, eAv_ref))
        self.assertTrue(np.allclose(eAv_ss.toarray(), eAv_ref))
                        
    def test_expm_vec(self):

        # Create a 3x3 array with large numbers
        A = np.array([[1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 9]])

        # Create a column vector
        v = np.array([[1], [2], [3]])

        # Calculate the action of matrix exponential to the vector using
        # sparse and dense matrices and vectors
        eAv_dd = _la.expm_vec(A, v, 1e-18)
        eAv_ds = _la.expm_vec(A, csc_array(v), 1e-18)
        eAv_sd = _la.expm_vec(csc_array(A), v, 1e-18)
        eAv_ss = _la.expm_vec(csc_array(A), csc_array(v), 1e-18)

        # Calculate reference
        eAv_ref = expm(A) @ v

        # Should be equal
        self.assertTrue(np.allclose(eAv_dd, eAv_ref))
        self.assertTrue(np.allclose(eAv_ds, eAv_ref))
        self.assertTrue(np.allclose(eAv_sd, eAv_ref))
        self.assertTrue(np.allclose(eAv_ss.toarray(), eAv_ref))

def spherical_tensor(l, q):
    """
    Helper function for tests.

    Calculates a 3x3 spherical tensor of rank l and projection q expressed
    in the Cartesian basis. Works by combining the covariant spherical basis
    vectors.

    Recipe described in Man: Cartesian and Spherical Tensors in NMR Hamiltonians
    https://doi.org/10.1002/cmr.a.21289
    """

    # Initialize the tensor
    t_lq = np.zeros((3, 3), dtype=complex)

    # Coupling of angular momenta
    for q1 in range(-1, 2):
        for q2 in range(-1, 2):
            t_lq += _la.CG_coeff(1, q1, 1, q2, l, q) * \
                    np.outer(spherical_vector(1, q1), spherical_vector(1, q2))

    return t_lq

def spherical_vector(l, q):
    """
    Helper function for tests.

    Constructs a covariant spherical vector of rank l and projection q expressed
    in the Cartesian basis.
    """

    # Cartesian basis vectors
    e_x = np.array([1, 0, 0])
    e_y = np.array([0, 1, 0])
    e_z = np.array([0, 0, 1])

    # Get the spherical vector
    if l == 1 and q == 1:
        v = -1/np.sqrt(2) * (e_x + 1j*e_y)
    elif l == 1 and q == 0:
        v = e_z
    elif l == 1 and q == -1:
        v = 1/np.sqrt(2) * (e_x - 1j*e_y)

    return v