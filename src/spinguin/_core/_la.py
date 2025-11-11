"""
This module provides various linear algebra tools required for spin dynamics
simulations.
"""

# Imports
import math
import numpy as np
from numpy.typing import ArrayLike
from scipy.sparse import eye_array, csc_array, block_array, issparse
from scipy.io import mmwrite, mmread
from io import BytesIO
from functools import lru_cache
from sympy.physics.quantum.cg import CG
from spinguin._core._sparse_dot import sparse_dot as _sparse_dot
from spinguin._core._intersect_indices import intersect_indices
from spinguin._core._hide_prints import HidePrints
from spinguin._core._nmr_isotopes import ISOTOPES
from multiprocessing.shared_memory import SharedMemory

def write_shared_sparse(A: csc_array) -> tuple[
    dict[str, str | np.dtype | tuple[int]],
    tuple[SharedMemory, SharedMemory, SharedMemory]]:
    """
    Creates a shared memory representation of a sparse CSC array.

    Parameters
    ----------
    A : csc_array
        Sparse array to be shared.

    Returns
    -------
    A_shared : dict
        Dictionary containing shared memory names and metadata for the sparse
        array's data, indices, and indptr, along with their shapes and dtypes.
    A_shm : tuple
        Tuple containing the shared memory objects for the sparse array's data,
        indices, and indptr.
    """
    # Create a shared memory of the sparse array
    A_data_shm = SharedMemory(create=True, size=A.data.nbytes)
    A_indices_shm = SharedMemory(create=True, size=A.indices.nbytes)
    A_indptr_shm = SharedMemory(create=True, size=A.indptr.nbytes)
    A_data_shared = np.ndarray(A.data.shape, dtype=A.data.dtype,
                               buffer=A_data_shm.buf)
    A_indices_shared = np.ndarray(A.indices.shape, dtype=A.indices.dtype,
                                  buffer=A_indices_shm.buf)
    A_indptr_shared = np.ndarray(A.indptr.shape, dtype=A.indptr.dtype,
                                 buffer=A_indptr_shm.buf)
    A_data_shared[:] = A.data[:]
    A_indices_shared[:] = A.indices[:]
    A_indptr_shared[:] = A.indptr[:]

    # Save the information of the memory to a dictionary
    A_shared = {
        'A_data_shm_name' : A_data_shm.name,
        'A_indices_shm_name' : A_indices_shm.name,
        'A_indptr_shm_name' : A_indptr_shm.name,
        'A_data_shape' : A.data.shape,
        'A_indices_shape' : A.indices.shape,
        'A_indptr_shape' : A.indptr.shape,
        'A_data_dtype' : A.data.dtype,
        'A_indices_dtype' : A.indices.dtype,
        'A_indptr_dtype' : A.indptr.dtype,
        'A_shape' : A.shape
    }

    # Combine the shared memories into one tuple
    A_shm = (A_data_shm, A_indices_shm, A_indptr_shm)

    return A_shared, A_shm

def read_shared_sparse(A_shared: dict[str, str | np.dtype | tuple[int]]) -> \
    tuple[csc_array, tuple[SharedMemory, SharedMemory, SharedMemory]]:
    """
    Reads a shared memory representation of a sparse CSC array and reconstructs
    it.

    Parameters
    ----------
    A_shared : dict
        Dictionary containing shared memory names and metadata for the sparse
        array's data, indices, and indptr, along with their shapes and dtypes.

    Returns
    -------
    A : csc_array
        Sparse array reconstructed from the shared memory.
    A_shm : tuple
        Tuple containing the shared memory objects for the sparse array's data,
        indices, and indptr.
    """
    # Parse the dictionary
    A_data_shm_name = A_shared['A_data_shm_name']
    A_indices_shm_name = A_shared['A_indices_shm_name']
    A_indptr_shm_name = A_shared['A_indptr_shm_name']
    A_data_shape = A_shared['A_data_shape']
    A_indices_shape = A_shared['A_indices_shape']
    A_indptr_shape = A_shared['A_indptr_shape']
    A_data_dtype = A_shared['A_data_dtype']
    A_indices_dtype = A_shared['A_indices_dtype']
    A_indptr_dtype = A_shared['A_indptr_dtype']
    A_shape = A_shared['A_shape']

    # Obtain the shared memories
    A_data_shm = SharedMemory(name=A_data_shm_name)
    A_indices_shm = SharedMemory(name=A_indices_shm_name)
    A_indptr_shm = SharedMemory(name=A_indptr_shm_name)

    # Obtain the previously shared array
    A_data = np.ndarray(shape=A_data_shape, dtype=A_data_dtype,
                        buffer=A_data_shm.buf)
    A_indices = np.ndarray(shape=A_indices_shape, dtype=A_indices_dtype,
                           buffer=A_indices_shm.buf)
    A_indptr = np.ndarray(shape=A_indptr_shape, dtype=A_indptr_dtype,
                          buffer=A_indptr_shm.buf)

    # Create the sparse array
    A = csc_array((A_data, A_indices, A_indptr), shape=A_shape, copy=False)

    # Combine the shared memories into one tuple
    A_shm = (A_data_shm, A_indices_shm, A_indptr_shm)

    return A, A_shm

def isvector(v: csc_array | np.ndarray, ord: str = "col") -> bool:
    """
    Checks if the given array is a vector.

    Parameters
    ----------
    v : csc_array or ndarray
        Array to be checked. Must be two-dimensional.
    ord : str
        Can be either "col" or "row".

    Returns
    -------
    bool
        True if the array is a vector.
    """

    # Check whether the array is two-dimensional
    if len(v.shape) != 2:
        raise ValueError("Input array must be two-dimensional.")

    # Determine whether to check for row or column vector
    if ord == "col":
        i = 1
    elif ord == "row":
        i = 0
    else:
        raise ValueError(f"Invalid value for ord: {ord}")

    # Check whether the array is a vector
    return v.shape[i] == 1

def norm_1(A: csc_array | np.ndarray, ord: str = 'row') -> float:
    """
    Calculates the 1-norm of a matrix.

    Parameters
    ----------
    A : csc_array or ndarray
        Array for which the norm is calculated.
    ord : str, default='row'
        Either 'row' or 'col', specifying the direction for the 1-norm
        calculation.

    Returns
    -------
    norm_1 : float
        1-norm of the given array `A`.
    """

    # Process either row- or column-wise
    if ord == 'row':
        axis = 1
    elif ord == 'col':
        axis = 0
    else:
        raise ValueError(f"Invalid value for ord: {ord}")

    # Calculate sums along rows or columns and get the maximum of them
    return abs(A).sum(axis).max()

def expm(A: np.ndarray | csc_array,
         zero_value: float) -> np.ndarray | csc_array:
    """
    Calculates the matrix exponential of a SciPy sparse CSC array using the
    scaling and squaring method with the Taylor series, shown to be the fastest
    method in:

    https://doi.org/10.1016/j.jmr.2010.12.004

    This function uses a custom dot product implementation, which is more
    memory-efficient and parallelized.

    Parameters
    ----------
    A : ndarray or csc_array
        Array to be exponentiated.
    zero_value : float
        Values below this threshold are considered zero. Used to increase the
        sparsity of the result and estimate the convergence of the Taylor
        series.

    Returns
    -------
    expm_A : ndarray or csc_array
        Matrix exponential of `A`.
    """

    print("Calculating the matrix exponential...")

    # Calculate the norm of A
    norm_A = norm_1(A, ord='col')

    # If the norm of the matrix is too large, scale the matrix down
    if norm_A > 1:

        # Calculate the scaling factor for the matrix
        scaling_count = int(math.ceil(math.log2(norm_A)))
        scaling_factor = 2 ** scaling_count

        print(f"Scaling the matrix down {scaling_count} times.")

        # Scale the matrix down
        A = A / scaling_factor

        # Calculate the expm of the scaled matrix using the Taylor series
        expm_A = expm_taylor(A, zero_value)

        # Scale the matrix exponential back up by repeated squaring
        for i in range(scaling_count):
            print(f"Squaring the matrix. Step {i+1} of {scaling_count}.")
            expm_A = custom_dot(expm_A, expm_A, zero_value)
    
    # If the norm of the matrix is small, proceed without scaling
    else:
        expm_A = expm_taylor(A, zero_value)

    print("Matrix exponential completed.")

    return expm_A

def expm_taylor(A: np.ndarray | csc_array,
                zero_value: float) -> np.ndarray | csc_array:
    """
    Computes the matrix exponential using the Taylor series. This function is 
    adapted from an older SciPy version.

    It uses a custom dot product implementation, which is more memory-efficient 
    and parallelized.

    Parameters
    ----------
    A : ndarray or csc_array
        Matrix (N, N) to be exponentiated.
    zero_value : float
        Values below this threshold are considered zero. Used to increase
        sparsity and check the convergence of the series.

    Returns
    -------
    eA : ndarray or csc_array
        Matrix exponential of A.
    """

    print("Calculating the matrix exponential using Taylor series.")

    # Increase sparsity of A
    eliminate_small(A, zero_value)
    
    # Create a unit matrix for the first term
    eA = eye_array(A.shape[0], A.shape[0], dtype=complex, format='csc')

    # Make a copy for the terms
    trm = eA.copy()

    # Calculate new terms until their significance becomes negligible
    k = 1
    cont = True
    while cont:

        print(f"Taylor series term: {k}")

        # Get the next term
        trm = custom_dot(trm, A / k, zero_value)

        # Add the term to the result
        eA += trm

        # Increment the counter
        k += 1

        # Continue if the convergence criterion is not met
        if issparse(A):
            cont = (trm.nnz != 0)
        else:
            cont = (np.count_nonzero(trm) != 0)

    print("Taylor series converged.")

    return eA

def eliminate_small(A: np.ndarray | csc_array, zero_value: float):
    """
    Eliminates small values from the input matrix `A` by replacing values
    smaller than `zero_value` with zeros. Modification happens inplace.

    Parameters
    ----------
    A : ndarray or csc_array
        Array to be modified.
    zero_value : float
        Values smaller than this threshold are set to zero.
    """
    # Identify values smaller than the threshold and set them to zero
    if issparse(A):
        nonzero_mask = np.abs(A.data) < zero_value
        A.data[nonzero_mask] = 0
        A.eliminate_zeros()
    else:
        nonzero_mask = np.abs(A) < zero_value
        A[nonzero_mask] = 0

def sparse_to_bytes(A: csc_array) -> bytes:
    """
    Converts the given SciPy sparse array into a byte representation.

    Parameters
    ----------
    A : csc_array
        Sparse matrix to be converted into bytes.

    Returns
    -------
    A_bytes : bytes
        Byte representation of the input matrix.
    """
    
    # Initialize a BytesIO object
    bytes_io = BytesIO()

    # Write the matrix A to bytes
    mmwrite(bytes_io, A)

    # Retrieve the bytes
    A_bytes = bytes_io.getvalue()

    return A_bytes

def bytes_to_sparse(A_bytes: bytes) -> csc_array:
    """
    Converts a byte representation back to a SciPy sparse array.

    Parameters
    ----------
    A_bytes : bytes
        Byte representation of a SciPy sparse array.

    Returns
    -------
    A : csc_array
        Sparse array reconstructed from the byte representation.
    """

    # Initialize a BytesIO object
    bytes_io = BytesIO(A_bytes)

    # Read the SciPy sparse array from bytes
    A = mmread(bytes_io)

    return A

def comm(A: csc_array | np.ndarray,
         B: csc_array | np.ndarray) -> csc_array | np.ndarray:
    """
    Calculates the commutator [A, B] of two operators.

    Parameters
    ----------
    A : csc_array or ndarray
        First operator.
    B : csc_array or ndarray
        Second operator.

    Returns
    -------
    C : csc_array or ndarray
        Commutator [A, B].
    """

    # Compute the commutator
    C = A @ B - B @ A

    return C

def find_common_rows(A: np.ndarray,
                     B: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Identifies the indices of common rows between two arrays, `A` and `B`.
    Each row must appear only once in the arrays and they must be sorted
    in lexicographical order.

    Parameters
    ----------
    A : ndarray
        First array to compare.
    B : ndarray
        Second array to compare.

    Returns
    -------
    A_ind : ndarray
        Indices of the common rows in array `A`.
    B_ind : ndarray
        Indices of the common rows in array `B`.
    """

    # Handle special case where both arrays are empty
    if A.shape[1] == 0 and B.shape[1] == 0:
        A_ind = np.array([0])
        B_ind = np.array([0])
        return A_ind, B_ind
    
    # Get the row length ensuring the correct data type
    row_length = np.longlong(A.shape[1])
    
    # Convert the arrays to 1D
    A = A.ravel()
    B = B.ravel()

    # Ensure that the data types are correct
    A = A.astype(np.longlong)
    B = B.astype(np.longlong)

    # Find the common indices
    A_ind, B_ind = intersect_indices(A, B, row_length)

    return A_ind, B_ind

def auxiliary_matrix_expm(A: np.ndarray | csc_array,
                          B: np.ndarray | csc_array,
                          C: np.ndarray | csc_array,
                          t: float,
                          zero_value: float) -> csc_array:   
    """
    Computes the matrix exponential of an auxiliary matrix. This is used to 
    calculate the Redfield integral.

    Based on Goodwin and Kuprov (Eq. 3): https://doi.org/10.1063/1.4928978
    
    Parameters
    ----------
    A : ndarray or csc_array
        Top-left block of the auxiliary matrix.
    B : ndarray or csc_array
        Top-right block of the auxiliary matrix.
    C : ndarray or csc_array
        Bottom-right block of the auxiliary matrix.
    t : float
        Integration time.
    zero_value : float
        Threshold below which values are considered zero when exponentiating the
        auxiliary matrix using the Taylor series. This significantly impacts
        performance. Use the largest value that still provides correct results.
    
    Returns
    -------
    expm_aux : ndarray or csc_array
        Matrix exponential of the auxiliary matrix. The output is sparse or
        dense matching the sparsity of the input.
    """

    # Ensure that the input arrays are all either sparse or dense
    if not (issparse(A) == issparse(B) == issparse(C)):
        raise ValueError(f"All arrays A, B and C must be of same type.")

    # Are we using sparse?
    sparse = issparse(A)

    # Construct the auxiliary matrix
    if sparse:
        empty_array = csc_array(A.shape)
        aux = block_array([[A, B],
                        [empty_array, C]], format='csc')
    else:
        empty_array = np.zeros(A.shape)
        aux = np.block([[A, B],
                        [empty_array, C]])

    # Compute the matrix exponential of the auxiliary matrix
    with HidePrints():
        expm_aux = expm(aux * t, zero_value)

    return expm_aux

def angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Computes the angle between two vectors in radians.

    Parameters
    ----------
    v1 : ndarray
        First vector.
    v2 : ndarray
        Second vector.

    Returns
    -------
    theta : float
        Angle between the vectors in radians.
    """

    # Handle the case where the vectors are identical
    if np.array_equal(v1, v2):
        theta = 0
    else:
        theta = np.arccos(
            np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

    return theta

def decompose_matrix(matrix: np.ndarray) \
    -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Decomposes a matrix into three components:

    - Isotropic part.
    - Antisymmetric part.
    - Symmetric traceless part.

    Parameters
    ----------
    matrix : ndarray
        Matrix to decompose.

    Returns
    -------
    isotropic : ndarray
        Isotropic part of the input matrix.
    antisymmetric : ndarray
        Antisymmetric part of the input matrix.
    symmetric_traceless : ndarray
        Symmetric traceless part of the input matrix.
    """

    # Compute the isotropic, antisymmetric, and symmetric traceless parts
    isotropic = np.trace(matrix) * np.eye(matrix.shape[0]) / matrix.shape[0]
    antisymmetric = (matrix - matrix.T) / 2
    symmetric_traceless = (matrix + matrix.T) / 2 - isotropic
    
    return isotropic, antisymmetric, symmetric_traceless

def principal_axis_system(tensor: np.ndarray) \
    -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Determines the principal axis system (PAS) of a Cartesian tensor
    and transforms the tensor into the PAS.

    The PAS is defined as the coordinate system that diagonalizes
    the symmetric traceless part of the tensor.

    The eigenvalues are ordered as `(|largest|, |middle|, |smallest|)`.

    Parameters
    ----------
    tensor : np.ndarray
        Cartesian tensor to transform.

    Returns
    -------
    eigenvalues : ndarray
        Eigenvalues of the tensor in the PAS.
    eigenvectors : ndarray
        Two-dimensional array where rows contain the eigenvectors of the PAS.
    tensor_PAS : ndarray
        Tensor transformed into the PAS.
    """

    # Extract the symmetric traceless part of the tensor
    _, _, symmetric_traceless = decompose_matrix(tensor)

    # Diagonalize the symmetric traceless part
    eigenvalues, eigenvectors = np.linalg.eig(symmetric_traceless)

    # Sort eigenvalues and eigenvectors by the absolute value of eigenvalues
    idx = np.argsort(np.abs(eigenvalues))[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx].T

    # Transform the tensor into the principal axis system
    tensor_PAS = eigenvectors @ tensor @ np.linalg.inv(eigenvectors)

    return eigenvalues, eigenvectors, tensor_PAS

def cartesian_tensor_to_spherical_tensor(C: np.ndarray) -> dict:
    """
    Converts a rank-2 Cartesian tensor to a spherical tensor.

    Uses the double outer product (DOP) convention from:
    Eqs. 293-298 in Man: Cartesian and Spherical Tensors in NMR Hamiltonians
    https://doi.org/10.1002/cmr.a.21289

    Parameters
    ----------
    C : ndarray
        Rank-2 tensor in Cartesian coordinates.

    Returns
    -------
    spherical_tensor : dict
        Keys specify the rank and the projection (l, q), and the values
        are the components.
    """

    # Extract the Cartesian components
    C_xx, C_xy, C_xz = C[0, :]
    C_yx, C_yy, C_yz = C[1, :]
    C_zx, C_zy, C_zz = C[2, :]
    
    # Build the spherical tensor components
    spherical_tensor = {
        (0, 0): -1 / math.sqrt(3) * (C_xx + C_yy + C_zz),
        (1, 0): -1j / math.sqrt(2) * (C_xy - C_yx),
        (1, 1): -1 / 2 * (C_zx - C_xz + 1j * (C_zy - C_yz)),
        (1, -1): -1 / 2 * (C_zx - C_xz - 1j * (C_zy - C_yz)),
        (2, 0): 1 / math.sqrt(6) * (-C_xx + 2 * C_zz - C_yy),
        (2, 1): -1 / 2 * (C_xz + C_zx + 1j * (C_yz + C_zy)),
        (2, -1): 1 / 2 * (C_xz + C_zx - 1j * (C_yz + C_zy)),
        (2, 2): 1 / 2 * (C_xx - C_yy + 1j * (C_xy + C_yx)),
        (2, -2): 1 / 2 * (C_xx - C_yy - 1j * (C_xy + C_yx))
    }
    
    return spherical_tensor

def vector_to_spherical_tensor(vector: np.ndarray) -> dict:
    """
    Converts a Cartesian vector to a spherical tensor of rank 1.

    Uses the covariant components.
    Eq. 230 in Man: Cartesian and Spherical Tensors in NMR Hamiltonians
    https://doi.org/10.1002/cmr.a.21289

    Parameters
    ----------
    vector : ndarray
        Vector in the format [x, y, z].

    Returns
    -------
    spherical_tensor : dict
        Keys specify the rank and the projection (l, q), and the values
        are the components.   
    """

    # Build the spherical tensor
    spherical_tensor = {
        (1, 1): -1 / math.sqrt(2) * (vector[0] + 1j * vector[1]),
        (1, 0): vector[2],
        (1, -1): 1 / math.sqrt(2) * (vector[0] - 1j * vector[1])
    }

    return spherical_tensor

@lru_cache(maxsize=32784)
def CG_coeff(j1: float, m1: float,
             j2: float, m2: float,
             j3: float, m3: float) -> float:
    """
    Computes the Clebsch-Gordan coefficients.

    Parameters
    ----------
    j1 : float
        Angular momentum of state 1.
    m1 : float
        Magnetic quantum number of state 1.
    j2 : float
        Angular momentum of state 2.
    m2 : float
        Magnetic quantum number of state 2.
    j3 : float
        Total angular momentum of the coupled system.
    m3 : float
        Magnetic quantum number of the coupled system.

    Returns
    -------
    coeff : float
        Clebsch-Gordan coefficient.
    """

    # Get the coefficient
    coeff = float(CG(j1, m1, j2, m2, j3, m3).doit())

    return coeff

def custom_dot(
        A: np.ndarray | csc_array,
        B: np.ndarray | csc_array,
        zero_value: float
) -> csc_array:
    """
    User-friendly wrapper for the custom sparse matrix multiplication, which
    saves memory usage by dropping values smaller than `zero_value` during the
    calculation. The sparse multiplication is implemented with C++ / Cython and
    is parallelized with OpenMP.

    NOTE: If either of the input arrays is NumPy array, this function falls
    back to the regular `@` multiplication.

    Parameters
    ----------
    A : ndarray or csc_array
        First matrix in the multiplication.
    B : ndarray or csc_array
        Second matrix in the multiplication.
    zero_value : float
        Threshold under which the resulting matrix elements are considered as
        zero.

    Returns
    -------
    C : ndarray or csc_array
        Result of matrix multiplication.
    """
    # Check input types
    if isinstance(A, np.ndarray) or isinstance(B, np.ndarray):
        C = A @ B
        eliminate_small(C, zero_value)
    elif issparse(A) and issparse(B):
        A = A.tocsc()
        B = B.tocsc()
        C = _sparse_dot(A, B, zero_value)
    else:
        raise ValueError("Invalid input type for custom dot.")

    return C

def arraylike_to_tuple(A: ArrayLike) -> tuple:
    """
    Converts a 1-dimensional `ArrayLike` object into a Python tuple.

    Parameters
    ----------
    A : ArrayLike
        An object that can be converted into NumPy array.

    Returns
    -------
    A : tuple
        The original object represented as Python tuple.
    """

    # Convert to tuple
    A = np.asarray(A)
    if A.ndim == 0:
        A = tuple([A.item()])
    elif A.ndim == 1:
        A = tuple(A)
    else:
        raise ValueError(f"Cannot convert {A.ndim}-dimensional array into "
                         "tuple.")
    
    return A

def arraylike_to_array(A: ArrayLike) -> np.ndarray:
    """
    Converts an `ArrayLike` object into a NumPy array while ensuring
    that at least one dimension is created.

    Parameters
    ----------
    A : ArrayLike
        An object that can be converted into NumPy array.

    Returns
    -------
    A : ndarray
        The original object converted into a NumPy array.
    """

    # Convert to NumPy array and ensure at least one dimension
    A = np.asarray(A)
    A = np.atleast_1d(A)

    return A

def expm_vec_taylor(
    A: np.ndarray | csc_array,
    v: np.ndarray | csc_array,
    zero_value: float
) -> np.ndarray | csc_array:
    """
    Computes the action of the matrix exponential of `A` on the vector `v`,
    i.e., `expm(A) @ v` using the Taylor series.

    Parameters
    ----------
    A : ndarray or csc_array
        Square matrix (N, N).
    v : ndarray or csc_array
        Column vector (N, 1).
    zero_value : float
        Used to estimate the convergence of the Taylor series.

    Returns
    -------
    eAv : ndarray or csc_array
        Result of `expm(A) @ v`. Returns a sparse CSC array only when both input
        arrays are sparse.
    """
    # First term (k = 0)
    trm = v
    eAv = trm

    # Calculate higher order terms until they converge to zero
    k = 1
    cont = True
    while cont:

        # Get the current term
        trm = A @ (trm / k)

        # Set very small values to zero
        eliminate_small(trm, zero_value)

        # Add the term to the result
        eAv = eAv + trm

        # Increment the counter
        k += 1

        # Continue if the convergence criterion is not met
        if issparse(trm):
            cont = (trm.nnz != 0)
        else:
            cont = (np.count_nonzero(trm) != 0)

    return eAv

def expm_vec(
    A: np.ndarray | csc_array,
    v: np.ndarray | csc_array,
    zero_value: float
) -> np.ndarray | csc_array:
    """
    Computes the action of the matrix exponential of `A` on the vector `v`,
    i.e., `expm(A) @ v` using the Taylor series combined with the scaling of the
    input matrix `A`.

    Parameters
    ----------
    A : ndarray or csc_array
        Square matrix (N, N).
    v : ndarray or csc_array
        Column vector (N, 1).
    zero_value : float
        Used to estimate the convergence of the Taylor series.

    Returns
    -------
    eAv : ndarray or csc_array
        Result of `expm(A) @ v`. Returns a sparse CSC array only when both input
        arrays are sparse.
    """
    print("Calculating the action of matrix exponential on a vector...")

    # Calculate the norm of A
    norm_A = norm_1(A, ord='col')

    # Calculate the scaling factor for the matrix
    scaling_A = int(math.ceil(norm_A))

    # Scale the matrix
    print(f"Scaling the matrix by {scaling_A}.")
    A = A / scaling_A

    # Calculate a scaling factor for the zero value
    scaling_zv = abs(v).max()

    # Scale the zero-value
    print(f"Scaling the zero-value by {scaling_zv}.")
    zero_value = zero_value / scaling_zv

    # Initialise the result
    eAv = v

    # Calculate the expm*vec using the scaled matrix
    for i in range(scaling_A):
        print(f"Calculating expm(A)*vec. Step {i+1} of {scaling_A}.")
        eAv = expm_vec_taylor(A, eAv, zero_value)

    return eAv

def clear_cache_CG_coeff():
    """
    Clears the cache of the `CG_coeff` function.
    """
    # Clear the cache
    CG_coeff.cache_clear()