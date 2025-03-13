"""
la.py

Provides different linear algebra tools that are required in spin dynamics
simulations.
"""

# Imports
import math
import numpy as np
from scipy.sparse import eye_array, csc_array, block_array, issparse
from scipy.io import mmwrite, mmread
from io import BytesIO
from functools import lru_cache
from sympy.physics.quantum.cg import CG
from typing import Union, Tuple
from hyppy.sparse_dot import sparse_dot as spdotcy

def isvector(v: Union[csc_array, np.ndarray], ord: str="col") -> bool:
    """
    Checks if the given array is a vector.

    Parameters
    ----------
    v : csc_array or numpy.ndarray
        Array to be checked. Must be two-dimensional.
    ord : str
        Can be either "col" or "row".

    Returns
    -------
    bool
        True, if the array is a vector.
    """

    # Check whether the array is two-dimensional
    if len(v.shape) != 2:
        raise ValueError("Invalid shape for the input array.")

    # Find whether row or column is checked
    if ord == "col":
        i = 1
    elif ord == "row":
        i = 0
    else:
        raise ValueError(f"Invalid value for ord: {ord}")

    # Check whether the array is a vector
    if v.shape[i] == 1:
        return True
    else:
        return False

def norm_1(A: Union[csc_array, np.ndarray], ord: str='row') -> float:
    """
    Calculates the 1-norm of a matrix.

    Parameters
    ----------
    A : csc_array or numpy.ndarray
        Norm is calclated for this array.
    ord : String
        Either 'row' or 'col' according to which the 1-norm is calculated.

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
    norm_1 = abs(A).sum(axis).max()

    return norm_1

def expm_custom_dot(A: csc_array, zero_value: float=1e-24) -> csc_array:
    """
    Calculates the matrix exponential of a SciPy sparse CSC array using the
    scaling and squaring method with the Taylor series, which was shown to
    be the fastest method here:

    https://doi.org/10.1016/j.jmr.2010.12.004

    This function uses the custom dot product implementation, which is more
    memory-friendly and is parallelized.

    Parameters
    ----------
    A : csc_array
        Array to be exponentiated.
    zero_value : float
        Default: 1e-24. Value that is considered to be zero. Used to increase
        the sparsity of the end result as well as to estimate the convergence
        of the Taylor series.

    Returns
    -------
    expm_A : csc_array
        Matrix exponential of `A`.
    """

    # Calculate the norm of A
    norm_A = norm_1(A, ord='col')

    # If the norm of the matrix is too large, scale the matrix down
    if norm_A > 1:

        # Calculate scaling factor for the matrix
        scaling_count = int(math.ceil(math.log2(norm_A)))
        scaling_factor = 2 ** scaling_count

        # Scale the matrix down
        A = A / scaling_factor

        # Calculate the matrix exponential of the scaled matrix using Taylor series
        expm_A = expm_taylor_custom_dot(A, zero_value)

        # Scale the matrix exponential back up by multiplying with itself
        for _ in range(scaling_count):

            # Multiply the expm with itself
            expm_A = sparse_dot(expm_A, expm_A, zero_value)
    
    # If the norm of the matrix is small, continue normally
    else:

        # Calculate the matrix exponential using Taylor series
        expm_A = expm_taylor_custom_dot(A, zero_value)

    return expm_A

def expm_taylor_custom_dot(A: csc_array, zero_value: float=1e-24) -> csc_array:
    """
    Compute the matrix exponential using Taylor series. Function adapted from 
    old SciPy version.

    This function uses the custom dot product implementation, which is more
    memory-friendly and is parallelized.

    Parameters
    ----------
    A : csc_array
        Matrix (N, N) to be exponentiated.
    zero_value : float
        Default: 1e-24. Value that is considered to be zero. Used to increase
        the sparsity and to check the convergence of the series. 

    Returns
    -------
    eA : csc_array
        Matrix exponential of A.
    """

    # Increase sparsity of A
    increase_sparsity(A, zero_value)
    
    # Create unit matrix for the first term
    eA = eye_array(A.shape[0], A.shape[0], dtype=complex, format='csc')

    # Make a copy for the terms
    trm = eA.copy()

    # Calculate new term until its significance is tiny
    k=1
    cont = True
    while cont:

        # Get the next term
        trm = sparse_dot(trm, A / k, zero_value)

        # Sum to the existing term
        eA += trm

        # Increase counter
        k+=1

        # Continue if convergence criterion is not met
        cont = (trm.nnz != 0)

    return eA

def expm(A: Union[csc_array, np.ndarray], zero_value: float=1e-24) -> Union[csc_array, np.ndarray]:
    """
    Calculates the matrix exponential of a SciPy sparse or NumPy array using
    the scaling and squaring method with the Taylor series, which was shown
    to be the fastest method here:

    https://doi.org/10.1016/j.jmr.2010.12.004

    Parameters
    ----------
    A : csc_array or numpy.ndarray
        Array to be exponentiated.
    zero_value : float
        Default: 1e-24. Value that is considered to be zero. Used to increase
        the sparsity of the end result as well as to estimate the convergence
        of the Taylor series.

    Returns
    -------
    expm_A : csc_array or numpy.ndarray
        Matrix exponential of `A`.
    """

    # Calculate the norm of A
    norm_A = norm_1(A, ord='col')

    # If the norm of the matrix is too large, scale the matrix down
    if norm_A > 1:

        # Calculate scaling factor for the matrix
        scaling_count = int(math.ceil(math.log2(norm_A)))
        scaling_factor = 2 ** scaling_count

        # Scale the matrix down
        A = A / scaling_factor

        # Calculate the matrix exponential of the scaled matrix using Taylor series
        expm_A = expm_taylor(A, zero_value)

        # Scale the matrix exponential back up by multiplying with itself
        for _ in range(scaling_count):

            # Multiply the expm with itself
            expm_A = expm_A @ expm_A
            
            # Increase sparsity of the result if using sparse matrices
            if issparse(expm_A):
                increase_sparsity(expm_A, zero_value)
    
    # If the norm of the matrix is small, continue normally
    else:

        # Calculate the matrix exponential using Taylor series
        expm_A = expm_taylor(A, zero_value)

        # Increase sparsity of the result if using sparse matrices
        if issparse(expm_A):
            increase_sparsity(expm_A, zero_value)

    return expm_A

def expm_taylor(A: Union[csc_array, np.ndarray], zero_value: float=1e-24) -> Union[csc_array, np.ndarray]:
    """
    Compute the matrix exponential using Taylor series. Function adapted from 
    old SciPy version.

    Parameters
    ----------
    A : csc_array or numpy.ndarray
        Matrix (N, N) to be exponentiated.
    zero_value : float
        Default: 1e-24. Value that is considered to be zero. Used to increase
        the sparsity and to check the convergence of the series. 

    Returns
    -------
    eA : csc_array or numpy.ndarray
        Matrix exponential of A.
    """

    # Increase sparsity of A if using sparse matrices
    if issparse(A):
        increase_sparsity(A, zero_value)
    
    # Create unit matrix for the first term
    eA = eye_array(A.shape[0], A.shape[0], dtype=complex, format='csc')

    # Convert to NumPy if not using sparse matrices
    if not issparse(A):
        eA = eA.toarray()

    # Make a copy for the terms
    trm = eA.copy()

    # Calculate new term until its significance is tiny
    k=1
    cont = True
    while cont:

        # Get the next term
        trm = trm @ (A / k)

        # Increase sparsity of the next term if using sparse matrices
        if issparse(trm):
            increase_sparsity(trm, zero_value)

        # Sum to the existing term
        eA += trm

        # Increase counter
        k+=1

        # Continue if convergence criterion is not met
        if issparse(trm):
            cont = (trm.nnz != 0)
        else:
            cont = np.all(np.abs(trm) > zero_value)

    return eA

def increase_sparsity(A: csc_array, zero_value: float=1e-24):
    """
    Increases sparsity of the given input matrix by replacing small values
    with zeros. Modification happens inplace.

    Parameters
    ----------
    A : csc_array
    zero_value : float
        Default: 1e-24. Values less than zero_value are set to zero.
    """

    # Get values smaller than the threshold and make them zero
    nonzero_mask = np.abs(A.data) < zero_value
    A.data[nonzero_mask] = 0
    A.eliminate_zeros()

def sparse_to_bytes(A: csc_array) -> bytes:
    """
    Convert the given SciPy sparse array into byte representation.

    Parameters
    ----------
    A : csc_array
        Array to be converted into bytes.

    Returns
    -------
    A_bytes : bytes
        Input matrix in the byte representation.
    """
    
    # Initialize the BytesIO
    bytes_io = BytesIO()

    # Write the matrix A to bytes
    mmwrite(bytes_io, A)

    # Get the bytes
    A_bytes = bytes_io.getvalue()

    return A_bytes

def bytes_to_sparse(A_bytes: bytes) -> csc_array:
    """
    Convert the bytes back to SciPy sparse array.

    Parameters
    ----------
    A_bytes : bytes
        Byte representation of a Scipy sparse array.

    Returns
    -------
    A : csc_array
        Bytes converted into a SciPy sparse array.
    """

    # Initialize the bytesIO
    bytes_io = BytesIO(A_bytes)

    # Get the SciPy sparse array
    A = mmread(bytes_io)

    return A

def comm(A: Union[csc_array, np.ndarray], B: Union[csc_array, np.ndarray]) -> Union[csc_array, np.ndarray]:
    """
    Calculates the commutator [`A`, `B`] of two operators.

    Parameters
    ----------
    A : csc_array or numpy.ndarray
    B : csc_array or numpy.ndarray

    Returns
    -------
    C : csc_array or numpy.ndarray
        Commutator [`A`, `B`]
    """

    # Calculate the commutator
    C = A @ B - B @ A

    return C

def find_common_rows(A: np.ndarray, B: np.ndarray) -> Tuple[list, list]:
    """
    Compares two arrays, `A` and `B`, and finds the indices of the common rows.
    Each row must appear only once in the arrays.

    Parameters
    ----------
    A : numpy.ndarray
    B : numpy.ndarray

    Returns
    -------
    A_ind : list
        Indices that return the common elements from array `A`.
    B_ind : list
        Indices that return the common elements from array `B`.
    """

    # Make a dictionary of the rows of B
    B_dict = {tuple(row): idx for idx, row in enumerate(B)}

    # Make empty lists for the indices
    A_ind = []
    B_ind = []

    # Loop over A
    for idx_A, row in enumerate(A):

        # Check whether the row of A is in B
        if tuple(row) in B_dict:

            # Append the indices
            A_ind.append(idx_A)
            B_ind.append(B_dict[tuple(row)])

    return A_ind, B_ind

def auxiliary_matrix_expm(A: csc_array, B: csc_array, C: csc_array, t: float, zero_value: float=1e-24) -> csc_array:   
    """
    Calculates the matrix exponential of an auxiliary matrix. Used to compute
    the Redfield integral.

    From Goodwin and Kuprov (Eq. 3): https://doi.org/10.1063/1.4928978
    
    Parameters
    ----------
    A : csc_array
        Top-left of the auxiliary matrix.
    B : csc_array
        Top-right of the auxiliary matrix.
    C : csc_array
        Bottom-right of the auxiliary matrix.
    t : float
        The integration time.
    zero_value : float
        Default: 1e-24. Value that is considered to be zero when exponentiating the
        auxiliary matrix using Taylor series. Significantly impacts the performance.
        Try to find largest value that still returns correct values.
    
    Returns
    -------
    expm_aux : csc_array
        Matrix exponential of the auxiliary matrix.
    """

    # Construct the auxiliary matrix
    empty_array = csc_array(A.shape)
    aux = block_array([[A, B],
                       [empty_array, C]], format='csc')

    # Exponentiate the auxiliary matrix
    expm_aux = expm(aux*t, zero_value)

    return expm_aux

def angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Find the angle between two vectors in radians.

    Parameters
    ----------
    v1 : numpy.ndarray
        First vector.
    v2 : numpy.ndarray
        Second vector

    Returns
    -------
    theta : float
        Angle between the vectors (rad).
    """

    # Consider identical vectors separately
    if np.array_equal(v1, v2):
        theta = 0
    else:
        theta = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

    return theta

def decompose_matrix(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Decomposes a matrix into three parts:
        - isotropic part
        - antisymmetric part
        - symmetric traceless part

    Parameters
    ----------
    matrix : numpy.ndarray
        Matrix to be decomposed.

    Returns
    -------
    isotropic : numpy.ndarray
        Isotropic part of the input matrix.
    antisymmetric : numpy.ndarray
        Antisymmetric part of the input matrix.
    symmetric_traceless : numpy.ndarray
        Symmetric traceless part of the input matrix.
    """

    # Find the isotropic, antisymmetric and symmetric traceless parts
    isotropic = np.trace(matrix) * np.eye(matrix.shape[0]) / matrix.shape[0]
    antisymmetric = (matrix - matrix.T) / 2
    symmetric_traceless = (matrix + matrix.T) / 2 - isotropic
    
    return isotropic, antisymmetric, symmetric_traceless

def principal_axis_system(tensor: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Finds the principal axis system (PAS) of a Cartesian tensor
    and converts the tensor to the PAS.

    The PAS is defined by the axis system that diagonalizes
    the symmetric traceless part of the tensor.

    Order is (|largest|, |middle|, |smallest|) according to the eigenvalues.

    Parameters
    ----------
    tensor : np.ndarray
        Cartesian tensor.

    Returns
    -------
    eigenvalues : numpy.ndarray
        Eigenvalues of the PAS.
    eigenvectors : numpy.ndarray
        Two-dimensional array, where the rows contain the eigenvectors of the PAS.
    tensor_PAS : numpy.ndarray
        Tensor in the PAS. 
    """

    # Get the symmetric part of the tensor
    _, _, symmetric_traceless = decompose_matrix(tensor)

    # Diagonalize the symmetric traceless part
    eigenvalues, eigenvectors = np.linalg.eig(symmetric_traceless)

    # Sort the according to the eigenvalues
    idx = np.argsort(np.abs(eigenvalues))[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx].T

    # Write the tensor in the principal axis system
    tensor_PAS = eigenvectors @ tensor @ np.linalg.inv(eigenvectors)

    return eigenvalues, eigenvectors, tensor_PAS

def cartesian_tensor_to_spherical_tensor(C: np.ndarray) -> dict:
    """
    Converts a rank-2 Cartesian tensor to a spherical tensor.

    Uses the double outer product (DOP) convention from:
    Eq. 293-298 in Man: Cartesian and Spherical Tensors in NMR Hamiltonians
    https://doi.org/10.1002/cmr.a.21289

    Parameters
    ----------
    C : numpy.ndarray
        Rank-2 tensor in Cartesian coordinates

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
        (0, 0) : -1/math.sqrt(3) * (C_xx + C_yy + C_zz),
        (1, 0) : -1j/math.sqrt(2) * (C_xy - C_yx),
        (1, 1) : -1/2 * (C_zx - C_xz + 1j*(C_zy - C_yz)),
        (1,-1) : -1/2 * (C_zx - C_xz - 1j*(C_zy - C_yz)),
        (2, 0) :  1/math.sqrt(6) * (-C_xx + 2*C_zz - C_yy),
        (2, 1) : -1/2 * (C_xz + C_zx + 1j*(C_yz + C_zy)),
        (2,-1) :  1/2 * (C_xz + C_zx - 1j*(C_yz + C_zy)),
        (2, 2) :  1/2 * (C_xx - C_yy + 1j*(C_xy + C_yx)),
        (2,-2) :  1/2 * (C_xx - C_yy - 1j*(C_xy + C_yx))
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
    vector : numpy.ndarray
        Vector in the format [x, y, z].

    Returns
    -------
    spherical_tensor : dict
        Keys specify the rank and the projection (l, q), and the values
        are the components.   
    """

    # Build the spherical tensor
    spherical_tensor = {
        (1, 1) : -1/math.sqrt(2) * (vector[0] + 1j * vector[1]),
        (1, 0) : vector[2],
        (1,-1) : 1/math.sqrt(2) * (vector[0] - 1j * vector[1])
    }

    return spherical_tensor

@lru_cache(maxsize=32784)
def CG_coeff(j1: float, m1: float, j2: float, m2: float, j3: float, m3: float) -> float:
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

def sparse_dot(A: csc_array, B: csc_array, zero_value: float=1e-24) -> csc_array:
    """
    Custom sparse matrix multiplication, which saves memory usage by dropping
    values smaller than `zero_value` during the calculation. Matrices `A` and `B`
    must be SciPy CSC arrays. This function is implemented with Cython and it is
    parallelized with OpenMP.

    Parameters
    ----------
    A : csc_array
        First matrix in the multiplication.
    B : csc_array
        Second matrix in the multiplication.
    zero_value : float
        Default: 1e-24. Threshold under which the resulting matrix elements are
        considered as zero.

    Returns
    -------
    C : csc_array
        Result of matrix multiplication.
    """

    # Make sure that the data types are correct
    A.data = A.data.astype(np.complex128)
    A.indices = A.indices.astype(np.longlong)
    A.indptr = A.indptr.astype(np.longlong)
    B.data = B.data.astype(np.complex128)
    B.indices = B.indices.astype(np.longlong)
    B.indptr = B.indptr.astype(np.longlong)
    A_nrows = np.longlong(A.shape[0])
    B_ncols = np.longlong(B.shape[1])
    zero_value = np.float64(zero_value)

    # Perform the matrix multiplication using the compiled function
    C_data, C_indices, C_indptr = spdotcy(A.data, A.indices, A.indptr, A_nrows,
                                          B.data, B.indices, B.indptr, B_ncols,
                                          zero_value)

    # Construct the SciPy sparse array
    C = csc_array((C_data, C_indices, C_indptr), shape=(A_nrows, B_ncols))

    return C