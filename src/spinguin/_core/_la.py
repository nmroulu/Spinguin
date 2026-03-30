"""
Linear algebra utilities used throughout the spin-dynamics code base.

The module contains helper routines for sparse-memory sharing, matrix
exponentials, tensor manipulations, and small conversion utilities needed by
the NMR simulation workflow.
"""

import math
from functools import lru_cache
from io import BytesIO
from multiprocessing.shared_memory import SharedMemory

import numpy as np
from numpy.typing import ArrayLike
from scipy.io import mmread, mmwrite
from scipy.sparse import block_array, csc_array, eye_array, issparse
from scipy.spatial.transform import Rotation
from sympy.physics.quantum.cg import CG
from sympy.physics.wigner import wigner_d

from spinguin._core._hide_prints import HidePrints
from spinguin._core._intersect_indices import intersect_indices
from spinguin._core._sparse_dot import sparse_dot as _sparse_dot
from spinguin._core._status import status

DenseOrSparse = np.ndarray | csc_array
SharedSparseMetadata = dict[str, str | np.dtype | tuple[int, ...]]
SharedSparseHandles = tuple[SharedMemory, SharedMemory, SharedMemory]


def write_shared_sparse(
    A: csc_array,
) -> tuple[SharedSparseMetadata, SharedSparseHandles]:
    """
    Create a shared-memory representation of a CSC sparse array.

    Parameters
    ----------
    A : csc_array
        Sparse array to be stored in shared memory.

    Returns
    -------
    A_shared : dict
        Metadata for reconstructing the sparse array from shared memory.
    A_shm : tuple
        Shared-memory handles for the data, indices, and index-pointer arrays.
    """

    # Allocate shared memory for the three internal CSC buffers.
    A_data_shm = SharedMemory(create=True, size=A.data.nbytes)
    A_indices_shm = SharedMemory(create=True, size=A.indices.nbytes)
    A_indptr_shm = SharedMemory(create=True, size=A.indptr.nbytes)

    # Expose the shared buffers as NumPy arrays and copy the CSC data into
    # them.
    A_data_shared = np.ndarray(
        A.data.shape,
        dtype=A.data.dtype,
        buffer=A_data_shm.buf,
    )
    A_indices_shared = np.ndarray(
        A.indices.shape,
        dtype=A.indices.dtype,
        buffer=A_indices_shm.buf,
    )
    A_indptr_shared = np.ndarray(
        A.indptr.shape,
        dtype=A.indptr.dtype,
        buffer=A_indptr_shm.buf,
    )
    A_data_shared[:] = A.data[:]
    A_indices_shared[:] = A.indices[:]
    A_indptr_shared[:] = A.indptr[:]

    # Store the metadata required to rebuild the sparse array later.
    A_shared = {
        "A_data_shm_name": A_data_shm.name,
        "A_indices_shm_name": A_indices_shm.name,
        "A_indptr_shm_name": A_indptr_shm.name,
        "A_data_shape": A.data.shape,
        "A_indices_shape": A.indices.shape,
        "A_indptr_shape": A.indptr.shape,
        "A_data_dtype": A.data.dtype,
        "A_indices_dtype": A.indices.dtype,
        "A_indptr_dtype": A.indptr.dtype,
        "A_shape": A.shape,
    }

    # Return both the serialisable metadata and the live handles.
    A_shm = (A_data_shm, A_indices_shm, A_indptr_shm)
    return A_shared, A_shm


def read_shared_sparse(
    A_shared: SharedSparseMetadata,
) -> tuple[csc_array, SharedSparseHandles]:
    """
    Reconstruct a CSC sparse array from shared-memory metadata.

    Parameters
    ----------
    A_shared : dict
        Metadata describing the shared-memory layout of the sparse array.

    Returns
    -------
    A : csc_array
        Sparse array reconstructed from the shared buffers.
    A_shm : tuple
        Shared-memory handles for the data, indices, and index-pointer arrays.
    """

    # Unpack the shared-memory metadata.
    A_data_shm_name = A_shared["A_data_shm_name"]
    A_indices_shm_name = A_shared["A_indices_shm_name"]
    A_indptr_shm_name = A_shared["A_indptr_shm_name"]
    A_data_shape = A_shared["A_data_shape"]
    A_indices_shape = A_shared["A_indices_shape"]
    A_indptr_shape = A_shared["A_indptr_shape"]
    A_data_dtype = A_shared["A_data_dtype"]
    A_indices_dtype = A_shared["A_indices_dtype"]
    A_indptr_dtype = A_shared["A_indptr_dtype"]
    A_shape = A_shared["A_shape"]

    # Attach to the existing shared-memory blocks.
    A_data_shm = SharedMemory(name=A_data_shm_name)
    A_indices_shm = SharedMemory(name=A_indices_shm_name)
    A_indptr_shm = SharedMemory(name=A_indptr_shm_name)

    # View the shared buffers as the original NumPy arrays.
    A_data = np.ndarray(
        shape=A_data_shape,
        dtype=A_data_dtype,
        buffer=A_data_shm.buf,
    )
    A_indices = np.ndarray(
        shape=A_indices_shape,
        dtype=A_indices_dtype,
        buffer=A_indices_shm.buf,
    )
    A_indptr = np.ndarray(
        shape=A_indptr_shape,
        dtype=A_indptr_dtype,
        buffer=A_indptr_shm.buf,
    )

    # Rebuild the sparse CSC array without copying the shared data.
    A = csc_array((A_data, A_indices, A_indptr), shape=A_shape, copy=False)

    # Return the reconstructed matrix together with the live handles.
    A_shm = (A_data_shm, A_indices_shm, A_indptr_shm)
    return A, A_shm


def isvector(
    v: DenseOrSparse,
    ord: str="col",
) -> bool:
    """
    Determine whether a two-dimensional array is a row or column vector.

    Parameters
    ----------
    v : ndarray or csc_array
        Array to inspect. The input must be two-dimensional.
    ord : str, default="col"
        Vector orientation to test. Accepted values are "col" and "row".

    Returns
    -------
    bool
        True if the requested vector condition is satisfied.

    Raises
    ------
    ValueError
        Raised if the input is not two-dimensional or if `ord` is invalid.
    """

    # Require a matrix-shaped input so that row and column vectors can be
    # distinguished unambiguously.
    if len(v.shape) != 2:
        raise ValueError("Input array must be two-dimensional.")

    # Select the axis that must have unit length.
    if ord == "col":
        axis = 1
    elif ord == "row":
        axis = 0
    else:
        raise ValueError(f"Invalid value for ord: {ord}")

    # Test whether the requested dimension is unity.
    return v.shape[axis] == 1


def norm_1(
    A: DenseOrSparse,
    ord: str="row",
) -> float:
    """
    Calculate the row-wise or column-wise matrix 1-norm.

    Parameters
    ----------
    A : ndarray or csc_array
        Array for which the norm is evaluated.
    ord : str, default="row"
        Direction of the reduction. Accepted values are "row" and "col".

    Returns
    -------
    norm_1 : float
        Maximum absolute row sum or column sum of `A`.

    Raises
    ------
    ValueError
        Raised if `ord` is neither "row" nor "col".
    """

    # Select the reduction axis according to the requested orientation.
    if ord == "row":
        axis = 1
    elif ord == "col":
        axis = 0
    else:
        raise ValueError(f"Invalid value for ord: {ord}")

    # Sum the absolute values along the requested axis and keep the largest
    # entry.
    return np.abs(A).sum(axis).max()


def expm(
    A: DenseOrSparse,
    zero_value: float,
) -> DenseOrSparse:
    """
    Evaluate the matrix exponential with scaling, squaring, and Taylor series.

    The implementation follows the approach benchmarked in
    https://doi.org/10.1016/j.jmr.2010.12.004 and uses the custom sparse
    matrix product when possible.

    Parameters
    ----------
    A : ndarray or csc_array
        Matrix to exponentiate.
    zero_value : float
        Values smaller than this threshold are dropped during sparse-aware
        operations.

    Returns
    -------
    expm_A : ndarray or csc_array
        Matrix exponential of `A`.
    """

    status("Computing the matrix exponential using Taylor series with scaling and squaring...")

    # Estimate the matrix magnitude to decide whether scaling is required.
    norm_A = norm_1(A, ord="col")

    # Scale large matrices before the Taylor expansion and undo the scaling by
    # repeated squaring afterwards.
    if norm_A > 1:
        scaling_count = int(math.ceil(math.log2(norm_A)))
        scaling_factor = 2**scaling_count

        A = A / scaling_factor
        expm_A = expm_taylor(A, zero_value)

        status(f"Matrix squaring...")
        for step in range(scaling_count):
            status(f"Step {step + 1}...")
            expm_A = custom_dot(expm_A, expm_A, zero_value)
    else:
        expm_A = expm_taylor(A, zero_value)

    status("Matrix exponential computed.")
    return expm_A


def expm_taylor(
    A: DenseOrSparse,
    zero_value: float,
) -> DenseOrSparse:
    """
    Evaluate the matrix exponential by direct Taylor summation.

    The routine is adapted from an older SciPy implementation and uses the
    custom sparse dot product to reduce memory consumption for CSC matrices.

    Parameters
    ----------
    A : ndarray or csc_array
        Square matrix to exponentiate.
    zero_value : float
        Threshold used when pruning numerically negligible values.

    Returns
    -------
    eA : ndarray or csc_array
        Matrix exponential of `A`.
    """

    status("Taylor series...")

    # Remove entries that are already below the effective numerical threshold.
    eliminate_small(A, zero_value)

    # Start from the identity term of the Taylor series.
    eA = eye_array(A.shape[0], A.shape[0], dtype=complex, format="csc")
    trm = eA.copy()

    # Accumulate higher-order terms until the current contribution vanishes.
    k = 1
    cont = True
    while cont:
        status(f"Term {k}...")

        # Form the next Taylor term and add it to the running sum.
        trm = custom_dot(trm, A / k, zero_value)
        eA += trm
        k += 1

        # Continue until the newly generated term is exactly zero after the
        # thresholding step.
        if issparse(trm):
            cont = trm.nnz != 0
        else:
            cont = np.count_nonzero(trm) != 0

    status("Taylor series converged.")
    return eA


def eliminate_small(
    A: DenseOrSparse,
    zero_value: float,
) -> None:
    """
    Set numerically small matrix elements to zero in place.

    Parameters
    ----------
    A : ndarray or csc_array
        Array to modify in place.
    zero_value : float
        Absolute threshold below which values are replaced by zero.
    """

    # Apply the threshold either to sparse data values or to a dense array.
    if issparse(A):
        nonzero_mask = np.abs(A.data) < zero_value
        A.data[nonzero_mask] = 0
        A.eliminate_zeros()
    else:
        nonzero_mask = np.abs(A) < zero_value
        A[nonzero_mask] = 0


def sparse_to_bytes(
    A: csc_array,
) -> bytes:
    """
    Serialise a sparse array into a byte string.

    Parameters
    ----------
    A : csc_array
        Sparse matrix to serialise.

    Returns
    -------
    A_bytes : bytes
        Matrix Market byte representation of `A`.
    """

    # Write the sparse matrix into an in-memory byte buffer.
    bytes_io = BytesIO()
    mmwrite(bytes_io, A)

    # Extract the raw bytes from the buffer.
    A_bytes = bytes_io.getvalue()
    return A_bytes


def bytes_to_sparse(
    A_bytes: bytes,
) -> csc_array:
    """
    Reconstruct a sparse array from a byte string.

    Parameters
    ----------
    A_bytes : bytes
        Matrix Market byte representation of a sparse array.

    Returns
    -------
    A : csc_array
        Sparse array reconstructed from `A_bytes`.
    """

    # Wrap the raw bytes in an in-memory stream for SciPy.
    bytes_io = BytesIO(A_bytes)

    # Read the sparse matrix from the Matrix Market representation.
    A = mmread(bytes_io)
    return A


def comm(
    A: DenseOrSparse,
    B: DenseOrSparse,
) -> DenseOrSparse:
    """
    Calculate the commutator $[A, B] = AB - BA$.

    Parameters
    ----------
    A : ndarray or csc_array
        First operator.
    B : ndarray or csc_array
        Second operator.

    Returns
    -------
    C : ndarray or csc_array
        Commutator of the two operators.
    """

    # Evaluate the commutator directly from the matrix products.
    return A @ B - B @ A


def find_common_rows(
    A: np.ndarray,
    B: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Find matching lexicographically sorted rows in two arrays.

    Each row must appear at most once in each input array.

    Parameters
    ----------
    A : ndarray
        First array to compare.
    B : ndarray
        Second array to compare.

    Returns
    -------
    A_ind : ndarray
        Indices of common rows in `A`.
    B_ind : ndarray
        Indices of common rows in `B`.
    """

    # Handle the degenerate case of arrays with zero columns.
    if A.shape[1] == 0 and B.shape[1] == 0:
        A_ind = np.array([0])
        B_ind = np.array([0])
        return A_ind, B_ind

    # Record the row length and flatten the arrays for the low-level helper.
    row_length = np.longlong(A.shape[1])
    A = A.ravel()
    B = B.ravel()

    # Convert the data explicitly to the integer type expected by the helper.
    A = A.astype(np.longlong)
    B = B.astype(np.longlong)

    # Recover the matching row indices in both arrays.
    A_ind, B_ind = intersect_indices(A, B, row_length)
    return A_ind, B_ind


def auxiliary_matrix_expm(
    A: DenseOrSparse,
    B: DenseOrSparse,
    C: DenseOrSparse,
    t: float,
    zero_value: float,
) -> csc_array:
    """
    Exponentiate the auxiliary matrix used in the Redfield integral.

    The block-matrix construction follows Goodwin and Kuprov,
    https://doi.org/10.1063/1.4928978, Eq. 3.

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
        Threshold used when pruning negligible values during exponentiation.

    Returns
    -------
    expm_aux : ndarray or csc_array
        Matrix exponential of the auxiliary matrix.

    Raises
    ------
    ValueError
        Raised if the three blocks are not all sparse or all dense.
    """

    # Require a consistent matrix representation for all blocks.
    if not (issparse(A) == issparse(B) == issparse(C)):
        raise ValueError("All arrays A, B and C must be of same type.")

    # Construct the block auxiliary matrix using either sparse or dense
    # storage.
    if issparse(A):
        empty_array = csc_array(A.shape)
        aux = block_array([[A, B], [empty_array, C]], format="csc")
    else:
        empty_array = np.zeros(A.shape)
        aux = np.block([[A, B], [empty_array, C]])

    # Suppress status messages while exponentiating the enlarged auxiliary
    # matrix.
    with HidePrints():
        expm_aux = expm(aux * t, zero_value)

    return expm_aux


def angle_between_vectors(
    v1: np.ndarray,
    v2: np.ndarray,
) -> float:
    """
    Compute the angle between two vectors in radians.

    Parameters
    ----------
    v1 : ndarray
        First vector.
    v2 : ndarray
        Second vector.

    Returns
    -------
    theta : float
        Angle between `v1` and `v2` in radians.
    """

    # Return the exact zero angle for identical vectors.
    if np.array_equal(v1, v2):
        theta = 0
    else:
        theta = np.arccos(
            np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        )

    return theta


def rotation_matrix_to_align_axes(
    axes1: np.ndarray,
    axes2: np.ndarray,
) -> np.ndarray:
    """
    Compute the rotation matrix that aligns one set of axes with another.

    Parameters
    ----------
    axes1 : ndarray
        Initial coordinate system as a 3x3 matrix with basis vectors as rows.
    axes2 : ndarray
        Target coordinate system as a 3x3 matrix with basis vectors as rows.

    Returns
    -------
    R : ndarray
        Rotation matrix that maps `axes1` onto `axes2`.
    """

    # Determine the best-fit rotation between the two ordered sets of axes.
    rotation, _ = Rotation.align_vectors(axes2, axes1)
    return rotation.as_matrix()


def Wigner_D_matrix(
    rotation_matrix: np.ndarray,
    j: int,
) -> np.ndarray:
    """
    Compute the Wigner D-matrix associated with a rotation.

    Parameters
    ----------
    rotation_matrix : ndarray
        3x3 rotation matrix.
    j : int
        Angular-momentum quantum number.

    Returns
    -------
    D : ndarray
        Complex Wigner D-matrix of shape (2j + 1, 2j + 1).
    """

    # Convert the Cartesian rotation matrix to Euler angles in the ZYZ
    # convention expected by SymPy.
    rotation = Rotation.from_matrix(rotation_matrix)
    alpha, beta, gamma = rotation.as_euler("ZYZ", degrees=False)

    # Evaluate the symbolic Wigner D-matrix and cast it to a NumPy array.
    D = wigner_d(j, alpha, beta, gamma)
    return np.array(D).astype(np.complex128)


def decompose_matrix(
    matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Decompose a matrix into isotropic, antisymmetric, and traceless parts.

    Parameters
    ----------
    matrix : ndarray
        Matrix to decompose.

    Returns
    -------
    isotropic : ndarray
        Isotropic contribution to `matrix`.
    antisymmetric : ndarray
        Antisymmetric contribution to `matrix`.
    symmetric_traceless : ndarray
        Symmetric traceless contribution to `matrix`.
    """

    # Separate the trace contribution from the symmetric and antisymmetric
    # components.
    isotropic = np.trace(matrix) * np.eye(matrix.shape[0]) / matrix.shape[0]
    antisymmetric = (matrix - matrix.T) / 2
    symmetric_traceless = (matrix + matrix.T) / 2 - isotropic
    return isotropic, antisymmetric, symmetric_traceless


def principal_axis_system(
    tensor: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Transform a Cartesian tensor into its principal-axis system.

    The principal-axis system is defined by diagonalising the symmetric
    traceless part of the tensor. The eigenvalues are ordered by decreasing
    absolute value.

    Parameters
    ----------
    tensor : ndarray
        Cartesian tensor to transform.

    Returns
    -------
    eigenvalues : ndarray
        Eigenvalues in the principal-axis system.
    eigenvectors : ndarray
        Row-wise eigenvectors defining the principal axes.
    tensor_PAS : ndarray
        Tensor transformed into the principal-axis system.
    """

    # Extract the symmetric traceless contribution that defines the PAS.
    _, _, symmetric_traceless = decompose_matrix(tensor)

    # Diagonalise the defining tensor and sort the eigenpairs consistently.
    eigenvalues, eigenvectors = np.linalg.eig(symmetric_traceless)
    idx = np.argsort(np.abs(eigenvalues))[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx].T

    # Transform the full tensor into the principal-axis basis.
    tensor_PAS = eigenvectors @ tensor @ np.linalg.inv(eigenvectors)
    return eigenvalues, eigenvectors, tensor_PAS


def cartesian_tensor_to_spherical_tensor(
    C: np.ndarray,
) -> dict:
    """
    Convert a rank-2 Cartesian tensor to spherical-tensor components.

    The implementation follows the double outer-product convention from
    Eqs. 293--298 of Man, https://doi.org/10.1002/cmr.a.21289.

    Parameters
    ----------
    C : ndarray
        Rank-2 tensor in Cartesian coordinates.

    Returns
    -------
    spherical_tensor : dict
        Dictionary whose keys are `(l, q)` and whose values are the spherical
        tensor components.
    """

    # Read out the Cartesian tensor elements once for compact formulas.
    C_xx, C_xy, C_xz = C[0, :]
    C_yx, C_yy, C_yz = C[1, :]
    C_zx, C_zy, C_zz = C[2, :]

    # Assemble the spherical tensor components.
    spherical_tensor = {
        (0, 0): -1 / math.sqrt(3) * (C_xx + C_yy + C_zz),
        (1, 0): -1j / math.sqrt(2) * (C_xy - C_yx),
        (1, 1): -1 / 2 * (C_zx - C_xz + 1j * (C_zy - C_yz)),
        (1, -1): -1 / 2 * (C_zx - C_xz - 1j * (C_zy - C_yz)),
        (2, 0): 1 / math.sqrt(6) * (-C_xx + 2 * C_zz - C_yy),
        (2, 1): -1 / 2 * (C_xz + C_zx + 1j * (C_yz + C_zy)),
        (2, -1): 1 / 2 * (C_xz + C_zx - 1j * (C_yz + C_zy)),
        (2, 2): 1 / 2 * (C_xx - C_yy + 1j * (C_xy + C_yx)),
        (2, -2): 1 / 2 * (C_xx - C_yy - 1j * (C_xy + C_yx)),
    }
    return spherical_tensor


def vector_to_spherical_tensor(
    vector: np.ndarray,
) -> dict:
    """
    Convert a Cartesian vector to a rank-1 spherical tensor.

    The covariant-component convention follows Eq. 230 of Man,
    https://doi.org/10.1002/cmr.a.21289.

    Parameters
    ----------
    vector : ndarray
        Cartesian vector in the order [x, y, z].

    Returns
    -------
    spherical_tensor : dict
        Dictionary whose keys are `(l, q)` and whose values are the spherical
        tensor components.
    """

    # Assemble the spherical rank-1 components.
    spherical_tensor = {
        (1, 1): -1 / math.sqrt(2) * (vector[0] + 1j * vector[1]),
        (1, 0): vector[2],
        (1, -1): 1 / math.sqrt(2) * (vector[0] - 1j * vector[1]),
    }
    return spherical_tensor


@lru_cache(maxsize=32784)
def CG_coeff(
    j1: float,
    m1: float,
    j2: float,
    m2: float,
    j3: float,
    m3: float,
) -> float:
    """
    Compute a Clebsch-Gordan coefficient.

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

    # Evaluate the symbolic coefficient and return a plain floating-point
    # value.
    coeff = float(CG(j1, m1, j2, m2, j3, m3).doit())
    return coeff


def custom_dot(
    A: DenseOrSparse,
    B: DenseOrSparse,
    zero_value: float,
) -> csc_array:
    """
    Multiply two arrays using the sparse-optimised backend when possible.

    Values smaller than `zero_value` are discarded from the result. If either
    operand is dense, the function falls back to the regular matrix product.

    Parameters
    ----------
    A : ndarray or csc_array
        Left matrix in the multiplication.
    B : ndarray or csc_array
        Right matrix in the multiplication.
    zero_value : float
        Threshold below which matrix elements are treated as zero.

    Returns
    -------
    C : ndarray or csc_array
        Matrix product `A @ B`.

    Raises
    ------
    ValueError
        Raised if the inputs are neither both sparse nor at least one dense
        NumPy array.
    """

    # Fall back to dense multiplication whenever a dense operand is present.
    if isinstance(A, np.ndarray) or isinstance(B, np.ndarray):
        C = A @ B
        eliminate_small(C, zero_value)

    # Route sparse CSC products through the custom low-level implementation.
    elif issparse(A) and issparse(B):
        A = A.tocsc()
        B = B.tocsc()
        C = _sparse_dot(A, B, zero_value)

    # Reject unsupported input combinations explicitly.
    else:
        raise ValueError("Invalid input type for custom dot.")

    return C


def arraylike_to_tuple(
    A: ArrayLike,
) -> tuple:
    """
    Convert a zero- or one-dimensional array-like object to a Python tuple.

    Parameters
    ----------
    A : ArrayLike
        Object that can be converted to a NumPy array.

    Returns
    -------
    A : tuple
        Tuple representation of the input.

    Raises
    ------
    ValueError
        Raised if the input has more than one dimension.
    """

    # Convert the input to a NumPy array for dimensionality inspection.
    A = np.asarray(A)

    # Return a length-one tuple for scalars and a direct tuple conversion for
    # one-dimensional arrays.
    if A.ndim == 0:
        A = (A.item(),)
    elif A.ndim == 1:
        A = tuple(A)
    else:
        raise ValueError(
            f"Cannot convert {A.ndim}-dimensional array into tuple."
        )

    return A


def arraylike_to_array(
    A: ArrayLike,
) -> np.ndarray:
    """
    Convert an array-like object to a NumPy array with at least one dimension.

    Parameters
    ----------
    A : ArrayLike
        Object that can be converted to a NumPy array.

    Returns
    -------
    A : ndarray
        NumPy representation of the input with `ndim >= 1`.
    """

    # Convert the input and promote scalars to one-dimensional arrays.
    A = np.asarray(A)
    A = np.atleast_1d(A)
    return A


def expm_vec_taylor(
    A: DenseOrSparse,
    v: DenseOrSparse,
    zero_value: float,
) -> DenseOrSparse:
    """
    Apply the matrix exponential to a vector by Taylor summation.

    Parameters
    ----------
    A : ndarray or csc_array
        Square matrix of shape (N, N).
    v : ndarray or csc_array
        Column vector of shape (N, 1).
    zero_value : float
        Threshold used when testing convergence of the Taylor series.

    Returns
    -------
    eAv : ndarray or csc_array
        Result of `expm(A) @ v`.
    """

    # Start from the zeroth-order Taylor contribution.
    trm = v
    eAv = trm

    # Add higher-order terms until the thresholded term vanishes.
    k = 1
    cont = True
    while cont:
        trm = A @ (trm / k)
        eliminate_small(trm, zero_value)
        eAv = eAv + trm
        k += 1

        if issparse(trm):
            cont = trm.nnz != 0
        else:
            cont = np.count_nonzero(trm) != 0

    return eAv


def expm_vec(
    A: DenseOrSparse,
    v: DenseOrSparse,
    zero_value: float,
) -> DenseOrSparse:
    """
    Apply the matrix exponential to a vector with matrix scaling.

    Parameters
    ----------
    A : ndarray or csc_array
        Square matrix of shape (N, N).
    v : ndarray or csc_array
        Column vector of shape (N, 1).
    zero_value : float
        Threshold used when testing convergence of the Taylor series.

    Returns
    -------
    eAv : ndarray or csc_array
        Result of `expm(A) @ v`.
    """

    status("Calculating the action of matrix exponential on a vector...")

    # Determine the matrix scaling factor from the column-wise 1-norm.
    norm_A = norm_1(A, ord="col")
    scaling_A = int(math.ceil(norm_A))

    # Scale the matrix before repeated Taylor applications.
    status(f"Scaling the matrix by {scaling_A}.")
    A = A / scaling_A

    # Scale the convergence threshold relative to the largest vector element.
    scaling_zv = np.abs(v).max()
    status(f"Scaling the zero-value by {scaling_zv}.")
    zero_value = zero_value / scaling_zv

    # Repeatedly apply the Taylor propagator for the scaled matrix.
    eAv = v
    for step in range(scaling_A):
        status(f"Calculating expm(A)*vec. Step {step + 1} of {scaling_A}.")
        eAv = expm_vec_taylor(A, eAv, zero_value)

    return eAv


def clear_cache_CG_coeff() -> None:
    """
    Clear the memoisation cache used by `CG_coeff`.
    """

    # Reset the cached symbolic Clebsch-Gordan evaluations.
    CG_coeff.cache_clear()