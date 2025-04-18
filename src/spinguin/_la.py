"""
la.py

Provides various linear algebra tools required for spin dynamics simulations.
"""

# Imports
import math
import numpy as np
from scipy.sparse import eye_array, csc_array, block_array, issparse
from scipy.io import mmwrite, mmread
from scipy.signal import find_peaks
from io import BytesIO
from functools import lru_cache
from sympy.physics.quantum.cg import CG
from typing import Tuple
from spinguin.sparse_dot import sparse_dot as spdotcy

def isvector(v: csc_array | np.ndarray, ord: str = "col") -> bool:
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
    A : csc_array or numpy.ndarray
        Array for which the norm is calculated.
    ord : str
        Either 'row' or 'col', specifying the direction for the 1-norm calculation.

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

def expm_custom_dot(A: csc_array, zero_value: float = 1e-24, disable_output: bool = False) -> csc_array:
    """
    Calculates the matrix exponential of a SciPy sparse CSC array using the
    scaling and squaring method with the Taylor series, shown to be the fastest
    method in:

    https://doi.org/10.1016/j.jmr.2010.12.004

    This function uses a custom dot product implementation, which is more
    memory-efficient and parallelized.

    Parameters
    ----------
    A : csc_array
        Array to be exponentiated.
    zero_value : float
        Default: 1e-24. Values below this threshold are considered zero. Used to
        increase the sparsity of the result and estimate the convergence of the
        Taylor series.
    disable_output : bool
        Default: False. If set to True, printing to the console will be disabled.

    Returns
    -------
    expm_A : csc_array
        Matrix exponential of `A`.
    """

    if not disable_output:
        print("Calculating the matrix exponential...")

    # Calculate the norm of A
    norm_A = norm_1(A, ord='col')

    # If the norm of the matrix is too large, scale the matrix down
    if norm_A > 1:

        # Calculate the scaling factor for the matrix
        scaling_count = int(math.ceil(math.log2(norm_A)))
        scaling_factor = 2 ** scaling_count

        if not disable_output:
            print(f"Scaling the matrix down by {scaling_factor}.")

        # Scale the matrix down
        A = A / scaling_factor

        # Calculate the matrix exponential of the scaled matrix using the Taylor series
        expm_A = expm_taylor_custom_dot(A, zero_value, disable_output)

        # Scale the matrix exponential back up by repeated squaring
        for i in range(scaling_count):
            if not disable_output:
                print(f"Squaring the matrix. Step {i+1} of {scaling_count}.")
            expm_A = sparse_dot(expm_A, expm_A, zero_value)
    
    # If the norm of the matrix is small, proceed without scaling
    else:
        expm_A = expm_taylor_custom_dot(A, zero_value, disable_output)

    if not disable_output:
        print("Matrix exponential completed.")

    return expm_A

def expm_taylor_custom_dot(A: csc_array, zero_value: float=1e-24, disable_output: bool=False) -> csc_array:
    """
    Computes the matrix exponential using the Taylor series. This function is 
    adapted from an older SciPy version.

    It uses a custom dot product implementation, which is more memory-efficient 
    and parallelized.

    Parameters
    ----------
    A : csc_array
        Matrix (N, N) to be exponentiated.
    zero_value : float
        Default: 1e-24. Values below this threshold are considered zero. Used to 
        increase sparsity and check the convergence of the series.
    disable_output : bool
        Default: False. If set to True, printing to the console will be disabled.

    Returns
    -------
    eA : csc_array
        Matrix exponential of A.
    """

    if not disable_output:
        print("Calculating the matrix exponential using Taylor series.")

    # Increase sparsity of A
    increase_sparsity(A, zero_value)
    
    # Create a unit matrix for the first term
    eA = eye_array(A.shape[0], A.shape[0], dtype=complex, format='csc')

    # Make a copy for the terms
    trm = eA.copy()

    # Calculate new terms until their significance becomes negligible
    k = 1
    cont = True
    while cont:

        if not disable_output:
            print(f"Taylor series term: {k}")

        # Get the next term
        trm = sparse_dot(trm, A / k, zero_value)

        # Add the term to the result
        eA += trm

        # Increment the counter
        k += 1

        # Continue if the convergence criterion is not met
        cont = (trm.nnz != 0)

    if not disable_output:
        print("Taylor series converged.")

    return eA

def expm(A: csc_array | np.ndarray, zero_value: float=1e-24, disable_output: bool=False) -> csc_array | np.ndarray:
    """
    Calculates the matrix exponential of a SciPy sparse or NumPy array using 
    the scaling and squaring method with the Taylor series. This method was 
    shown to be the fastest in:

    https://doi.org/10.1016/j.jmr.2010.12.004

    Parameters
    ----------
    A : csc_array or numpy.ndarray
        Array to be exponentiated.
    zero_value : float
        Default: 1e-24. Values below this threshold are considered zero. Used to 
        increase sparsity of the result and estimate the convergence of the 
        Taylor series.
    disable_output : bool
        Default: False. If set to True, printing to the console will be disabled.

    Returns
    -------
    expm_A : csc_array or numpy.ndarray
        Matrix exponential of `A`.
    """

    if not disable_output:
        print("Calculating the matrix exponential...")

    # Calculate the norm of A
    norm_A = norm_1(A, ord='col')

    # If the norm of the matrix is too large, scale the matrix down
    if norm_A > 1:

        # Calculate the scaling factor for the matrix
        scaling_count = int(math.ceil(math.log2(norm_A)))
        scaling_factor = 2 ** scaling_count

        if not disable_output:
            print(f"Scaling the matrix down by {scaling_factor}.")

        # Scale the matrix down
        A = A / scaling_factor

        # Calculate the matrix exponential of the scaled matrix using the Taylor series
        expm_A = expm_taylor(A, zero_value, disable_output)

        # Scale the matrix exponential back up by repeated squaring
        for i in range(scaling_count):

            if not disable_output:
                print(f"Squaring the matrix. Step {i+1} of {scaling_count}.")

            # Multiply the matrix exponential with itself
            expm_A = expm_A @ expm_A
            
            # Increase sparsity of the result if using sparse matrices
            if issparse(expm_A):
                increase_sparsity(expm_A, zero_value)
    
    # If the norm of the matrix is small, proceed without scaling
    else:

        # Calculate the matrix exponential using the Taylor series
        expm_A = expm_taylor(A, zero_value, disable_output)

        # Increase sparsity of the result if using sparse matrices
        if issparse(expm_A):
            increase_sparsity(expm_A, zero_value)

    if not disable_output:
        print("Matrix exponential completed.")

    return expm_A

def expm_taylor(A: csc_array | np.ndarray, zero_value: float=1e-24, disable_output: bool=False) -> csc_array | np.ndarray:
    """
    Computes the matrix exponential using the Taylor series. This function is 
    adapted from an older SciPy version.

    Parameters
    ----------
    A : csc_array or numpy.ndarray
        Matrix (N, N) to be exponentiated.
    zero_value : float
        Default: 1e-24. Values below this threshold are considered zero. Used to 
        increase sparsity and check the convergence of the series.
    disable_output : bool
        Default: False. If set to True, printing to the console will be disabled.

    Returns
    -------
    eA : csc_array or numpy.ndarray
        Matrix exponential of A.
    """

    if not disable_output:
        print("Calculating the matrix exponential using Taylor series.")

    # Increase sparsity of A if using sparse matrices
    if issparse(A):
        increase_sparsity(A, zero_value)
    
    # Create a unit matrix for the first term
    eA = eye_array(A.shape[0], A.shape[0], dtype=complex, format='csc')

    # Convert to NumPy if not using sparse matrices
    if not issparse(A):
        eA = eA.toarray()

    # Make a copy for the terms
    trm = eA.copy()

    # Calculate new terms until their significance becomes negligible
    k = 1
    cont = True
    while cont:

        if not disable_output:
            print(f"Taylor series term: {k}")

        # Get the next term
        trm = trm @ (A / k)

        # Increase sparsity of the next term if using sparse matrices
        if issparse(trm):
            increase_sparsity(trm, zero_value)

        # Add the term to the result
        eA += trm

        # Increment the counter
        k += 1

        # Continue if the convergence criterion is not met
        if issparse(trm):
            cont = (trm.nnz != 0)
        else:
            cont = np.any(np.abs(trm) > zero_value)

    if not disable_output:
        print("Taylor series converged.")

    return eA

def increase_sparsity(A: csc_array, zero_value: float=1e-24):
    """
    Increases the sparsity of the given input matrix by replacing small values
    with zeros. Modification happens in-place.

    Parameters
    ----------
    A : csc_array
        Sparse matrix to be modified.
    zero_value : float
        Default: 1e-24. Values smaller than this threshold are set to zero.
    """

    # Identify values smaller than the threshold and set them to zero
    nonzero_mask = np.abs(A.data) < zero_value
    A.data[nonzero_mask] = 0
    A.eliminate_zeros()

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

def comm(A: csc_array | np.ndarray, B: csc_array | np.ndarray) -> csc_array | np.ndarray:
    """
    Calculates the commutator [A, B] of two operators.

    Parameters
    ----------
    A : csc_array or numpy.ndarray
        First operator.
    B : csc_array or numpy.ndarray
        Second operator.

    Returns
    -------
    C : csc_array or numpy.ndarray
        Commutator [A, B].
    """

    # Compute the commutator
    C = A @ B - B @ A

    return C

def find_common_rows(A: np.ndarray, B: np.ndarray) -> Tuple[list, list]:
    """
    Identifies the indices of common rows between two arrays, `A` and `B`.
    Each row must appear only once in the arrays.

    Parameters
    ----------
    A : numpy.ndarray
        First array to compare.
    B : numpy.ndarray
        Second array to compare.

    Returns
    -------
    A_ind : list
        Indices of the common rows in array `A`.
    B_ind : list
        Indices of the common rows in array `B`.
    """

    # Create a dictionary of the rows of B
    B_dict = {tuple(row): idx for idx, row in enumerate(B)}

    # Initialize lists for the indices
    A_ind = []
    B_ind = []

    # Iterate over rows of A
    for idx_A, row in enumerate(A):

        # Check if the row of A exists in B
        if tuple(row) in B_dict:

            # Append the indices
            A_ind.append(idx_A)
            B_ind.append(B_dict[tuple(row)])

    return A_ind, B_ind

def auxiliary_matrix_expm(A: csc_array, B: csc_array, C: csc_array, t: float, zero_value: float=1e-24) -> csc_array:   
    """
    Computes the matrix exponential of an auxiliary matrix. This is used to 
    calculate the Redfield integral.

    Based on Goodwin and Kuprov (Eq. 3): https://doi.org/10.1063/1.4928978
    
    Parameters
    ----------
    A : csc_array
        Top-left block of the auxiliary matrix.
    B : csc_array
        Top-right block of the auxiliary matrix.
    C : csc_array
        Bottom-right block of the auxiliary matrix.
    t : float
        Integration time.
    zero_value : float
        Default: 1e-24. Threshold below which values are considered zero when 
        exponentiating the auxiliary matrix using the Taylor series. This 
        significantly impacts performance. Use the largest value that still 
        provides correct results.
    
    Returns
    -------
    expm_aux : csc_array
        Matrix exponential of the auxiliary matrix.
    """

    # Construct the auxiliary matrix
    empty_array = csc_array(A.shape)
    aux = block_array([[A, B],
                       [empty_array, C]], format='csc')

    # Compute the matrix exponential of the auxiliary matrix
    expm_aux = expm(aux * t, zero_value, disable_output=True)

    return expm_aux

def angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Computes the angle between two vectors in radians.

    Parameters
    ----------
    v1 : numpy.ndarray
        First vector.
    v2 : numpy.ndarray
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
        theta = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

    return theta

def decompose_matrix(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Decomposes a matrix into three components:
        - Isotropic part.
        - Antisymmetric part.
        - Symmetric traceless part.

    Parameters
    ----------
    matrix : numpy.ndarray
        Matrix to decompose.

    Returns
    -------
    isotropic : numpy.ndarray
        Isotropic part of the input matrix.
    antisymmetric : numpy.ndarray
        Antisymmetric part of the input matrix.
    symmetric_traceless : numpy.ndarray
        Symmetric traceless part of the input matrix.
    """

    # Compute the isotropic, antisymmetric, and symmetric traceless parts
    isotropic = np.trace(matrix) * np.eye(matrix.shape[0]) / matrix.shape[0]
    antisymmetric = (matrix - matrix.T) / 2
    symmetric_traceless = (matrix + matrix.T) / 2 - isotropic
    
    return isotropic, antisymmetric, symmetric_traceless

def principal_axis_system(tensor: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Determines the principal axis system (PAS) of a Cartesian tensor
    and transforms the tensor into the PAS.

    The PAS is defined as the coordinate system that diagonalizes
    the symmetric traceless part of the tensor.

    The eigenvalues are ordered as (|largest|, |middle|, |smallest|).

    Parameters
    ----------
    tensor : np.ndarray
        Cartesian tensor to transform.

    Returns
    -------
    eigenvalues : numpy.ndarray
        Eigenvalues of the tensor in the PAS.
    eigenvectors : numpy.ndarray
        Two-dimensional array where rows contain the eigenvectors of the PAS.
    tensor_PAS : numpy.ndarray
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
    C : numpy.ndarray
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
        (1, 1): -1 / math.sqrt(2) * (vector[0] + 1j * vector[1]),
        (1, 0): vector[2],
        (1, -1): 1 / math.sqrt(2) * (vector[0] - 1j * vector[1])
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

def sparse_dot(A: csc_array, B: csc_array, zero_value: float = 1e-24) -> csc_array:
    """
    Custom sparse matrix multiplication, which saves memory usage by dropping
    values smaller than `zero_value` during the calculation. Matrices `A` and `B`
    must be SciPy CSC arrays. This function is implemented with Cython and is
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

    # Ensure that the data types are correct
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

def fourier_transform(signal: np.ndarray, dt: float, normalize: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes the Fourier transform of a given time-domain signal and returns 
    the corresponding frequency-domain representation. The Fourier transform 
    can be normalized to ensure consistent peak intensities regardless of the time step.

    Parameters
    ----------
    signal : numpy.ndarray
        Input signal in the time domain.
    dt : float
        Time step between consecutive samples in the signal.
    normalize : bool
        Whether to normalize the Fourier transform. Default is True.

    Returns
    -------
    freqs : numpy.ndarray
        Frequencies corresponding to the Fourier-transformed signal.
    fft_signal : numpy.ndarray
        Fourier-transformed signal in the frequency domain (normalized if specified).
    """
    # Compute the frequencies
    freqs = np.fft.fftfreq(len(signal), dt)

    # Compute the Fourier transform
    fft_signal = np.fft.fft(signal)

    # Normalize the Fourier transform if specified
    if normalize:
        fft_signal = fft_signal * dt

    # Apply frequency shifting
    freqs = np.fft.fftshift(freqs)
    fft_signal = np.fft.fftshift(fft_signal)

    return freqs, fft_signal

def spectrum(signal: np.ndarray, dt: float, normalize: bool = True, part: str = "real") -> Tuple[np.ndarray, np.ndarray]:
    """
    A wrapper function for the Fourier transform. Computes the Fourier transform
    and returns the frequency and spectrum (either the real or imaginary part of 
    the Fourier transform).

    Parameters
    ----------
    signal : numpy.ndarray
        Input signal in the time domain.
    dt : float
        Time step between consecutive samples in the signal.
    normalize : bool
        Whether to normalize the Fourier transform. Default is True.
    part : str
        Specifies which part of the Fourier transform to return. Can be "real" 
        or "imag". Default is "real".

    Returns
    -------
    freqs : numpy.ndarray
        Frequencies corresponding to the Fourier-transformed signal.
    spectrum : numpy.ndarray
        Specified part (real or imaginary) of the Fourier-transformed signal 
        in the frequency domain.
    """
    # Compute the Fourier transform
    freqs, fft_signal = fourier_transform(signal, dt, normalize=normalize)

    # Get the specified part of the Fourier transform
    if part == "real":
        spectrum = np.real(fft_signal)
    elif part == "imag":
        spectrum = np.imag(fft_signal)
    else:
        raise ValueError("Invalid value for 'part'. Must be 'real' or 'imag'.")

    return freqs, spectrum

def frequency_to_chemical_shift(frequency: float | np.ndarray, 
                                reference_frequency: float,
                                spectrometer_frequency: float) -> float | np.ndarray:
    """
    Converts a frequency (or an array of frequencies, e.g., a frequency axis) to a
    chemical shift value based on the reference frequency and the spectrometer
    frequency.

    Parameters
    ----------
    frequency : float or numpy.ndarray
        Frequency (or array of frequencies) to convert [in Hz].
    reference_frequency : float
        Reference frequency for the conversion [in Hz].
    spectrometer_frequency : float
        Spectrometer frequency for the conversion [in Hz].

    Returns
    -------
    chemical_shift : float or numpy.ndarray
        Converted chemical shift value (or array of values).
    """
    return (frequency - reference_frequency) / spectrometer_frequency * 1e6