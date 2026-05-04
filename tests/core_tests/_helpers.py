"""
Helper functions for writing the unit tests.
"""

import os

import numpy as np
import spinguin as sg
from spinguin._core import _la

def build_spin_system(
    isotopes: list[str],
    max_spin_order: int
) -> sg.SpinSystem:
    """
    Create a spin system and build its basis set.

    Parameters
    ----------
    isotopes : list of str
        Isotope labels of the spin system.
    max_spin_order : int
        Maximum spin order used to build the basis set.

    Returns
    -------
    SpinSystem
        Spin system with a built basis set.
    """

    # Create the spin system used in the test.
    spin_system = sg.SpinSystem(isotopes)

    # Build the basis set with the requested truncation level.
    spin_system.basis.max_spin_order = max_spin_order
    spin_system.basis.build()

    return spin_system

def test_data_path(filename: str) -> str:
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

    # Locate the shared directory that stores the test input files.
    test_data_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "test_data",
    )

    return os.path.join(test_data_dir, filename)

def spherical_tensor(l: int, q: int) -> np.ndarray:
    """
    Construct a spherical tensor in the Cartesian basis for testing.

    The tensor is obtained by combining covariant spherical basis vectors.

    Recipe described in Man: Cartesian and Spherical Tensors in NMR Hamiltonians
    https://doi.org/10.1002/cmr.a.21289
    """

    # Initialise the Cartesian tensor.
    t_lq = np.zeros((3, 3), dtype=complex)

    # Couple the spherical basis vectors to rank `l` and projection `q`.
    for q1 in range(-1, 2):
        for q2 in range(-1, 2):
            t_lq += _la.CG_coeff(1, q1, 1, q2, l, q) * np.outer(
                spherical_vector(1, q1),
                spherical_vector(1, q2)
            )

    return t_lq

def spherical_vector(l: int, q: int) -> np.ndarray:
    """
    Construct a covariant spherical vector in the Cartesian basis.
    """

    # Define the Cartesian basis vectors.
    e_x = np.array([1, 0, 0])
    e_y = np.array([0, 1, 0])
    e_z = np.array([0, 0, 1])

    # Construct the requested spherical basis vector.
    if l == 1 and q == 1:
        v = -1/np.sqrt(2) * (e_x + 1j*e_y)
    elif l == 1 and q == 0:
        v = e_z
    elif l == 1 and q == -1:
        v = 1/np.sqrt(2) * (e_x - 1j*e_y)

    return v