"""
Data-input helpers for plain-text arrays and Cartesian tensor data.

This module provides helper functions for reading plain-text arrays, XYZ
coordinates, and Cartesian interaction tensors from files.
"""

import numpy as np


def read_array(file_path: str, data_type: type) -> np.ndarray:
    """
    Read a whitespace-separated text file into a NumPy array.

    Parameters
    ----------
    file_path : str
        Path to the file to be read.
    data_type : type
        Data type of the values to be read, for example `float` or `str`.

    Returns
    -------
    value_array : ndarray
        Array containing the values read from the file.
    """

    # Open the input file and pass it to NumPy.
    with open(file_path, 'r', encoding='utf-8') as file:

        # Read the whitespace-separated values into an array.
        value_array = np.loadtxt(file, delimiter=None, dtype=data_type)

    return value_array


def read_xyz(file_path: str) -> np.ndarray:
    """
    Read Cartesian coordinates from an XYZ file.

    The first line contains the number of atoms, the second line contains a
    comment, and the remaining lines contain atomic symbols followed by the
    Cartesian coordinates.

    Parameters
    ----------
    file_path : str
        Path to the XYZ file to be read.

    Returns
    -------
    xyz : ndarray
        Array of Cartesian coordinates with shape `(n_atoms, 3)`.
    """

    # Open the XYZ file for line-by-line parsing.
    with open(file_path, 'r', encoding='utf-8') as file:

        # Collect the Cartesian coordinates in a temporary list.
        xyz = []

        # Read the number of atoms and skip the comment line.
        n_atoms = int(file.readline())
        file.readline()

        # Extract the three Cartesian coordinates for each atom.
        for _ in range(n_atoms):

            # Ignore the atomic symbol and store only the coordinates.
            xyz.append(file.readline().split()[1:])

    # Convert the collected coordinates to a floating-point array.
    xyz = np.array(xyz, dtype=float)

    return xyz


def read_tensors(file_path: str) -> np.ndarray:
    """
    Read Cartesian interaction tensors from a text file.

    The file is assumed to contain one 3x3 tensor for each spin or spin pair.
    Each tensor starts with a line whose first entry is an integer index,
    followed by the first tensor row. The next two lines contain the remaining
    tensor rows.

    The first column identifies the spin entry, and the remaining columns
    contain the tensor components.

    Example:
    ```
    1 0.1 0.2 0.3
      0.4 0.5 0.6
      0.7 0.8 0.9
    2 0.2 0.3 0.4
      0.5 0.6 0.7
      0.8 0.9 1.0
    ```

    Parameters
    ----------
    file_path : str
        Path to the file containing the tensors.

    Returns
    -------
    tensors : ndarray
        Array of Cartesian tensors with shape `(n_tensors, 3, 3)`.
    """

    # Initialise the tensor collection and the current tensor buffer.
    tensors = []
    matrix_rows = []
    current_index = None

    # Open the tensor file for line-by-line parsing.
    with open(file_path, 'r', encoding='utf-8') as file:

        # Process each line and group rows into individual tensors.
        for line in file:
            values = line.strip().split()

            # Skip empty lines between tensor blocks if they are present.
            if not values:
                continue

            # Detect the first row of a new tensor block.
            if values[0].isdigit() and len(values) == 4:

                # Store the previous tensor before starting a new one.
                if current_index is not None:
                    tensors.append(np.array(matrix_rows, dtype=float))

                # Record the current tensor index and its first row.
                current_index = int(values[0])
                matrix_rows = [list(map(float, values[1:]))]
            else:

                # Append a continuation row to the current tensor.
                matrix_rows.append(list(map(float, values)))

        # Append the final tensor after the loop ends.
        if current_index is not None:
            tensors.append(np.array(matrix_rows, dtype=float))

    # Convert the tensor list to a single floating-point array.
    tensors = np.array(tensors, dtype=float)

    return tensors