"""
data_io.py

This module contains functions that are used to read data and convert that to suitable
formats.
"""

# Imports
import numpy as np

def read_array(file_path:str, data_type:type) -> np.ndarray:
    """
    Read a .txt file where the values are written in a space-separated format.

    Parameters
    ----------
    file_path : str
    data_type : type
        Data type to be read. For example: float or str

    Returns
    -------
    value_array : numpy.ndarray
    """

    # Open the file
    with open(file_path, 'r') as file:

        # Get the isotopes
        value_array = np.loadtxt(file, delimiter=None, dtype=data_type)

    return value_array

def read_xyz(file_path:str) -> np.ndarray:
    """
    Read a .xyz file where the first line contains the number of atoms and the second line contains
    the comment line. The following lines contain the atom symbol and the coordinates in Cartesian
    coordinates.

    Parameters
    ----------
    file_path : str
    
    Returns
    -------
    xyz : numpy.ndarray

    Returns the atom symbols and the coordinates as NumPy arrays.
    """

    # Open the file
    with open(file_path, 'r') as file:

        # Initialize a list for the xyz coordinates
        xyz = []

        # Read the number of atoms and skip the comment line
        n_atoms = int(file.readline())
        file.readline()

        # Get the coordinates for each atom
        for _ in range(n_atoms):

            # Read only the coordinates
            xyz.append(file.readline().split()[1:])

    # Convert the list to NumPy array
    xyz = np.array(xyz, dtype=float)

    return xyz

def read_tensors(file_path:str) -> np.ndarray:
    """
    Reads a file with Cartesian interaction tensors (from quantum chemistry calculations)
    for each spin or spin pair.
    
    The file should have the following format:
        - The first column is the index of the spin.
        - The following columns are the components of a 3x3 tensor.
    This is repeated for each spin.

    Parameters
    ----------
    file_path : str

    Returns
    -------
    tensors : numpy.ndarray
    """

    # Initialize the lists and the current index
    tensors = []
    matrix_rows = []
    current_index = None
    
    # Open the file
    with open(file_path, 'r') as file:

        # Process each line
        for line in file:

            # Process the index lines differently
            if line.strip().split()[0].isdigit() and len(line.strip().split()) == 4:
                if current_index is not None:
                    tensors.append(np.array(matrix_rows, dtype=float))
                current_index = int(line.strip().split()[0])
                matrix_rows = [list(map(float, line.strip().split()[1:]))]
            else:
                matrix_rows.append(list(map(float, line.strip().split())))
        
        # Append the last tensor
        if current_index is not None:
            tensors.append(np.array(matrix_rows, dtype=float))
    
    # Convert to NumPy
    tensors = np.array(tensors, dtype=float)

    return tensors