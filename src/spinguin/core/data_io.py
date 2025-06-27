"""
This module contains functions for reading data from files and converting it
into suitable formats.
"""

# Imports
import numpy as np

def read_array(file_path: str, data_type: type) -> np.ndarray:
    """
    Reads a .txt file where values are stored in a space-separated format and
    converts that into a NumPy array.

    Parameters
    ----------
    file_path : str
        Path to the file to be read.
    data_type : type
        Data type of the values to be read (e.g., float or str).

    Returns
    -------
    value_array : ndarray
        A NumPy array containing the values read from the file.
    """

    # Open the file
    with open(file_path, 'r') as file:

        # Read the values into a NumPy array
        value_array = np.loadtxt(file, delimiter=None, dtype=data_type)

    return value_array

def read_xyz(file_path: str) -> np.ndarray:
    """
    Reads a .xyz file where the first line specifies the number of atoms, the second line 
    contains a comment, and the subsequent lines contain the atom symbols and Cartesian coordinates.

    Parameters
    ----------
    file_path : str
        Path to the .xyz file to be read.

    Returns
    -------
    xyz : ndarray
        A NumPy array containing the atom symbols and Cartesian coordinates.
    """

    # Open the file
    with open(file_path, 'r') as file:

        # Initialize a list for the xyz coordinates
        xyz = []

        # Read the number of atoms and skip the comment line
        n_atoms = int(file.readline())
        file.readline()

        # Extract the coordinates for each atom
        for _ in range(n_atoms):

            # Read only the coordinates
            xyz.append(file.readline().split()[1:])

    # Convert the list to a NumPy array
    xyz = np.array(xyz, dtype=float)

    return xyz

def read_tensors(file_path: str) -> np.ndarray:
    """
    Reads a file containing Cartesian interaction tensors (from quantum chemistry calculations)
    for each spin or spin pair.

    The file should have the following format:
    
    - The first column is the index of the spin.
    - The subsequent columns represent the components of a 3x3 tensor.
    
    This structure is repeated for each spin.

    TODO: Input mahdollinen ilman nollatensoreita?

    Parameters
    ----------
    file_path : str
        Path to the file containing the tensors.

    Returns
    -------
    tensors : ndarray
        A NumPy array containing the tensors.
    """

    # Initialize the lists and the current index
    tensors = []
    matrix_rows = []
    current_index = None
    
    # Open the file
    with open(file_path, 'r') as file:

        # Process each line
        for line in file:

            # Handle lines with spin indices differently
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
    
    # Convert to a NumPy array
    tensors = np.array(tensors, dtype=float)

    return tensors