# Imports
from scipy.sparse import csc_array

def liouvillian(H: csc_array = None, R: csc_array = None, K: csc_array = None) -> csc_array:
    """
    Constructs the Liouvillian superoperator from the Hamiltonian, relaxation superoperator,
    and exchange superoperator.

    Parameters
    ----------
    H : csc_array
        Hamiltonian superoperator.
    R : csc_array
        Relaxation superoperator
    K : csc_array
        Exchange superoperator.

    Returns
    -------
    L : csc_array
        Liouvillian superoperator.
    """

    # Check for totally empty input
    if H is None and R is None and K is None:
        raise ValueError("H, R and K cannot all be None simultaneously.")

    # Assign zeroes if None
    if H is None:
        H = 0
    if R is None:
        R = 0
    if K is None:
        K = 0

    # Construct the Liouvillian
    L = -1j*H - R + K

    return L