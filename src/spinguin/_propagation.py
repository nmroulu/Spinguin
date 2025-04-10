"""
propagation.py

This module is responsible for calculating time propagators.
"""

# For referencing the SpinSystem class
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from spinguin._spin_system import SpinSystem

# Imports
import time
import numpy as np
from scipy.sparse import csc_array
from spinguin import _la
from spinguin._operators import superoperator
from typing import Union

def propagator(t: float,
               H: csc_array = None,
               R: csc_array = None,
               zero_value: float = 1e-18,
               density_threshold: float = 0.5,
               custom_dot: bool = False) -> Union[csc_array, np.ndarray]:
    """
    Constructs the time propagator.
    
    Parameters
    ----------
    t : float
        Time step of the simulation in seconds.
    H : csc_array, optional
        SciPy sparse array containing the Hamiltonian superoperator.
    R : csc_array, optional
        SciPy sparse array containing the relaxation superoperator.
    zero_value : float, optional
        Default: 1e-18. Values smaller than zero_value are treated as zero
        in the matrix exponential. Larger values improve performance by increasing sparsity.
        If a thermalized relaxation superoperator is used, consider reducing this parameter,
        especially at low fields.  
    density_threshold : float, optional
        Default: 0.5. If the propagator density exceeds density_threshold,
        the propagator is returned as a dense NumPy array.
    custom_dot : bool, optional
        Default: False. If False, dot products in the matrix exponentials are computed
        using the default SciPy implementation. If True, the custom implementation is used,
        which removes small values during computation. The custom implementation is
        parallelized using OpenMP.
        
    Returns
    -------
    exp_Lt : csc_array or numpy.ndarray
        Time propagator exp[(-iH - R)*t].
    """

    print("Constructing propagator...") # NOTE: Perttu's edit
    time_start = time.time()

    # Compute the total Liouvillian. Handle cases where either H or R is None.
    if H is None and R is None:
        raise ValueError("Both H and R cannot be None.")
    elif H is None:
        L = -R
    elif R is None:
        L = -1j * H
    else:
        L = -1j * H - R

    # Compute the matrix exponential
    if custom_dot:
        expm_Lt = _la.expm_custom_dot(L * t, zero_value)
    else:
        expm_Lt = _la.expm(L * t, zero_value)

    # Calculate the density of the propagator
    density = expm_Lt.nnz / (expm_Lt.shape[0] ** 2)

    # print(f"Propagator density: {density}")
    print(f"Propagator density: {density:.4f}") # NOTE: Perttu's edit

    # Convert to NumPy array if density exceeds the threshold
    if density > density_threshold:
        print("Density exceeds threshold. Converting to NumPy array.")
        expm_Lt = expm_Lt.toarray()

    print(f'Propagator constructed in {time.time() - time_start:.4f} seconds.') # NOTE: Perttu's edit
    print()

    return expm_Lt

# TODO: Update this
def pulse(spin_system: SpinSystem, operators: Union[str, list], indices: Union[int, list], angle: float, zero_value: float = 1e-18) -> csc_array:
    """
    Generates a superoperator corresponding to the pulse described
    by the given operator, list of spin indices, and angle.

    Parameters
    ----------
    spin_system : SpinSystem
        The spin system on which the pulse is applied.
    operators : str or list
        Defines the pulse to be generated. Can be either a string or a list of strings.
        - str :
            A superoperator is generated for each spin specified in `indices`, and the
            sum of the operators is used to generate the pulse superoperator.
            For example: 'I_z'
        - list :
            A product superoperator is generated based on the operators specified in the list.
            The pulse superoperator is then constructed. The length of the list must match
            the length of `indices`.
            For example: ['I_+', 'I_-']
    indices : int or list
        Indices of the spins to which the pulse will be applied.
    angle : float
        Pulse angle in degrees.
    zero_value : float, optional
        Default: 1e-18. Values smaller than zero_value are treated as zero when
        computing the pulse superoperator.

    Returns
    -------
    pul : csc_array
        Superoperator corresponding to the applied pulse.
    """

    # Generate the operator
    op = superoperator(spin_system, operators, indices)

    # Convert the angle to radians
    angle = angle / 180 * np.pi

    # Construct the pulse propagator
    pul = _la.expm(-1j * angle * op, zero_value)

    return pul