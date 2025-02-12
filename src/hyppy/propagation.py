"""
propagation.py

This module is responsible for calculating the time propagators.
"""

# For referencing the SpinSystem class
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hyppy.spin_system import SpinSystem

# Imports
import time
import numpy as np
from scipy.sparse import csc_array
from hyppy import la
from hyppy.operators import superoperator
from typing import Union

def propagator(t:float,
            H: csc_array=None,
            R: csc_array=None,
            zero_value: float=1e-18,
            density_threshold: float=0.5,
            custom_dot: bool=False) -> Union[csc_array, np.ndarray]:
    """
    Builds the time propagator.
    
    Parameters
    ----------
    t : float
        Time step of the simulation in seconds.
    H : csc_array
        SciPy sparse array containing the Hamiltonian superoperator.
    R : csc_array
        SciPy sparse array containing the Relaxation superoperator.
    zero_value : float
        Default: 1e-18. Values less than zero_value are considered to be zero
        in the matrix exponential. Larger values result in faster performance due
        to increased sparsity. If thermalized relaxation superoperator is used, consider
        tightening this parameter, especially at low field.  
    density_threshold : float
        Default: 0.5. If the propagator density is larger than density_threshold,
        the propagator is returned as a dense NumPy array.
    custom_dot : bool
        Default: False. If set to False, dot products in the matrix exponentials are calculated
        using the default SciPy implementation. If set to True, the propagator function uses the
        custom implementation, which cleans small values on the go. The custom implementation is
        parallelized using OpenMP.
        
    Returns
    -------
    exp_Lt : csc_array or numpy.ndarray
        Time propagator exp[(-iH - R)*t).
    """

    print("Building the time propagator.")
    time_start = time.time()

    # Get the total Liouvillian. Consider cases where either H and R are None.
    if H is None and R is None:
        raise ValueError("Both H and R cannot be None.")
    elif H is None:
        L = -R
    elif R is None:
        L = -1j*H
    else:
        L = -1j*H - R

    # Calculate the matrix exponential
    if custom_dot:
        expm_Lt = la.expm_custom_dot(L*t, zero_value)
    else:
        expm_Lt = la.expm(L*t, zero_value)

    # Find out the density of the propagator
    density = expm_Lt.nnz / (expm_Lt.shape[0] ** 2)

    print(f"Propagator density: {density}")

    # Convert to NumPy if density exceeds threshold
    if density > density_threshold:
        print("Density exceeds threshold. Converting to NumPy.")
        expm_Lt = expm_Lt.toarray()

    print("Propagator constructed.")
    print(f"Elapsed time: {time.time() - time_start} seconds.")

    return expm_Lt

def pulse(spin_system:SpinSystem, operators: Union[str, list], indices: Union[int, list], angle: float, zero_value: float=1e-18) -> csc_array:
    """
    Generates a propagator that corresponds to the pulse described
    by the given operator and list of spins.

    Parameters
    ----------
    spin_system : SpinSystem
    operators : str or list
        Defines the pulse to be generated. Can be either a string, or a list of strings.
        - str :
            A superoperator is generated for each spin specified in `spins`, a sum of the
            operators is calculated, and a pulse corresponding to that operator is returned.
            For example: 'I_z'
        - list :
            A product superoperator corresponding to the operators specified in the list
            generated. A pulse corresponding to that operator is returned. Must match the
            length of `spins`.
            For example: ['I_+', 'I_-']
    indices : int or list
        Indices of the spins for which the pulse will be applied.
    angle : float
        Pulse angle in degrees.
    zero_value : float
        Default: 1e-18. Values less than zero_value are considered to be zero when
        computing the pulse propagator.

    Returns
    -------
    pul : csc_array
        Propagator that corresponds to applying the pulse.
    """

    # Get the operator
    op = superoperator(spin_system, operators, indices)

    # Convert angle to radians
    angle = angle / 180 * np.pi

    # Make the pulse
    pul = la.expm(-1j*angle*op, zero_value)

    return pul