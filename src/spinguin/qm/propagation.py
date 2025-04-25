"""
propagation.py

This module is responsible for calculating time propagators.
"""

# For referencing the SpinSystem class
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from spinguin.system.spin_system import SpinSystem

# Imports
import time
import numpy as np
import warnings
from scipy.sparse import csc_array
from spinguin.utils.la import expm, expm_custom_dot
from spinguin.qm.hamiltonian import hamiltonian_zeeman
from spinguin.qm.operators import superoperator, sop_prod
from spinguin.config import Config

def propagator(L: csc_array,
               t: float,
               rotating_frame: bool = False,
               spin_system: SpinSystem = None,
               B: float = None,
               custom_dot: bool = False) -> csc_array | np.ndarray:
    """
    Constructs the time propagator, with an optional transformation to the rotating frame.

    Parameters
    ----------
    L : csc_array
        Liouvillian superoperator, L = -iH - R + K.
    t : float
        Time step of the simulation in seconds.
    rotating_frame : bool, optional
        Default: False. If True, transforms the propagator to the rotating frame
        with respect to the bare-nucleus Zeeman Hamiltonian.
    spin_system : SpinSystem, optional
        Required if rotating_frame is True. The spin system object containing
        information about the spins.
    B : float, optional
        Required if rotating_frame is True. Magnetic field strength in Tesla (T).
    custom_dot : bool, optional
        Default: False. If False, dot products in the matrix exponentials are computed
        using the default SciPy implementation. If True, the custom implementation is used,
        which removes small values during computation. The custom implementation is
        parallelized using OpenMP.

    Returns
    -------
    exp_Lt : csc_array or numpy.ndarray
        Time propagator exp[L*t], optionally transformed to the rotating frame.
    """

    print("Constructing propagator...")
    time_start = time.time()

    # Compute the matrix exponential
    if custom_dot:
        expm_Lt = expm_custom_dot(L * t, Config.ZERO_PROPAGATOR)
    else:
        expm_Lt = expm(L * t, Config.ZERO_PROPAGATOR)

    # Calculate the density of the propagator
    density = expm_Lt.nnz / (expm_Lt.shape[0] ** 2)
    print(f"Propagator density: {density:.4f}")

    # Convert to NumPy array if density exceeds the threshold
    if density > Config.DENSITY_THRESHOLD:
        print("Density exceeds threshold. Converting to NumPy array.")
        expm_Lt = expm_Lt.toarray()

    # Apply rotating frame transformation if requested
    if rotating_frame:
        if spin_system is None or B is None:
            raise ValueError("spin_system and B must be provided when rotating_frame is True.")
        
        print("Applying rotating frame transformation...")
        H0 = hamiltonian_zeeman(spin_system, include_shifts=False)

        if custom_dot:
            expm_H0t = expm_custom_dot(1j * H0 * t, Config.ZERO_PROPAGATOR, disable_output=True)
        else:
            expm_H0t = expm(1j * H0 * t, Config.ZERO_PROPAGATOR, disable_output=True)

        expm_Lt = expm_H0t @ expm_Lt
        print("Rotating frame transformation applied.")

    print(f'Propagator constructed in {time.time() - time_start:.4f} seconds.')
    print()

    return expm_Lt

def pulse(spin_system: SpinSystem, operator: str, angle: float) -> csc_array:
    """
    Generates a superoperator corresponding to the pulse described
    by the given operator and angle.

    Parameters
    ----------
    spin_system : SpinSystem
        The spin system on which the pulse is applied.
    operator : str
        Defines the operator to be generated. The operator string must follow the rules below:

        - Cartesian and ladder operators: I(component,index). Example: I(x,4) --> Creates x-operator for spin at index 4.
        - Spherical tensor operators: T(l,q,index). Example: T(1,-1,3) --> Creates operator with l=1, q=-1 for spin at index 3.
        - Sums of operators have `+` in between the operators: I(x,0) + I(x,1)
        - The unit operator is not typed. Example: I(z,2) will generate E*I_z in case of a two-spin system. 
        - Whitespace will be ignored in the input.
    angle : float
        Pulse angle in degrees.

    Returns
    -------
    pul : csc_array
        Superoperator corresponding to the applied pulse.
    """

    time_start = time.time()
    print("Creating a pulse superoperator...")

    # Show a warning if pulse is generated using a product operator
    if '*' in operator:
        warnings.warn("Applying a pulse using a product operator does not have a well-defined angle.")

    # Generate the operator
    op = superoperator(spin_system, operator)

    # Convert the angle to radians
    angle = angle / 180 * np.pi

    # Construct the pulse propagator
    pul = expm(-1j * angle * op, Config.ZERO_PULSE, disable_output=True)

    print(f'Pulse constructed in {time.time() - time_start:.4f} seconds.\n')

    return pul

def spectrum_timestep(spin_system: SpinSystem,
                      B: float,
                      safety_factor: float = 1.2,
                      rotating_frame: bool = False,
                      return_bandwidth: bool = False) -> float | tuple[float, float]:
    """
    Computes the time step for a spectrum simulation based on the resonance frequencies
    of the spins and the spectrometer frequency. Optionally includes the bare-nucleus
    Zeeman frequencies if not in the rotating frame. Optionally returns the bandwidth as well.

    Parameters
    ----------
    spin_system : SpinSystem
        The spin system object containing information about the spins.
    B : float
        Magnetic field strength in Tesla (T).
    safety_factor : float, optional
        Default: 1.2. A factor to scale the bandwidth for safety.
    rotating_frame : bool, optional
        Default: False. If True, uses the resonance frequencies in the rotating frame.
        If False, includes the bare-nucleus Zeeman frequencies.
    return_bandwidth : bool, optional
        Default: False. If True, also returns the bandwidth in rad/s.

    Returns
    -------
    float or tuple[float, float]
        The time step in seconds. If return_bandwidth is True, also returns the bandwidth in rad/s.
    """
    # Extract relevant parameters from the spin system
    cs = spin_system.chemical_shifts
    ys = spin_system.gammas

    if rotating_frame:
        # Compute the resonance frequencies with respect to the spectrometer frequency
        resonance_frequencies = [(-ys[i] * B * cs[i] * 1e-6) for i in range(spin_system.size)]
    else:
        # Include bare-nucleus Zeeman frequencies
        resonance_frequencies = [(-ys[i] * B * (1 + cs[i] * 1e-6)) for i in range(spin_system.size)]

    # Get the most negative and most positive resonance frequencies
    min_freq = min(resonance_frequencies)
    max_freq = max(resonance_frequencies)

    # Calculate the bandwidth
    bandwidth = abs(max_freq - min_freq)

    # Apply the safety factor
    bandwidth *= safety_factor

    # Calculate the time step (Nyquist criterion)
    time_step = 1 / (4 * bandwidth)

    if return_bandwidth:
        return time_step, bandwidth
    return time_step
