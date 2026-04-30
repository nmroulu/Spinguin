"""
Simple pulse-and-acquire sequence for simulating a free induction decay.

This module provides a basic hard-pulse excitation sequence followed by
direct detection of the transverse signal in the rotating frame.
"""

import numpy as np
import spinguin._core as sg


def pulse_and_acquire(
    spin_system: sg.SpinSystem,
    isotope: str,
    center_frequency: float,
    npoints: int,
    dwell_time: float,
    angle: float,
) -> np.ndarray:
    """
    Perform a simple pulse-and-acquire experiment.

    This experiment requires the following spin system properties to be defined:

    - ``spin_system.basis``: must be built
    - ``spin_system.relaxation.theory``
    - ``spin_system.relaxation.thermalization``: must be ``True``

    This experiment requires the following parameters to be defined:

    - ``parameters.magnetic_field``: magnetic field of the spectrometer in T
    - ``parameters.temperature``: temperature of the sample in K

    Parameters
    ----------
    spin_system : SpinSystem
        Spin system on which the pulse-and-acquire experiment is performed.
    isotope : str
        Isotope, for example ``'1H'``, that is excited and observed.
    center_frequency : float
        Rotating-frame centre frequency in Hz for the observed isotope
        channel.
    npoints : int
        Number of complex points in the simulated free induction decay.
    dwell_time : float
        Time step between successive data points, in seconds.
    angle : float
        Pulse flip angle in degrees.

    Returns
    -------
    fid : ndarray
        Free induction decay signal.

    Raises
    ------
    ValueError
        Propagated if the underlying Hamiltonian, relaxation, or equilibrium
        state calculations are missing required settings, or if the requested
        isotope is not present in the spin system.
    """

    # Construct the Liouvillian governing the spin dynamics.
    H = sg.hamiltonian(spin_system)
    R = sg.relaxation(spin_system)
    L = sg.liouvillian(H, R)

    # Construct the thermal-equilibrium state.
    rho = sg.equilibrium_state(spin_system)

    # Identify the spins belonging to the observed isotope channel.
    indices = np.where(spin_system.isotopes == isotope)[0]
    if indices.size == 0:
        raise ValueError(
            f"Isotope {isotope} is not present in the spin system."
        )

    # Apply the hard excitation pulse to the selected isotope channel.
    pulse_operator_string = "+".join(f"I(y,{i})" for i in indices)
    pulse_operator = sg.pulse(spin_system, pulse_operator_string, angle)
    rho = pulse_operator @ rho

    # Construct the propagator for one dwell-time step.
    P = sg.propagator(L=L, t=dwell_time)
    P = sg.propagator_to_rotframe(
        spin_system=spin_system,
        P=P,
        t=dwell_time,
        center_frequencies={isotope: center_frequency},
    )

    # Allocate the free induction decay array.
    fid = np.zeros(npoints, dtype=complex)

    # Propagate the state and record the detected transverse signal.
    measurement_operator = "+".join(f"I(-,{i})" for i in indices)
    for step in range(npoints):
        fid[step] = sg.measure(spin_system, rho, measurement_operator)
        rho = P @ rho

    return fid