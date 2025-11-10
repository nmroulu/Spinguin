# Imports
import numpy as np
import spinguin._core as sg

def pulse_and_acquire(
    spin_system: sg.SpinSystem,
    isotope: str,
    center_frequency: float,
    npoints: int,
    dwell_time: float,
    angle: float       
) -> np.ndarray:
    """
    Simple pulse-and-acquire experiment.

    This experiment requires the following spin system properties to be defined:

    - spin_system.basis : must be built
    - spin_system.relaxation.theory
    - spin_system.relaxation.thermalization : must be True

    This experiment requires the following parameters to be defined:

    - parameters.magnetic_field : magnetic field of the spectrometer in Tesla
    - parameters.temperature : temperature of the sample in Kelvin

    Parameters
    ----------
    spin_system : SpinSystem
        Spin system to which the pulse-and-acquire experiment is performed.

    Returns
    -------
    fid : ndarray
        Free induction decay signal.
    """
    # Obtain the Liouvillian
    H = sg.hamiltonian(spin_system)
    R = sg.relaxation(spin_system)
    L = sg.liouvillian(H, R)

    # Obtain the equilibrium state
    rho = sg.equilibrium_state(spin_system)

    # Find indices of the isotopes to be measured
    indices = np.where(spin_system.isotopes == isotope)[0]

    # Apply pulse
    op_pulse = "+".join(f"I(y,{i})" for i in indices)
    Px = sg.pulse(spin_system, op_pulse, angle)
    rho = Px @ rho

    # Construct the time propagator
    P = sg.propagator(L=L, t=dwell_time)
    P = sg.propagator_to_rotframe(
        spin_system = spin_system,
        P = P,
        t = dwell_time,
        center_frequencies = {isotope: center_frequency}
    )

    # Initialize an array for storing results
    fid = np.zeros(npoints, dtype=complex)

    # Perform the time evolution
    op_measure = "+".join(f"I(-,{i})" for i in indices)
    for step in range(npoints):
        fid[step] = sg.measure(spin_system, rho, op_measure)
        rho = P @ rho
    
    return fid