# Imports
from copy import deepcopy
import numpy as np
import spinguin._core as sg

def inversion_recovery(
    spin_system: sg.SpinSystem,
    isotope: str,
    npoints: int,
    time_step: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Performs the inversion-recovery experiment. The experiment differs slightly
    from the actual inversion-recovery experiments performed on spectrometers.
    In this experiment, the inversion is performed only once, and the
    magnetization is detected at each step during the recovery (much faster).
    
    If the traditional inversion recovery is desired, use the function
    `inversion_recovery_fid()`.

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
        Spin system to which the inversion-recovery experiment is performed.
    isotope : str
        Specifies the isotope, for example "1H", whose magnetization is inverted
        and detected. This function applies hard pulses.
    npoints : int
        Number of points in the simulation. Defines the total simulation time
        together with `time_step`.
    time_step : float
        Time step in the simulation (in seconds). Should be kept relatively
        short (e.g. 1 ms).

    Returns
    -------
    magnetizations : ndarray
        Two-dimensional array of size (nspins, npoints) containing the
        observed z-magnetizations for each spin at various times.
    """
    # Operate on a copy of the SpinSystem object
    spin_system = deepcopy(spin_system)

    # Check that the required attributes have been set
    if spin_system.basis.basis is None:
        raise ValueError("Please build the basis before using "
                         "inversion recovery.")
    if spin_system.relaxation.theory is None:
        raise ValueError("Please set the relaxation theory before using "
                         "inversion recovery.")
    if spin_system.relaxation.thermalization is False:
        raise ValueError("Please set thermalization to True before using "
                         "inversion recovery.")
    if sg.parameters.magnetic_field is None:
        raise ValueError("Please set the magnetic field before using "
                         "inversion recovery.")
    if sg.parameters.temperature is None:
        raise ValueError("Please set the temperature before using "
                         "inversion recovery.")
    
    # Obtain the Liouvillian
    H = sg.hamiltonian(spin_system)
    R = sg.relaxation(spin_system)
    L = sg.liouvillian(H, R)

    # Obtain the equilibrium state
    rho = sg.equilibrium_state(spin_system)

    # Find indices of the isotopes to be measured
    indices = np.where(spin_system.isotopes == isotope)[0]
    nspins = indices.shape[0]

    # Apply 180-degree pulse
    operator = "+".join(f"I(x,{i})" for i in indices)
    P180 = sg.pulse(spin_system, operator, 180)
    rho = P180 @ rho

    # Change to ZQ-basis to speed up the calculations
    L, rho = spin_system.basis.truncate_by_coherence([0], L, rho)

    # Construct the time propagator
    P = sg.propagator(L, time_step)

    # Initialize an array for storing results
    magnetizations = np.zeros((nspins, npoints), dtype=complex)

    # Perform the time evolution
    for step in range(npoints):
        for i, idx in enumerate(indices):
            magnetizations[i, step] = \
                sg.measure(spin_system, rho, f"I(z,{idx})")
        rho = P @ rho

    return magnetizations