"""
Inversion-recovery pulse sequence with continuous recovery detection.
"""

# Imports
from copy import deepcopy
import numpy as np
import spinguin._core as sg
from tqdm import tqdm

def inversion_recovery(
    spin_system: sg.SpinSystem,
    isotope: str,
    npoints: int,
    time_step: float,
) -> np.ndarray:
    """
    Perform an inversion-recovery experiment with direct recovery monitoring.

    This implementation differs slightly from a conventional spectrometer
    experiment. The inversion pulse is applied only once, and the recovery is
    monitored continuously by measuring the longitudinal magnetisation at each
    simulated time step.

    If a traditional inversion-recovery experiment is desired, use
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
        Spin system on which the inversion-recovery experiment is performed.
    isotope : str
        Isotope, for example `'1H'`, whose longitudinal magnetisation is
        inverted and detected. Hard pulses are applied.
    npoints : int
        Number of points in the simulation. Defines the total simulation time
        together with `time_step`.
    time_step : float
        Time step in the simulation, in seconds. This should usually be kept
        relatively short, for example 1 ms.

    Returns
    -------
    magnetizations : ndarray
        Two-dimensional array of shape `(nspins, npoints)` containing the
        observed z-magnetisations for the selected spins at each time point.

    Raises
    ------
    ValueError
        Raised if the basis, relaxation settings, magnetic field, or
        temperature required by the sequence have not been defined.
    """

    # Operate on a copy so that truncation does not modify the input object.
    spin_system = deepcopy(spin_system)

    # Check that the sequence prerequisites have been defined.
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

    # Construct the Liouvillian governing the spin dynamics.
    H = sg.hamiltonian(spin_system)
    R = sg.relaxation(spin_system)
    L = sg.liouvillian(H, R)

    # Construct the thermal-equilibrium state.
    rho = sg.equilibrium_state(spin_system)

    # Identify the spins whose magnetisation is inverted and detected.
    indices = np.where(spin_system.isotopes == isotope)[0]
    nspins = indices.shape[0]

    # Apply a hard 180-degree pulse to the selected isotope channel.
    operator = "+".join(f"I(x,{i})" for i in indices)
    P180 = sg.pulse(spin_system, operator, 180)
    rho = P180 @ rho

    # Restrict the dynamics to the zero-quantum basis for efficiency.
    L, rho = spin_system.basis.truncate_by_coherence([0], L, rho)

    # Construct the propagator for one simulation time step.
    P = sg.propagator(L, time_step)

    # Allocate the array used to store the longitudinal magnetisations.
    magnetizations = np.zeros((nspins, npoints), dtype=complex)

    # Propagate the state and record the longitudinal magnetisation.
    # for step in range(npoints):
    for step in tqdm(range(npoints), desc="Time propagation"):
        for i, idx in enumerate(indices):
            magnetizations[i, step] = sg.measure(
                spin_system,
                rho,
                f"I(z,{idx})",
            )
        rho = P @ rho

    return magnetizations