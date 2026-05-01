"""
Inversion-recovery sequence with continuous recovery monitoring.

This module provides a simplified inversion-recovery experiment in which the
longitudinal recovery is sampled directly at each simulated time step.
"""

from copy import deepcopy

import numpy as np
import spinguin._core as sg
from spinguin._core._validation import require
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
    ``inversion_recovery_fid()``.

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
        Spin system on which the inversion-recovery experiment is performed.
    isotope : str
        Isotope, for example ``'1H'``, whose longitudinal magnetisation is
        inverted and detected. Hard pulses are applied.
    npoints : int
        Number of points in the simulation. Defines the total simulation time
        together with ``time_step``.
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
        temperature required by the sequence have not been defined, or if the
        requested isotope is not present in the spin system.
    """

    # Operate on a copy so that basis truncation does not modify the input
    # object.
    spin_system = deepcopy(spin_system)

    # Check that the sequence prerequisites have been defined.
    require(
        spin_system,
        ["basis.basis", "relaxation.theory"],
        "using inversion recovery"
    )
    require(
        sg.parameters,
        ["magnetic_field", "temperature"],
        "using inversion recovery"
    )

    if spin_system.relaxation.thermalization is False:
        raise ValueError("Please set thermalization to True before using "
                         "inversion recovery.")

    # Construct the Liouvillian governing the spin dynamics.
    H = sg.hamiltonian(spin_system)
    R = sg.relaxation(spin_system)
    L = sg.liouvillian(H, R)

    # Construct the thermal-equilibrium state.
    rho = sg.equilibrium_state(spin_system)

    # Identify the spins whose magnetisation is inverted and detected.
    indices = np.where(spin_system.isotopes == isotope)[0]
    if indices.size == 0:
        raise ValueError(
            f"Isotope {isotope} is not present in the spin system."
        )
    nspins = indices.shape[0]

    # Apply a hard 180-degree pulse to the selected isotope channel.
    pulse_operator = "+".join(f"I(x,{i})" for i in indices)
    inversion_pulse = sg.pulse(spin_system, pulse_operator, 180)
    rho = inversion_pulse @ rho

    # Restrict the dynamics to the zero-quantum basis for efficiency.
    L, rho = spin_system.basis.truncate_by_coherence([0], L, rho)

    # Construct the propagator for one simulation time step.
    P = sg.propagator(L, time_step)

    # Allocate the array used to store the longitudinal magnetisations.
    magnetizations = np.zeros((nspins, npoints), dtype=complex)

    # Propagate the state and record the longitudinal magnetisation.
    for step in tqdm(range(npoints), desc="Time propagation"):
        for i, idx in enumerate(indices):
            magnetizations[i, step] = sg.measure(
                spin_system,
                rho,
                f"I(z,{idx})",
            )
        rho = P @ rho

    return magnetizations