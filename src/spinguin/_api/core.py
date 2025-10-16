"""
core.py

This module provides user-friendly wrapper functions for the core functionality
of the Spinguin package.
"""

# Imports
import numpy as np
import scipy.sparse as sp
from typing import Literal

from spinguin._core._config import config
from spinguin._core._parameters import parameters
from spinguin._core._spin_system import SpinSystem
from spinguin._core._hamiltonian import hamiltonian
from spinguin._core._hamiltonian import _sop_H
from spinguin._core.hide_prints import HidePrints
from spinguin._core.la import norm_1
from spinguin._core.liouvillian import (
    sop_L as liouvillian,
    sop_L_to_rotframe as _sop_L_to_rotframe
)
from spinguin._core.nmr_isotopes import gamma, quadrupole_moment, spin
from spinguin._core.propagation import (
    propagator_to_rotframe as _propagator_to_rotframe,
    sop_pulse as _sop_pulse,
    propagator
)
from spinguin._core.specutils import (
    frequency_to_chemical_shift,
    resonance_frequency as _resonance_frequency,
    spectral_width_to_dwell_time as _spectral_width_to_dwell_time,
    spectrum as _spectrum
)
from spinguin._core._superoperators import superoperator
from spinguin._core.states import (
    alpha_state as _alpha_state,
    beta_state as _beta_state,
    equilibrium_state as _equilibrium_state,
    measure as _measure,
    singlet_state as _singlet_state,
    state_from_string as _state_from_string,
    state_to_zeeman as _state_to_zeeman,
    triplet_minus_state as _triplet_minus_state,
    triplet_plus_state as _triplet_plus_state,
    triplet_zero_state as _triplet_zero_state,
    unit_state as _unit_state
)

__all__ = [
    "alpha_state",
    "associate",
    "beta_state",
    "dissociate",
    "equilibrium_state",
    "frequency_to_chemical_shift",
    "gamma",
    "hamiltonian",
    "inversion_recovery",
    "liouvillian",
    "measure",
    "permute_spins",
    "propagator",
    "propagator_to_rotframe",
    "pulse",
    "pulse_and_acquire",
    "quadrupole_moment",
    "resonance_frequency",
    "rotating_frame"
    "singlet_state",
    "spectral_width_to_dwell_time",
    "spectrum",
    "spin",
    "state",
    "state_to_zeeman",
    "superoperator",
    "time_axis",
    "triplet_minus_state",
    "triplet_plus_state",
    "triplet_zero_state",
    "unit_state"
]

INTERACTIONDEFAULT = ["zeeman", "chemical_shift", "J_coupling"]

def equilibrium_state(spin_system: SpinSystem) -> np.ndarray | sp.csc_array:
    """
    Creates the thermal equilibrium state for the spin system.

    Parameters
    ----------
    spin_system : SpinSystem
        Spin system for which the equilibrium state is going to be created.

    Returns
    -------
    rho : ndarray or csc_array
        Equilibrium state vector.
    """

    # Check that the required attributes are set
    if spin_system.basis.basis is None:
        raise ValueError("Please build the basis before constructing the "
                         "equilibrium state.")
    if parameters.magnetic_field is None:
        raise ValueError("Please set the magnetic field before "
                         "constructing the equilibrium state.")
    if parameters.temperature is None:
        raise ValueError("Please set the temperature before "
                         "constructing the equilibrium state.")

    # Build the left Hamiltonian superoperator
    with HidePrints():
        H_left = hamiltonian(
            spin_system = spin_system,
            interactions = INTERACTIONDEFAULT,
            side = "left"
        )

    # Build the equilibrium state
    rho = _equilibrium_state(
        basis = spin_system.basis.basis,
        spins = spin_system.spins,
        H_left = H_left,
        T = parameters.temperature,
        sparse = config.sparse_state,
        zero_value = config.zero_equilibrium
    )

    return rho

def singlet_state(spin_system: SpinSystem,
                  index_1 : int,
                  index_2 : int) -> np.ndarray | sp.csc_array:
    """
    Generates the singlet state between two spin-1/2 nuclei. Unit state is
    assigned to the other spins.

    Parameters
    ----------
    spin_system : SpinSystem
        Spin system for which the singlet state is created.
    index_1 : int
        Index of the first spin in the singlet state.
    index_2 : int
        Index of the second spin in the singlet state.

    Returns
    -------
    rho : ndarray or csc_array
        State vector corresponding to the singlet state.
    """
    # Check that the required attributes are set
    if spin_system.basis.basis is None:
        raise ValueError("Please build the basis before constructing the "
                         "singlet state.")

    # Build the singlet state
    rho = _singlet_state(
        basis = spin_system.basis.basis,
        spins = spin_system.spins,
        index_1 = index_1,
        index_2 = index_2,
        sparse = config.sparse_state
    )

    return rho

def pulse(spin_system: SpinSystem,
          operator: str,
          angle: float) -> np.ndarray | sp.csc_array:
    """
    Creates a pulse superoperator that is applied to a state by multiplying
    from the left.

    Parameters
    ----------
    spin_system : SpinSystem
        Spin system for which the pulse superoperator is going to be created.
    operator : str
        Defines the pulse to be generated. The operator string must
        follow the rules below:

        - Cartesian and ladder operators: `I(component,index)` or
          `I(component)`. Examples:

            - `I(x,4)` --> Creates x-operator for spin at index 4.
            - `I(x)`--> Creates x-operator for all spins.

        - Spherical tensor operators: `T(l,q,index)` or `T(l,q)`. Examples:

            - `T(1,-1,3)` --> \
              Creates operator with `l=1`, `q=-1` for spin at index 3.
            - `T(1, -1)` --> \
              Creates operator with `l=1`, `q=-1` for all spins.
            
        - Product operators have `*` in between the single-spin operators:
          `I(z,0) * I(z,1)`
        - Sums of operators have `+` in between the operators:
          `I(x,0) + I(x,1)`
        - Unit operators are ignored in the input. Interpretation of these
          two is identical: `E * I(z,1)`, `I(z,1)`
        
        Special case: An empty `operator` string is considered as unit operator.

        Whitespace will be ignored in the input.

        NOTE: Indexing starts from 0!
    angle : float
        Pulse angle in degrees.

    Returns
    -------
    P : ndarray or csc_array
        Pulse superoperator.
    """

    # Check that the required attributes have been set
    if spin_system.basis.basis is None:
        raise ValueError("Please build the basis before constructing pulse "
                         "superoperators.")

    # Construct the pulse superoperator
    P = _sop_pulse(
        basis = spin_system.basis.basis,
        spins = spin_system.spins,
        operator = operator,
        angle = angle,
        sparse = config.sparse_pulse,
        zero_value = config.zero_pulse
    )

    return P

def propagator_to_rotframe(spin_system: SpinSystem,
                           P: np.ndarray | sp.csc_array,
                           t: float,
                           center_frequencies: dict=None
                           ) -> np.ndarray | sp.csc_array:
    """
    Transforms the time propagator to the rotating frame.

    Parameters
    ----------
    spin_system : SpinSystem
        Spin system whose time propagator is going to be transformed.
    P : ndarray or csc_array
        Time propagator in the laboratory frame.
    t : float
        Time step of the simulation in seconds.
    center_frequencies : dict
        Dictionary that describes the center frequencies for each isotope in the
        units of ppm.

    Returns
    -------
    P_rot : ndarray or csc_array
        The time propagator transformed into the rotating frame.
    """
    # Obtain an array of center frequencies for each spin
    center = np.zeros(spin_system.nspins)
    for spin in range(spin_system.nspins):
        if spin_system.isotopes[spin] in center_frequencies:
            center[spin] = center_frequencies[spin_system.isotopes[spin]]

    # Construct Hamiltonian that specifies the interaction frame
    H_frame = _sop_H(
        basis = spin_system.basis.basis,
        spins = spin_system.spins,
        gammas = spin_system.gammas,
        B = parameters.magnetic_field,
        chemical_shifts = center,
        interactions = ["zeeman", "chemical_shift"],
        side = "comm",
        sparse = config.sparse_hamiltonian,
        zero_value = config.zero_hamiltonian
    )

    # Convert the propagator to rotating frame
    P_rot = _propagator_to_rotframe(
        sop_P = P,
        sop_H0 = H_frame,
        t = t,
        zero_value = config.zero_propagator
    )
    
    return P_rot

def measure(spin_system: SpinSystem,
            rho: np.ndarray | sp.csc_array,
            operator: str) -> complex:
    """
    Computes the expectation value of the specified operator for a given state
    vector. Assumes that the state vector `rho` represents a trace-normalized
    density matrix.

    Parameters
    ----------
    spin_system : SpinSystem
        Spin system whose state is going to be measured.
    rho : ndarray or csc_array
        State vector that describes the density matrix.
    operator : str
        Defines the operator to be measured. The operator string must follow the
        rules below:

        - Cartesian and ladder operators: `I(component,index)` or
          `I(component)`. Examples:

            - `I(x,4)` --> Creates x-operator for spin at index 4.
            - `I(x)` --> Creates x-operator for all spins.

        - Spherical tensor operators: `T(l,q,index)` or `T(l,q)`. Examples:

            - `T(1,-1,3)` --> \
              Creates operator with `l=1`, `q=-1` for spin at index 3.
            - `T(1, -1)` --> \
              Creates operator with `l=1`, `q=-1` for all spins.
            
        - Product operators have `*` in between the single-spin operators:
          `I(z,0) * I(z,1)`
        - Sums of operators have `+` in between the operators:
          `I(x,0) + I(x,1)`
        - Unit operators are ignored in the input. Interpretation of these
          two is identical: `E * I(z,1)`, `I(z,1)`
        
        Special case: An empty `operator` string is considered as unit operator.

        Whitespace will be ignored in the input.

        NOTE: Indexing starts from 0!

    Returns
    -------
    ex : complex
        Expectation value.
    """
    # Check that the required attributes are set
    if spin_system.basis.basis is None:
        raise ValueError("Please build the basis before measuring an "
                         "expectation value of an operator.")
    
    # Perform the measurement
    ex = _measure(
        basis = spin_system.basis.basis,
        spins = spin_system.spins,
        rho = rho,
        operator = operator
    )

    return ex

def spectral_width_to_dwell_time(
        spectral_width: float,
        isotope: str
) -> float:
    """
    Calculates the dwell time (in seconds) from the spectral width given in ppm.

    Parameters
    ----------
    spectral_width : float
        Spectral width in ppm.
    isotope: str
        Nucleus symbol (e.g. `'1H'`) used to select the gyromagnetic ratio 
        required for the conversion.

    Returns
    -------
    dwell_time : float
        Dwell time in seconds.

    Notes
    -----
    Requires that the following is set:

    - parameters.magnetic_field
    """
    # Obtain the dwell time
    dwell_time = _spectral_width_to_dwell_time(
        spectral_width = spectral_width,
        isotope = isotope,
        B = parameters.magnetic_field
    )

    return dwell_time

def spectrum(signal: np.ndarray,
             dwell_time: float,
             normalize: bool = True,
             part: Literal["real", "imag"] = "real"
             ) -> tuple[np.ndarray, np.ndarray]:
    """
    A wrapper function for the Fourier transform. Computes the Fourier transform
    and returns the frequency and spectrum (either the real or imaginary part of 
    the Fourier transform).

    Parameters
    ----------
    signal : ndarray
        Input signal in the time domain.
    dwell_time : float
        Time step between consecutive samples in the signal.
    normalize : bool, default=True
        Whether to normalize the Fourier transform.
    part : {'real', 'imag'}
        Specifies which part of the Fourier transform to return. Can be "real" 
        or "imag".

    Returns
    -------
    freqs : ndarray
        Frequencies corresponding to the Fourier-transformed signal.
    spectrum : ndarray
        Specified part (real or imaginary) of the Fourier-transformed signal 
        in the frequency domain.

    Notes
    -----
    Required global parameters:

    - parameters.dwell_time
    """
    # Compute the Fourier transform
    freqs, spectrum = _spectrum(
        signal = signal,
        dt = dwell_time,
        normalize = normalize,
        part = part
    )

    return freqs, spectrum

def resonance_frequency(
        isotope: str,
        offset: float = 0,
        unit: Literal["Hz", "rad/s"] = "Hz"
) -> float:
    """
    Computes the resonance frequency of the given `isotope` at the specified
    magnetic field.

    Parameters
    ----------
    isotope : str
        Nucleus symbol (e.g. `'1H'`) used to select the gyromagnetic ratio 
        required for the conversion.
    offset : float, default=0
        Offset in ppm.
    unit : {'Hz', 'rad/s'}
        Specifies in which units the frequency is returned.
    
    Returns
    -------
    omega : float
        Resonance frequency in the requested units.

    Notes
    -----
    Required global parameters:

    - parameters.magnetic_field
    """
    # Get the resonance frequency
    omega = _resonance_frequency(
        isotope = isotope,
        B = parameters.magnetic_field,
        delta = offset,
        unit = unit
    )

    return omega

def time_axis(npoints: int, time_step: float):
    """
    Generates a 1D array with `npoints` elements using a constant `time_step`.

    Parameters
    ----------
    npoints : int
        Number of points.
    time_step : float
        Time step (in seconds).
    """
    # Obtain the time array
    start = 0
    stop = npoints * time_step
    num = npoints
    t_axis = np.linspace(start, stop, num, endpoint=False)

    return t_axis

def unit_state(spin_system: SpinSystem,
               normalized: bool=True) -> np.ndarray | sp.csc_array:
    """
    Returns a unit state vector, which represents the identity operator. The
    output can be either normalised (trace equal to one) or unnormalised (raw
    identity matrix).

    Parameters
    ----------
    spin_system : SpinSystem
        Spin system to which the unit state is created.
    normalized : bool, default=True
        If set to True, the function will return a state vector that represents
        the trace-normalized density matrix. If False, returns a state vector
        that corresponds to the identity operator.

    Returns
    -------
    rho : ndarray or csc_array
        State vector corresponding to the unit state.
    """
    # Create the unit state
    rho = _unit_state(
        basis = spin_system.basis.basis,
        spins = spin_system.spins,
        sparse = config.sparse_state,
        normalized = normalized
    )

    return rho

def state(spin_system: SpinSystem,
          operator: str) -> np.ndarray | sp.csc_array:
    """
    This function returns a column vector representing the density matrix as a
    linear combination of spin operators. Each element of the vector corresponds
    to the coefficient of a specific spin operator in the expansion.
    
    Normalization:
    The output of this function uses a normalised basis built from normalised
    products of single-spin spherical tensor operators. However, the
    coefficients are scaled so that the resulting linear combination represents
    the non-normalised version of the requested operator.

    NOTE: This function is sometimes called often and is cached for high
    performance.

    Parameters
    ----------
    spin_system : SpinSystem
        Spin system for which the state is created.
    operator : str
        Defines the state to be generated. The operator string must follow the
        rules below:

        - Cartesian and ladder operators: `I(component,index)` or
          `I(component)`. Examples:

            - `I(x,4)` --> Creates x-operator for spin at index 4.
            - `I(x)`--> Creates x-operator for all spins.

        - Spherical tensor operators: `T(l,q,index)` or `T(l,q)`. Examples:

            - `T(1,-1,3)` --> \
              Creates operator with `l=1`, `q=-1` for spin at index 3.
            - `T(1, -1)` --> \
              Creates operator with `l=1`, `q=-1` for all spins.
            
        - Product operators have `*` in between the single-spin operators:
          `I(z,0) * I(z,1)`
        - Sums of operators have `+` in between the operators:
          `I(x,0) + I(x,1)`
        - Unit operators are ignored in the input. Interpretation of these
          two is identical: `E * I(z,1)`, `I(z,1)`
        
        Special case: An empty `operator` string is considered as unit operator.

        Whitespace will be ignored in the input.

        NOTE: Indexing starts from 0!

    Returns
    -------
    rho : ndarray or csc_array
        State vector corresponding to the requested state.
    """
    # Check that the required attributes are set
    if spin_system.basis.basis is None:
        raise ValueError("Please build the basis before constructing a state.")
    
    # Build the state
    rho = _state_from_string(
        basis = spin_system.basis.basis,
        spins = spin_system.spins,
        operator = operator,
        sparse = config.sparse_state
    )

    return rho

def state_to_zeeman(
        spin_system: SpinSystem,
        rho: np.ndarray | sp.csc_array
        ) -> np.ndarray | sp.csc_array:
    """
    Takes the state vector defined in the normalized spherical tensor basis
    and converts it into the Zeeman eigenbasis. Useful for error checking.

    Parameters
    ----------
    spin_system : SpinSystem
        Spin system whose state vector is going to be converted into a density
        matrix.
    rho : ndarray or csc_array
        State vector defined in the normalized spherical tensor basis.

    Returns
    -------
    rho_zeeman : ndarray or csc_array
        Spin density matrix defined in the Zeeman eigenbasis.
    """
    # Check that the required attributes are set
    if spin_system.basis.basis is None:
        raise ValueError("Please build the basis before converting the "
                         "state vector into density matrix.")
    
    # Convert the state vector into density matrix
    rho_zeeman = _state_to_zeeman(
        basis = spin_system.basis.basis,
        spins = spin_system.spins,
        rho = rho,
        sparse = config.sparse_state
    )
    
    return rho_zeeman

def alpha_state(spin_system: SpinSystem,
                index: int) -> np.ndarray | sp.csc_array:
    """
    Generates the alpha state for a given spin-1/2 nucleus. Unit state is
    assigned to the other spins.

    Parameters
    ----------
    spin_system: SpinSystem
        Spin system for which the alpha state is created.
    index : int
        Index of the spin that has the alpha state.

    Returns
    -------
    rho : ndarray or csc_array
        State vector corresponding to the alpha state of the given spin index.
    """
    # Check that the required attributes are set
    if spin_system.basis.basis is None:
        raise ValueError("Please build the basis before constructing the "
                         "alpha state.")
    
    # Create the alpha state
    rho = _alpha_state(
        basis = spin_system.basis.basis,
        spins = spin_system.spins,
        index = index,
        sparse = config.sparse_state
    )

    return rho

def beta_state(spin_system: SpinSystem,
               index: int) -> np.ndarray | sp.csc_array:
    """
    Generates the beta state for a given spin-1/2 nucleus. Unit state is
    assigned to the other spins.

    Parameters
    ----------
    spin_system: SpinSystem
        Spin system for which the beta state is created.
    index : int
        Index of the spin that has the beta state.

    Returns
    -------
    rho : ndarray or csc_array
        State vector corresponding to the beta state of the given spin index.
    """
    # Check that the required attributes are set
    if spin_system.basis.basis is None:
        raise ValueError("Please build the basis before constructing the "
                         "beta state.")
    
    # Create the beta state
    rho = _beta_state(
        basis = spin_system.basis.basis,
        spins = spin_system.spins,
        index = index,
        sparse = config.sparse_state
    )

    return rho

def triplet_zero_state(spin_system: SpinSystem,
                       index_1: int,
                       index_2: int) -> np.ndarray | sp.csc_array:
    """
    Generates the triplet zero state between two spin-1/2 nuclei. Unit state is
    assigned to the other spins.

    Parameters
    ----------
    spin_system: SpinSystem
        Spin system for which the triplet zero state is created.
    index_1 : int
        Index of the first spin in the triplet zero state.
    index_2 : int
        Index of the second spin in the triplet zero state.

    Returns
    -------
    rho : ndarray or csc_array
        State vector corresponding to the triplet zero state.
    """
    # Check that the required attributes are set
    if spin_system.basis.basis is None:
        raise ValueError("Please build the basis before constructing the "
                         "triplet zero state.")
    
    # Make the triplet zero state
    rho = _triplet_zero_state(
        basis = spin_system.basis.basis,
        spins = spin_system.spins,
        index_1 = index_1,
        index_2 = index_2,
        sparse = config.sparse_state
    )

    return rho

def triplet_plus_state(spin_system: SpinSystem,
                       index_1: int,
                       index_2: int) -> np.ndarray | sp.csc_array:
    """
    Generates the triplet plus state between two spin-1/2 nuclei. Unit state is
    assigned to the other spins.

    Parameters
    ----------
    spin_system : SpinSystem
        Spin system for which the triplet plus state is created.
    index_1 : int
        Index of the first spin in the triplet plus state.
    index_2 : int
        Index of the second spin in the triplet plus state.

    Returns
    -------
    rho : ndarray or csc_array
        State vector corresponding to the triplet plus state.
    """
    # Check that the required attributes are set
    if spin_system.basis.basis is None:
        raise ValueError("Please build the basis before constructing the "
                         "triplet plus state.")
    
    # Create the triplet plus state
    rho = _triplet_plus_state(
        basis = spin_system.basis.basis,
        spins = spin_system.spins,
        index_1 = index_1,
        index_2 = index_2,
        sparse = config.sparse_state
    )

    return rho

def triplet_minus_state(spin_system: SpinSystem,
                        index_1: int,
                        index_2: int) -> np.ndarray | sp.csc_array:
    """
    Generates the triplet minus state between two spin-1/2 nuclei. Unit state is
    assigned to the other spins.

    Parameters
    ----------
    spin_system : SpinSystem
        Spin system for which the triplet minus state is created.
    index_1 : int
        Index of the first spin in the triplet minus state.
    index_2 : int
        Index of the second spin in the triplet minus state.

    Returns
    -------
    rho : ndarray or csc_array
        State vector corresponding to the triplet minus state.
    """
    # Check that the required attributes are set
    if spin_system.basis.basis is None:
        raise ValueError("Please build the basis before constructing the "
                         "triplet minus state.")
    
    # Create the triplet minus state
    rho = _triplet_minus_state(
        basis = spin_system.basis.basis,
        spins = spin_system.spins,
        index_1 = index_1,
        index_2 = index_2,
        sparse = config.sparse_state
    )

    return rho

def rotating_frame(
    spin_system: SpinSystem,
    L: np.ndarray | sp.csc_array,
    isotopes: list,
    orders: list = [],
    center_frequencies: list = [],
) -> np.ndarray | sp.csc_array:
    """
    Transforms the Liouvillian into the rotating frame.

    Parameters
    ----------
    spin_system : SpinSystem
        Spin system whose Liouvillian is going to be transformed.
    L : ndarray or csc_array
        Liouvillian superoperator in the laboratory frame.
    isotopes : list
        List of isotopes whose rotating frames are applied.
    orders : list, default=[]
        List of integers that define the order of the rotating frame for each
        isotope. If empty, the default value defined in
        `parameters.rotating_frame_order` is used for all isotopes.
    center_frequencies : list, default=[]
        List of center frequencies (in ppm) for each isotope. If empty, zero is
        used for all isotopes.

    Returns
    -------
    L_rot : ndarray or csc_array
        Liouvillian superoperator in the rotating frame.
    """
    # Check input types
    if not isinstance(spin_system, SpinSystem):
        raise ValueError("spin_system must be a SpinSystem object")
    if not isinstance(L, (np.ndarray, sp.csc_array)):
        raise ValueError("L must be NumPy array or SciPy CSC array")
    if not isinstance(isotopes, list):
        raise ValueError("isotopes must be a list")
    if not isinstance(orders, list):
        raise ValueError("orders must be a list")
    if not isinstance(center_frequencies, list):
        raise ValueError("center_frequencies must be a list")
    
    # Check list lengths
    if len(isotopes) == 0:
        raise ValueError("isotopes cannot be an empty list")
    if not (len(orders) == len(isotopes) or len(orders) == 0):
        raise ValueError(
            "orders must have the same length as isotopes or be empty")
    if not (len(center_frequencies) == len(isotopes) or
            len(center_frequencies) == 0):
        raise ValueError("center_frequencies must have the same length as "
                         "isotopes or be empty")
    
    # Check that the isotopes exist in the spin system
    for isotope in isotopes:
        if isotope not in spin_system.isotopes:
            raise ValueError(f"isotope {isotope} is not in the spin system")
        
    # Check that each given isotope is unique
    if not len(isotopes) == len(set(isotopes)):
        raise ValueError("given isotopes must be unique")
    
    # Fill input lists with default values
    if len(orders) == 0:
        orders = [parameters.rotating_frame_order for _ in range(len(isotopes))]
    if len(center_frequencies) == 0:
        center_frequencies = [0 for _ in range(len(isotopes))]

    # Frequencies for the interaction frames
    freqs = []
    for i in range(len(isotopes)):
        freq = resonance_frequency(isotopes[i], center_frequencies[i], "rad/s")
        freqs.append(freq)

    # Corresponding Hamiltonians
    hamiltonians = []
    for i in range(len(isotopes)):
        spins = np.where(spin_system.isotopes == isotopes[i])[0]
        dim = spin_system.basis.dim
        hamiltonian = sp.csc_array((dim, dim))
        for n in spins:
            Iz_n = superoperator(spin_system, f"I(z, {n})")
            hamiltonian = hamiltonian + freqs[i] * Iz_n
        hamiltonians.append(hamiltonian)

    # Corresponding Liouvillians
    L0s = []
    for hamiltonian in hamiltonians:
        L0s.append(-1j*hamiltonian)

    # Norms of the Liouvillians
    norms = []
    for L0 in L0s:
        norms.append(norm_1(L0))

    # Re-order based on the norms
    sort = np.argsort(norms)
    freqs = [freqs[i] for i in sort]
    orders = [orders[i] for i in sort]
    L0s = [L0s[i] for i in sort]

    # Calculate the periods
    Ts = []
    for freq in freqs:
        Ts.append(2*np.pi / freq)

    # Apply each rotating frame
    for i in range(len(L0s)):
        L0 = L0s[i]
        L1 = L - L0
        T = Ts[i]
        order = orders[i]
        L = _sop_L_to_rotframe(L0, L1, T, order, config.zero_aux_rotframe)

    return L