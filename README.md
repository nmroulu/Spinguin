# HypPy

## Description
HypPy is a spin-dynamics package for Python tailored for hyperpolarization experiments. It provides tools for performing spin-dynamics simulations using restricted basis sets, allowing the use of large spin systems of more than 10 spins on a relatively weak hardware. HypPy supports the simulation of coherent dynamics, relaxation, and chemical exchange.

## Installation

### Requirements
HypPy has been developed and tested using Python 3.11.9 with the following versions of the freely-available modules:
- `numpy`: 1.26.4
- `scipy`: 1.14.1
- `sympy`: 1.13.1
- `cython`: 3.0.11

While it is not required to use the exact same versions, it is not quaranteed that the program works properly on other versions.

### Install using the prebuilt wheel
1. Download the prebuild wheel (.whl file). Ensure that the python version and the platform matches your system. For example, `hyppy-1.0-cp311-cp311-win_amd64.whl`, requires Python version 3.11 and Windows. If there are no compatible wheels, the installation must be performed using the source distribution.
2. Install using pip:
    ```bash
    pip install hyppy-1.0-cpXXX-cpXXX-PLATFORM.whl
    ```
### Install using the source distribution
1. Ensure that `build` is installed:
    ```bash
    pip install build
    ```
2. Download the source code (.zip or.tar.gz file).
3. Extract the archive (for example, using 7-Zip).
4. Navigate into the extracted folder:
    ```bash
    cd /your/path/hyppy-1.0
    ```
5. Build the wheel from the extracted source:
    ```bash
    python -m build --wheel
    ```
6. Navigate to the `dist` folder:
    ```bash
    cd /your/path/hyppy-1.0/dist
    ```
7. Install using pip:
    ```bash
    pip install hyppy-1.0-cpXXX-cpXXX-PLATFORM.whl
    ```

## Usage
HypPy contains several modules, which are:
- `basis`
- `chem`
- `data_io`
- `hamiltonian`
- `la`
- `nmr_isotopes`
- `operators`
- `propagation`
- `relaxation`
- `spin_system`
- `states`

These modules can be imported normally. For example, `spin_system` can be imported using:
```python
from hyppy import spin_system
```
Usually it is more convenient to import only the items that are required:
```python
from hyppy.spin_system import SpinSystem
```

### Simple example
We will now go through a simple SABRE example, which can easily be extended for other purposes. This example, as well as more sophisticated examples, can be found from the GitHub repository under `examples`.
1. Import the necessary modules.
    ```python
    import numpy as np
    import matplotlib.pyplot as plt
    from hyppy.spin_system import SpinSystem
    from hyppy.hamiltonian import hamiltonian
    from hyppy.propagation import propagator
    from hyppy.states import singlet, measure
    ```
2. Write down the simulation settings.
    ```python
    magnetic_field = 7e-3
    time_step = 1e-3
    nsteps = 1000
    ```
3. Define the spin system.
    ```python
    isotopes = np.array(['1H', '1H', '1H'])
    chemical_shifts = np.array([-22.7, -22.7, 8.34])
    scalar_couplings = np.array([\
        [ 0,     0,      0],
        [-6.53,  0,      0],
        [ 0.00,  1.66,   0]
    ])
    ```
4. Create the SpinSystem object.
    ```python
    spin_system = SpinSystem(isotopes, chemical_shifts, scalar_couplings)
    ```
5. Calculate the Hamiltonian.
    ```python
    H = hamiltonian(spin_system, magnetic_field)
    ```
6. Calculate the time propagator.
    ```python
    P = propagator(time_step, H)
    ```
7. Create the initial state.
    ```python
    rho = singlet(spin_system, 0, 1)
    ```
8. Create an empty array for storing the result.
    ```python
    magnetizations = np.empty((nsteps, isotopes.size), dtype=complex)
    ```
9. Evolve the spin system, and perform a measurement on every time step.
    ```python
    for step in range(nsteps):
        rho = P @ rho
        for i in range(isotopes.size):
            magnetizations[step, i] = measure(spin_system, rho, 'I_z', i)
    ```
10. Plot the results.
    ```python
    for i in range(isotopes.size):
        plt.plot(np.real(magnetizations[:,i]), label=f"Spin {i+1}")
    plt.legend(loc="upper right")
    plt.xlabel("Time step")
    plt.ylabel("Magnetization")
    plt.title("SABRE-hyperpolarization of Pyridine")
    plt.show()
    plt.clf()
    ```

## License
This project is licensed under the [MIT License](LICENSE).