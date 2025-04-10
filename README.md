# Spinguin

## Description
Spinguin is a user-friendly Python package designed for versatile numerical spin-dynamics simulations. It offers tools for performing simulations using restricted basis sets, enabling the use of large spin systems with more than 10 spins on consumer-level hardware. Spinguin supports the simulation of coherent dynamics, relaxation, and chemical exchange processes.

## Installation

### Requirements
Spinguin has been developed and tested with Python 3.11.9 and the following module versions:
- `numpy`: 1.26.4
- `scipy`: 1.14.1
- `sympy`: 1.13.1
- `cython`: 3.0.11

While exact versions are not mandatory, compatibility with newer versions is not guaranteed.

### Install using the prebuilt wheel
1. Download the prebuilt wheel (.whl file) matching your Python version and platform. For example, `spinguin-1.0-cp311-cp311-win_amd64.whl` requires Python 3.11 on Windows. If no compatible wheels are available, use the source distribution for installation.
2. Install the wheel using pip:
    ```bash
    pip install spinguin-1.0-cpXXX-cpXXX-PLATFORM.whl
    ```

### Install using the source distribution
1. Ensure the `build` module is installed:
    ```bash
    pip install build
    ```
2. Download the source code archive (.zip or .tar.gz).
3. Extract the archive (e.g., using 7-Zip).
4. Navigate to the extracted folder:
    ```bash
    cd /your/path/spinguin-1.0
    ```
5. Build the wheel from the source:
    ```bash
    python -m build --wheel
    ```
6. Navigate to the `dist` folder:
    ```bash
    cd /your/path/spinguin-1.0/dist
    ```
7. Install the wheel using pip:
    ```bash
    pip install spinguin-1.0-cpXXX-cpXXX-PLATFORM.whl
    ```

## Usage
Spinguin's functionality is organized into several modules:
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

Modules can be imported as needed. For example, to import the `spin_system` module:
```python
from spinguin import spin_system
```
Alternatively, import specific items:
```python
from spinguin.spin_system import SpinSystem
```

### Simple Example
Below is a basic SABRE example, which can be extended for other purposes. Additional examples are available in the GitHub repository under `examples`.

1. Import the required modules:
    ```python
    import numpy as np
    import matplotlib.pyplot as plt
    from spinguin.spin_system import SpinSystem
    from spinguin.hamiltonian import hamiltonian
    from spinguin.propagation import propagator
    from spinguin.states import singlet, measure
    ```
2. Define simulation settings:
    ```python
    magnetic_field = 7e-3
    time_step = 1e-3
    nsteps = 1000
    ```
3. Define the spin system:
    ```python
    isotopes = np.array(['1H', '1H', '1H'])
    chemical_shifts = np.array([-22.7, -22.7, 8.34])
    J_couplings = np.array([\
        [ 0,     0,      0],
        [-6.53,  0,      0],
        [ 0.00,  1.66,   0]
    ])
    ```
4. Create a `SpinSystem` object:
    ```python
    spin_system = SpinSystem(isotopes, chemical_shifts, J_couplings)
    ```
5. Calculate the Hamiltonian:
    ```python
    H = hamiltonian(spin_system, magnetic_field)
    ```
6. Compute the time propagator:
    ```python
    P = propagator(time_step, H)
    ```
7. Define the initial state:
    ```python
    rho = singlet(spin_system, 0, 1)
    ```
8. Initialize an array to store results:
    ```python
    magnetizations = np.empty((nsteps, isotopes.size), dtype=complex)
    ```
9. Evolve the spin system and measure at each time step:
    ```python
    for step in range(nsteps):
        rho = P @ rho
        for i in range(isotopes.size):
            magnetizations[step, i] = measure(spin_system, rho, 'I_z', i)
    ```
10. Plot the results:
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