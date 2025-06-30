# Spinguin

## Description
Spinguin is a user-friendly Python package designed for versatile numerical
spin-dynamics simulations in liquid state. It offers tools for performing
simulations using restricted basis sets, enabling the use of large spin systems
with more than 10 spins on consumer-level hardware. Spinguin supports the
simulation of coherent dynamics, relaxation, and chemical exchange processes.

## Documentation
Documentation for the package is available here:
https://nmroulu.github.io/Spinguin/

## Installation
Spinguin can be installed in various ways, depending on your needs and
preferences. You can install it from the Python Package Index (PyPI) or build it
from the source code available on GitHub. Below are the instructions for both
methods.

### Installation from PyPI
Spinguin is available from the Python Package Index (PyPI) repository for
Windows and Linux. To install the package, simply issue the command::

```bash
pip install spinguin
```

### Installation from source
1. Ensure the `build` module is installed:
    ```bash
    pip install build
    ```
2. Download the source code archive (.zip or .tar.gz).
3. Extract the archive (e.g., using 7-Zip).
4. Navigate to the extracted folder:
    ```bash
    cd /your/path/spinguin-X.Y.Z
    ```
5. Build the wheel from the source:
    ```bash
    python -m build --wheel
    ```
6. Navigate to the `dist` folder:
    ```bash
    cd /your/path/spinguin-X.Y.Z/dist
    ```
7. Install the wheel using pip:
    ```bash
    pip install spinguin-X.Y.Z-cpXYZ-cpXYZ-PLATFORM.whl
    ```

## License
This project is licensed under the [MIT License](LICENSE).