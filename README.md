# Spinguin

## Description
Spinguin is a user-friendly Python package for versatile numerical
spin-dynamics simulations in the liquid state. It provides tools for
coherent dynamics, relaxation, and chemical-exchange simulations, and it
supports restricted basis sets in Liouville space. This makes it possible
to study spin systems with more than 10 spins on consumer-level hardware.

## Documentation
The full documentation is available at
<https://nmroulu.github.io/Spinguin/>.

## Installation
Spinguin can be installed either from the Python Package Index (PyPI) or
from source.

### Installation from PyPI
Spinguin is available on PyPI for Windows and Linux. Install it with:

```bash
pip install spinguin
```

### Installation from source
1. Ensure that the `build` module is installed:

   ```bash
   pip install build
   ```

2. Download the source archive (`.zip` or `.tar.gz`).
3. Extract the archive.
4. Navigate to the extracted source directory:

   ```bash
   cd /your/path/spinguin-X.Y.Z
   ```

5. Build the wheel:

   ```bash
   python -m build --wheel
   ```

6. Navigate to the `dist` directory:

   ```bash
   cd /your/path/spinguin-X.Y.Z/dist
   ```

7. Install the generated wheel with `pip`:

   ```bash
   pip install spinguin-X.Y.Z-cpXYZ-cpXYZ-PLATFORM.whl
   ```

## License
This project is licensed under the [MIT License](LICENSE).