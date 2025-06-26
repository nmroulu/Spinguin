Installation
============

Spinguin can be installed in various ways, depending on your needs and
preferences. You can install it from the Python Package Index (PyPI) or build it
from the source code available on GitHub. Below are the instructions for both
methods.

Installation from PyPI
----------------------

Spinguin is available from the Python Package Index (PyPI) repository for
Windows and Linux. To install the package, simply issue the command::

    pip install spinguin

Installation from source
------------------------

1. Ensure that the ``build`` module is installed::

    pip install build

2. Download the source code archive from GitHub (.zip or .tar.gz).
3. Extract the archive (e.g., using 7-Zip).
4. Navigate to the extracted folder::

    cd /your/path/spinguin-X.Y.Z

5. Build the wheel from the source::

    python -m build --wheel

6. Navigate to the ``dist`` folder::

    cd /your/path/spinguin-X.Y.Z/dist

7. Install the wheel using pip::

    pip install spinguin-X.Y.Z-cpXYZ-cpXYZ-PLATFORM.whl

