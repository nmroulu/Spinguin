"""
Spinguin package is desined to be imported as `import spinguin as sg`, which
reveals the user-friendly functionality to the user. For example, to create a 
SpinSystem object with one 1H nucleus, the user should::

    import spinguin as sg
    spin_system = sg.SpinSystem(["1H"])
"""

# Make the core functionality directly available under the spinguin namespace
from spinguin._core import *