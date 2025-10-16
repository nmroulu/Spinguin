"""
This module provides the core functionality of Spinguin which is not dependent
on any inbuilt class. Therefore, these functions can be reused in any context
elsewhere. These can be accessed using::

    # Replace module and function with something
    from spinguin.core.module.function

The preferred way to use Spinguin is to use the functionality under `spinguin`
namespace, described in Spinguin (Basic), using::

    import spinguin as sg
"""

# Imports
from spinguin._core._parameters import parameters
from spinguin._core._spin_system import SpinSystem

__all__ = [
    # spinguin._core._parameters
    "parameters",

    # spinguin._core._spin_system
    "SpinSystem",
]