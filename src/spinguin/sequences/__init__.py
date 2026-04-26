"""
Ready-to-use pulse-sequences in Spinguin.

This package exposes the public sequence constructors that implement common
NMR simulation workflows.
"""

from ._inversion_recovery import inversion_recovery
from ._pulse_and_acquire import pulse_and_acquire

__all__ = [
    "inversion_recovery",
    "pulse_and_acquire",
]