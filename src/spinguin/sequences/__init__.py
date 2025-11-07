"""
This is a subpackage of Spinguin that contains ready-to-use pulse sequences.
"""

from ._inversion_recovery import inversion_recovery
from ._pulse_and_acquire import pulse_and_acquire

__all__ = [
    "inversion_recovery",
    "pulse_and_acquire",
]