"""
Helper functions for writing the unit tests.
"""

import spinguin as sg

def build_spin_system(
    isotopes: list[str],
    max_spin_order: int
) -> sg.SpinSystem:
    """
    Create a spin system and build its basis set.

    Parameters
    ----------
    isotopes : list of str
        Isotope labels of the spin system.
    max_spin_order : int
        Maximum spin order used to build the basis set.

    Returns
    -------
    SpinSystem
        Spin system with a built basis set.
    """

    # Create the spin system used in the test.
    spin_system = sg.SpinSystem(isotopes)

    # Build the basis set with the requested truncation level.
    spin_system.basis.max_spin_order = max_spin_order
    spin_system.basis.build()

    return spin_system