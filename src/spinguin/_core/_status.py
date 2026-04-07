"""
Status-message helpers controlled by the global verbosity setting.

This module provides lightweight helper functions for printing status
messages and section headers when verbose output is enabled.
"""


def status(
    msg: str,
) -> None:
    """
    Print a status message when verbose output is enabled.

    Usage
    -----
    ``status(msg)``

    Parameters
    ----------
    msg : str
        Message to print.

    Returns
    -------
    None
        The message is printed only when verbose output is enabled.
    """

    # Import the global parameters lazily to avoid a circular dependency.
    from spinguin._core._parameters import parameters

    # Print the message only when verbose output is enabled.
    if parameters.verbose:
        print(msg)


def status_section(title: str) -> None:
    """
    Print a titled separator for a new section in the status output.

    Usage
    -----
    ``status_section(title)``

    This function is intended for visually separating different stages of the
    computation when verbose output is enabled. It prints a separator line,
    the centred title, and a second separator line.

    Parameters
    ----------
    title : str
        Title to print for the new section in the status output.

    Returns
    -------
    None
        The section header is printed only when verbose output is enabled.
    """

    # Import the global parameters lazily to avoid a circular dependency.
    from spinguin._core._parameters import parameters

    # Print the section header only when verbose output is enabled.
    if parameters.verbose:
        print("#" * 80)
        print(title.center(80))
        print("#" * 80)