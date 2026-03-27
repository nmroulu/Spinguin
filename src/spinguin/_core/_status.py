"""
status.py

Provides a lightweight helper for printing status messages controlled by the
global verbosity setting.
"""


def status(
    msg: str,
) -> None:
    """
    Print a status message when verbose output is enabled.

    Parameters
    ----------
    msg : str
        Message to be printed.

    Returns
    -------
    None
        The message is printed only when verbose output is enabled.
    """

    # Import the global parameters lazily to avoid a circular dependency.
    from spinguin._core._parameters import parameters

    # Print the message only when status output has been enabled.
    if parameters.verbose:
        print(msg)