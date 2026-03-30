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


def status_section(title: str) -> None:
    """
    Begin a clean new section in the status output, with a separator and a title.

    This function is intended to be used for visually separating different stages
    of the computation in the status output when verbose mode is enabled. It
    prints a separator line followed by the provided title, and another
    separator line.

    Parameters
    ----------
    title : str
        Title to be printed for the new section in the status output.
    """

    # Import the global parameters lazily to avoid a circular dependency.
    from spinguin._core._parameters import parameters

    # Print the section header only when status output has been enabled.
    if parameters.verbose:
        print("#" * 80)
        print(title.center(80))
        print("#" * 80)