"""
This module provides a status function that is used to print the status messages
to the console in the program.
"""

def status(msg: str):
    """
    Prints the given input to the console if status messages are enabled in the
    global parameters.

    Parameters
    ----------
    msg: str
        Message to be printed.
    """
    # Avoid circular import
    from spinguin._core._parameters import parameters

    # Pring if status messages are enabled
    if parameters.verbose == True:
        print(msg)