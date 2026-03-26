"""
hide_prints.py

Provides a small context manager for temporarily redirecting standard output to
`os.devnull`.
"""

# Imports
import os
import sys

class HidePrints:
    """
    Temporarily suppress printing to standard output.

    Usage::

        with HidePrints():
            do_something()

    Notes
    -----
    The implementation redirects `sys.stdout` to `os.devnull` for the duration
    of the context.
    """

    def __enter__(self):
        """
        Redirect standard output to `os.devnull`.

        Returns
        -------
        HidePrints
            The active context manager instance.
        """

        # Store the original output stream before redirecting it.
        self.stdout = sys.stdout

        # Redirect standard output to the null device.
        sys.stdout = open(os.devnull, 'w')

        return self

    def __exit__(self, *_):
        """
        Restore the original standard output stream.

        Returns
        -------
        None
        """

        # Close the temporary null-device stream.
        sys.stdout.close()

        # Restore the original standard output stream.
        sys.stdout = self.stdout