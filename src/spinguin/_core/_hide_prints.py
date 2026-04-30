"""
Context manager for temporarily suppressing standard output.

This module provides a small helper that redirects `sys.stdout` to
`os.devnull` within a `with` block.
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
    The implementation redirects `sys.stdout` to `os.devnull` 
    for the duration of the context.
    """

    def __enter__(self):
        """
        Redirect standard output to `os.devnull`.
        """

        # Store the original output stream before redirecting it.
        self.stdout = sys.stdout

        # Open the null-device stream and redirect standard output to it.
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, *_):
        """
        Restore the original standard output stream.
        """

        # Close the temporary null-device stream.
        sys.stdout.close()

        # Restore the original standard output stream.
        sys.stdout = self.stdout