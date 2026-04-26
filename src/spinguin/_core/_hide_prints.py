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

        Returns
        -------
        HidePrints
            The active context manager instance.
        """

        # Store the original output stream before redirecting it.
        self.stdout = sys.stdout

        # Open the null-device stream and redirect standard output to it.
        self._null_stream = open(os.devnull, 'w', encoding='utf-8')
        sys.stdout = self._null_stream

        return self

    def __exit__(self, *_):
        """
        Restore the original standard output stream.

        Returns
        -------
        None
        """

        # Close the temporary null-device stream.
        self._null_stream.close()

        # Restore the original standard output stream.
        sys.stdout = self.stdout