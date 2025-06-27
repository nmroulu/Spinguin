"""
This module contains a class that can be used to hide printing to the console.
"""

# Imports
import os
import sys

class HidePrints:
    """
    This class can be used to hide printing to the console. Usage::

        with HidePrints():
            do_something()

    Solution from:
    https://stackoverflow.com/questions/8391411/how-to-block-calls-to-print
    """

    def __enter__(self):
        
        # Disable stdout
        self.stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, *_):

        # Restore the original stdout
        sys.stdout.close()
        sys.stdout = self.stdout