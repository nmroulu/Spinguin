"""
Tests for utility functions related to index conversion and coherence order.
"""

import unittest

import numpy as np

import spinguin as sg


class TestUtils(unittest.TestCase):
    """
    Test helper utilities.
    """

    def test_idx_conversions(self):
        """
        Test the conversion between index and (l, q) representations.
        """
        # Convert `(l, q)` indices to flat indices and back again.
        for l in range(10):
            for q in range(-l, l + 1):
                self.assertEqual(sg.idx_to_lq(sg.lq_to_idx(l, q)), (l, q))
        
        # Convert flat indices to `(l, q)` form and back again.
        for idx in range(100):
            self.assertEqual(sg.lq_to_idx(*sg.idx_to_lq(idx)), idx)

    def test_coherence_order(self):
        """
        Test the calculation of coherence orders for given states.
        """

        # Define reference states and their expected coherence orders.
        test_cases = [
            (np.array([0]), 0),
            (np.array([1]), 1),
            (np.array([2]), 0),
            (np.array([3]), -1),
            (np.array([4]), 2),
            (np.array([5]), 1),
            (np.array([6]), 0),
            (np.array([7]), -1),
            (np.array([8]), -2),
            (np.array([1, 4]), 3),
            (np.array([0, 1, 2, 3, 4, 5, 6, 7, 8]), 0),
        ]

        # Compare the calculated coherence orders with the known values.
        for state, reference_order in test_cases:
            self.assertEqual(sg.coherence_order(state), reference_order)