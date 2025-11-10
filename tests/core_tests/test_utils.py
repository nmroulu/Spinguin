import unittest
import spinguin as sg
import numpy as np

class TestUtils(unittest.TestCase):

    def test_idx_conversions(self):
        """
        Test the conversion between index and (l, q) representations.
        """
        # Convert back and forth and compare
        for l in range(10):
            for q in range(-l, l + 1):
                self.assertEqual(sg.idx_to_lq(sg.lq_to_idx(l, q)), (l, q))    
        for idx in range(100):
            self.assertEqual(sg.lq_to_idx(*sg.idx_to_lq(idx)), idx)

    def test_coherence_order(self):
        """
        Test the calculation of coherence orders for given states.
        """
        # Test against states and their known coherence orders
        self.assertEqual(sg.coherence_order(np.array([0])), 0)
        self.assertEqual(sg.coherence_order(np.array([1])), 1)
        self.assertEqual(sg.coherence_order(np.array([2])), 0)
        self.assertEqual(sg.coherence_order(np.array([3])), -1)
        self.assertEqual(sg.coherence_order(np.array([4])), 2)
        self.assertEqual(sg.coherence_order(np.array([5])), 1)
        self.assertEqual(sg.coherence_order(np.array([6])), 0)
        self.assertEqual(sg.coherence_order(np.array([7])), -1)
        self.assertEqual(sg.coherence_order(np.array([8])), -2)
        self.assertEqual(sg.coherence_order(np.array([1, 4])), 3)
        self.assertEqual(
            sg.coherence_order(np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])), 0
        )