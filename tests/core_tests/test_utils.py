import unittest
import numpy as np
import spinguin as sg
from scipy.sparse import random_array

class TestUtils(unittest.TestCase):

    def test_sparse_bytes(self):

        # Create a large random array
        A = random_array((1000, 1000), density=0.5, format='csc')

        # Convert to byte representation
        A_bytes = sg.utils.sparse_to_bytes(A)
        
        # Convert back to sparse representation
        B = sg.utils.bytes_to_sparse(A_bytes)

        # Compare the arrays
        self.assertTrue(np.allclose(A.toarray(), B.toarray()))