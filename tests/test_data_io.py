import unittest
import numpy as np
import os
from spinguin import data_io

class TestDataIOMethods(unittest.TestCase):

    def test_read_array(self):

        # Hard-code the values for comparison
        isotopes_1 = np.array(['1H', '19F', '14N'])
        chemical_shifts_1 = np.array([8.00, -127.5, 40.50])
        scalar_couplings_1 = np.array([\
            [0,     0,     0],
            [1.05,  0,     0],
            [0.50,  9.17,  0]
        ])

        # Read values from .txt files
        test_dir = os.path.dirname(__file__)
        isotopes_2 = data_io.read_array(os.path.join(test_dir, 'test_data', 'isotopes.txt'), data_type=str)
        chemical_shifts_2 = data_io.read_array(os.path.join(test_dir, 'test_data', 'chemical_shifts.txt'), data_type=float)
        scalar_couplings_2 = data_io.read_array(os.path.join(test_dir, 'test_data', 'scalar_couplings.txt'), data_type=float)

        # Make a comparison
        self.assertTrue((isotopes_1 == isotopes_2).all())
        self.assertTrue((chemical_shifts_1 == chemical_shifts_2).all())
        self.assertTrue((scalar_couplings_1 == scalar_couplings_2).all())

    def test_read_xyz(self):

        # Hard-code the values for comparison
        xyz_1 = np.array([\
            [1.0527, 2.2566, 0.9925],
            [0.0014, 1.5578, 2.1146],
            [1.3456, 0.3678, 1.4251]
        ])

        # Read values from .txt files
        test_dir = os.path.dirname(__file__)
        xyz_2 = data_io.read_xyz(os.path.join(test_dir, 'test_data', 'xyz.txt'))

        # Make a comparison
        self.assertTrue((xyz_1 == xyz_2).all())
    
    def test_read_tensors(self):

        # Hard-code the values for comparison
        shielding_1 = np.zeros((3, 3, 3))
        shielding_1[1] = np.array([\
            [101.6, -75.2, 11.1],
            [30.5,   10.1, 87.4],
            [99.7,  -21.1, 11.2]
        ])
        shielding_1[2] = np.array([\
            [171.9, -58.6, 91.1],
            [37.5,   10.7, 86.9],
            [109.7, -91.1, 81.8]
        ])
        efg_1 = np.zeros((3, 3, 3))
        efg_1[1] = np.array([\
            [ 0.31, 0.00, 0.01],
            [-0.20, 0.04, 0.87],
            [ 0.11, 0.16, 0.65]
        ])
        efg_1[2] = np.array([\
            [0.34, 0.67, 0.23],
            [0.38, 0.65, 0.26],
            [0.29, 0.82, 0.06]
        ])

        # Read values from .txt files
        test_dir = os.path.dirname(__file__)
        shielding_2 = data_io.read_tensors(os.path.join(test_dir, 'test_data', 'shielding.txt'))
        efg_2 = data_io.read_tensors(os.path.join(test_dir, 'test_data', 'efg.txt'))

        # Make a comparison
        self.assertTrue((shielding_1 == shielding_2).all())
        self.assertTrue((efg_1 == efg_2).all())