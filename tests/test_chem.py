import unittest
from spinguin import _chem, _states
from spinguin._spin_system import SpinSystem
import numpy as np

class TestChemMethods(unittest.TestCase):

    def test_associate(self):

        # Assign isotopes
        isotopes1 = np.array(['1H', '1H'])
        isotopes2 = np.array(['1H', '1H'])
        isotopes3 = np.array(['1H', '1H', '1H', '1H'])

        # Assign indices after association
        spinmap1 = (0, 1)
        spinmap2 = (2, 3)

        # Construct the spin systems
        ss1 = SpinSystem(isotopes1, max_spin_order=2)
        ss2 = SpinSystem(isotopes2, max_spin_order=2)
        ss3 = SpinSystem(isotopes3, max_spin_order=3)

        # Create alpha states
        rho1 = _states.alpha(ss1, 0)
        rho2 = _states.alpha(ss2, 0)

        # Associate
        rho3 = _chem.associate(ss1, ss2, ss3, rho1, rho2, spinmap1, spinmap2)

        # Create the state directly and compare
        self.assertTrue(np.allclose(rho3, _states.triplet_plus(ss3, 0, 2)))

    def test_dissociate(self):
        
        # Assign isotopes
        isotopes1 = np.array(['1H', '1H'])
        isotopes2 = np.array(['1H', '1H'])
        isotopes3 = np.array(['1H', '1H', '1H', '1H'])

        # Assign indices after dissociation
        spinmap1 = (0, 1)
        spinmap2 = (2, 3)

        # Construct the spin systems
        ss1 = SpinSystem(isotopes1, max_spin_order=2)
        ss2 = SpinSystem(isotopes2, max_spin_order=2)
        ss3 = SpinSystem(isotopes3, max_spin_order=3)

        # Create triplet plus state
        rho3 = _states.triplet_plus(ss3, 0, 2)

        # Dissociate
        rho1, rho2 = _chem.dissociate(ss1, ss2, ss3, rho3, spinmap1, spinmap2)

        # Create the states directly and compare
        self.assertTrue(np.allclose(rho1, _states.alpha(ss1, 0)))
        self.assertTrue(np.allclose(rho2, _states.alpha(ss2, 0)))

    def test_rotate_molecule(self):

        # Assign isotopes
        isotopes = np.array(['1H', '1H', '1H'])

        # Create the spin system
        ss = SpinSystem(isotopes)

        # Make an alpha state for the first spin
        rho = _states.alpha(ss, 0)

        # Compare with manual results
        self.assertTrue((_chem.rotate_molecule(ss, rho, (0, 1, 2)) == _states.alpha(ss, 0)).all())
        self.assertTrue((_chem.rotate_molecule(ss, rho, (0, 2, 1)) == _states.alpha(ss, 0)).all())
        self.assertTrue((_chem.rotate_molecule(ss, rho, (1, 0, 2)) == _states.alpha(ss, 1)).all())
        self.assertTrue((_chem.rotate_molecule(ss, rho, (1, 2, 0)) == _states.alpha(ss, 1)).all())
        self.assertTrue((_chem.rotate_molecule(ss, rho, (2, 0, 1)) == _states.alpha(ss, 2)).all())
        self.assertTrue((_chem.rotate_molecule(ss, rho, (2, 1, 0)) == _states.alpha(ss, 2)).all())
