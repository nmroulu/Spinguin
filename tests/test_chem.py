import unittest
from spinguin import _chem, _states
from spinguin._spin_system import SpinSystem
import numpy as np

class TestChemMethods(unittest.TestCase):
    """
    Unit tests for chemical methods in the spinguin library.
    """

    def test_associate(self):
        """
        Test the associate function for combining spin systems.
        """

        # Define isotopes for the spin systems
        isotopes1 = np.array(['1H', '1H'])
        isotopes2 = np.array(['1H', '1H'])
        isotopes3 = np.array(['1H', '1H', '1H', '1H'])

        # Define spin mappings after association
        spinmap1 = (0, 1)
        spinmap2 = (2, 3)

        # Create spin systems
        ss1 = SpinSystem(isotopes1, max_spin_order=2)
        ss2 = SpinSystem(isotopes2, max_spin_order=2)
        ss3 = SpinSystem(isotopes3, max_spin_order=3)

        # Create alpha states for the spin systems
        rho1 = _states.alpha(ss1, 0)
        rho2 = _states.alpha(ss2, 0)

        # Perform association
        rho3 = _chem.associate(ss1, ss2, ss3, rho1, rho2, spinmap1, spinmap2)

        # Create the expected state directly and compare
        self.assertTrue(np.allclose(rho3, _states.triplet_plus(ss3, 0, 2)))

    def test_dissociate(self):
        """
        Test the dissociate function for splitting spin systems.
        """

        # Define isotopes for the spin systems
        isotopes1 = np.array(['1H', '1H'])
        isotopes2 = np.array(['1H', '1H'])
        isotopes3 = np.array(['1H', '1H', '1H', '1H'])

        # Define spin mappings after dissociation
        spinmap1 = (0, 1)
        spinmap2 = (2, 3)

        # Create spin systems
        ss1 = SpinSystem(isotopes1, max_spin_order=2)
        ss2 = SpinSystem(isotopes2, max_spin_order=2)
        ss3 = SpinSystem(isotopes3, max_spin_order=3)

        # Create a triplet plus state for the combined system
        rho3 = _states.triplet_plus(ss3, 0, 2)

        # Perform dissociation
        rho1, rho2 = _chem.dissociate(ss1, ss2, ss3, rho3, spinmap1, spinmap2)

        # Create the expected states directly and compare
        self.assertTrue(np.allclose(rho1, _states.alpha(ss1, 0)))
        self.assertTrue(np.allclose(rho2, _states.alpha(ss2, 0)))

    def test_permute_spins(self):
        """
        Test the rotate_molecule function for permuting spin indices.
        """

        # Define isotopes for the spin system
        isotopes = np.array(['1H', '1H', '1H'])

        # Create the spin system
        ss = SpinSystem(isotopes)

        # Create an alpha state for the first spin
        rho = _states.alpha(ss, 0)

        # Compare the results of rotations with expected states
        self.assertTrue((_chem.permute_spins(ss, rho, (0, 1, 2)) == _states.alpha(ss, 0)).all())
        self.assertTrue((_chem.permute_spins(ss, rho, (0, 2, 1)) == _states.alpha(ss, 0)).all())
        self.assertTrue((_chem.permute_spins(ss, rho, (1, 0, 2)) == _states.alpha(ss, 1)).all())
        self.assertTrue((_chem.permute_spins(ss, rho, (1, 2, 0)) == _states.alpha(ss, 1)).all())
        self.assertTrue((_chem.permute_spins(ss, rho, (2, 0, 1)) == _states.alpha(ss, 2)).all())
        self.assertTrue((_chem.permute_spins(ss, rho, (2, 1, 0)) == _states.alpha(ss, 2)).all())
