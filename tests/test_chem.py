import unittest
from spinguin.qm import states
from spinguin.qm import chem
from spinguin.system.spin_system import SpinSystem
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
        rho1 = states.alpha_state(ss1, 0)
        rho2 = states.alpha_state(ss2, 0)

        # Perform association
        rho3 = chem.associate(ss1, ss2, ss3, rho1, rho2, spinmap1, spinmap2)

        # Create the expected state directly and compare
        self.assertTrue(np.allclose(rho3, states.triplet_plus_state(ss3, 0, 2)))

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
        rho3 = states.triplet_plus_state(ss3, 0, 2)

        # Perform dissociation
        rho1, rho2 = chem.dissociate(ss1, ss2, ss3, rho3, spinmap1, spinmap2)

        # Create the expected states directly and compare
        self.assertTrue(np.allclose(rho1, states.alpha_state(ss1, 0)))
        self.assertTrue(np.allclose(rho2, states.alpha_state(ss2, 0)))

    def test_permute_spins(self):
        """
        Test the rotate_molecule function for permuting spin indices.
        """

        # Define isotopes for the spin system
        isotopes = np.array(['1H', '1H', '1H'])

        # Create the spin system
        ss = SpinSystem(isotopes)

        # Create an alpha state for the first spin
        rho = states.alpha_state(ss, 0)

        # Compare the results of rotations with expected states
        self.assertTrue((chem.permute_spins(ss, rho, (0, 1, 2)) == states.alpha_state(ss, 0)).all())
        self.assertTrue((chem.permute_spins(ss, rho, (0, 2, 1)) == states.alpha_state(ss, 0)).all())
        self.assertTrue((chem.permute_spins(ss, rho, (1, 0, 2)) == states.alpha_state(ss, 1)).all())
        self.assertTrue((chem.permute_spins(ss, rho, (1, 2, 0)) == states.alpha_state(ss, 1)).all())
        self.assertTrue((chem.permute_spins(ss, rho, (2, 0, 1)) == states.alpha_state(ss, 2)).all())
        self.assertTrue((chem.permute_spins(ss, rho, (2, 1, 0)) == states.alpha_state(ss, 2)).all())
