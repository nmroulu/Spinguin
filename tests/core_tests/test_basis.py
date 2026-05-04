"""
Tests for basis-set construction and truncation in the Spinguin core.
"""

import math
import unittest
from copy import deepcopy

import numpy as np

import spinguin as sg

class TestBasis(unittest.TestCase):
    """
    Test basis-set generation and basis-set truncation utilities.
    """

    def test_make_basis_1(self):
        """
        Test basis-set construction against a hard-coded reference.
        """

        # Define the reference basis set explicitly.
        basis_ref = np.array([
            [0, 0],
            [0, 1],
            [0, 2],
            [0, 3],
            [1, 0],
            [2, 0],
            [3, 0]
        ])
        
        # Build the basis set for the test system.
        spin_system = sg.SpinSystem(['1H', '1H'])
        spin_system.basis.max_spin_order = 1
        spin_system.basis.build()

        # Compare the generated basis set with the reference result.
        self.assertTrue(np.array_equal(spin_system.basis.basis, basis_ref))

    def test_make_basis_2(self):
        """
        Test basis-set dimensions.
        """

        # Create the test system.
        spin_system = sg.SpinSystem(['1H', '1H', '1H', '1H'])
        nspins = spin_system.nspins

        # Compare the basis dimension with the combinatorial reference value.
        for max_so in range(1, nspins):
            spin_system.basis.max_spin_order = max_so
            spin_system.basis.build()
            dim_ref = sum(
                math.comb(nspins, k) * 3**k for k in range(max_so + 1)
            )
            self.assertEqual(spin_system.basis.dim, dim_ref)

    def test_state_idx(self):
        """
        Test the mapping of states to their corresponding indices.
        """

        # Build the basis set for the test system.
        spin_system = sg.SpinSystem(["14N", "1H", "1H"])
        spin_system.basis.max_spin_order = 2
        spin_system.basis.build()

        # Check state indexing for multiple supported input types.
        self.assertEqual(spin_system.basis.indexof([0, 1, 0]), 4)
        self.assertEqual(spin_system.basis.indexof((1, 0, 3)), 19)
        self.assertEqual(spin_system.basis.indexof(np.array([8, 0, 3])), 68)

        # Check that a non-existent state raises an error.
        with self.assertRaises(ValueError):
            spin_system.basis.indexof([9, 9, 9])

        # Check that an undersized state vector raises an error.
        with self.assertRaises(ValueError):
            spin_system.basis.indexof([0])

        # Check that an oversized state vector raises an error.
        with self.assertRaises(ValueError):
            spin_system.basis.indexof([0, 1, 2, 0])

    def test_truncate_basis_by_coherence(self):
        """
        Test the creation of the truncated basis using coherence order as the
        selection criterion.
        """
        # Example system
        spin_system = sg.SpinSystem(['1H', '1H', '1H', '1H', '14N'])

        # Create a basis set with all coherence orders 
        spin_system.basis.max_spin_order = 4
        spin_system.basis.build()

        # Save the original basis set for further testing
        basis_org = spin_system.basis.basis.copy()

        # Create a superoperator and a state in the full basis set
        oper_org = sg.superoperator(spin_system, "I(z,0) * I(+,1) * I(-,2)")
        state_org = sg.state(spin_system, "I(+,1) * I(z,3) * I(-,4)")

        # Truncate the basis (retain coherence orders of -2, 0, and 1)
        # Obtain also the superoperator and the state in the truncated basis
        coherence_orders = [-2, 0, 1]
        oper_org_tr, state_org_tr = spin_system.basis.truncate_by_coherence(
            coherence_orders, oper_org, state_org
        )

        # Obtain the superoperator and state directly in the truncated basis
        oper_tr = sg.superoperator(spin_system, "I(z,0) * I(+,1) * I(-,2)")
        state_tr = sg.state(spin_system, "I(+,1) * I(z,3) * I(-,4)")

        # Verify that the superoperators and states are equal
        self.assertTrue(np.allclose(oper_org_tr.toarray(), oper_tr.toarray()))
        self.assertTrue(np.allclose(state_org_tr, state_tr))

        # Check that only the coherence orders [-2, 0, 1] remain in the basis
        for op_def in basis_org:

            # Should not raise an error
            if sg.coherence_order(op_def) in coherence_orders:
                spin_system.basis.indexof(op_def)

            # Raises an error
            else:
                with self.assertRaises(ValueError):
                    spin_system.basis.indexof(op_def)

    def test_truncate_basis_by_coupling_1(self):
        """
        Test truncating the basis set by coupling.
        """
        # Example system
        spin_system = sg.SpinSystem(['1H', '1H', '1H'])

        # Assign J-couplings
        spin_system.J_couplings = [
            [0, 0, 0],
            [1, 0, 0],
            [1, 0, 0]
        ]

        # Create a basis set with all coherence orders 
        spin_system.basis.max_spin_order = 3
        spin_system.basis.build()

        # Save the original basis set for further testing
        basis_org = spin_system.basis.basis.copy()

        # Truncate the basis set
        spin_system.basis.truncate_by_coupling()

        # Test each state in the original basis
        for op_def in basis_org:

            # Determine whether a state should exist in the basis
            deleted = op_def[0] == 0 and op_def[1] != 0 and op_def[2] != 0 

            # Check if the state is deleted as it should be
            if deleted:
                with self.assertRaises(ValueError):
                    spin_system.basis.indexof(op_def)

            # Otherwise the state should be in the basis set (no Error)
            else:
                spin_system.basis.indexof(op_def)

    def test_truncate_basis_by_coupling_2(self):
        """
        Test truncating the basis set by coupling.
        """
        # Example system
        spin_system = sg.SpinSystem(['1H', '1H', '1H'])

        # Assign XYZ
        spin_system.xyz = [
            [0, 0, 0],
            [1, 1, 1],
            [99, 99, 99]
        ]

        # Create a basis set with all coherence orders 
        spin_system.basis.max_spin_order = 3
        spin_system.basis.build()

        # Save the original basis set for further testing
        basis_org = spin_system.basis.basis.copy()

        # Truncate the basis set
        spin_system.basis.truncate_by_coupling()

        # Test each state in the original basis
        for op_def in basis_org:

            # Determine whether a state should exist in the basis
            deleted = (op_def[0] != 0 or op_def[1] != 0) and op_def[2] != 0 

            # Check if the state is deleted as it should be
            if deleted:
                with self.assertRaises(ValueError):
                    spin_system.basis.indexof(op_def)

            # Otherwise the state should be in the basis set (no Error)
            else:
                spin_system.basis.indexof(op_def)

    def test_truncate_basis_by_coupling_3(self):
        """
        Test truncating the basis set by coupling.
        """
        # Example system
        spin_system = sg.SpinSystem(['1H', '1H', '1H'])

        # Assign J-couplings (spins 1 and 3 are coupled)
        spin_system.J_couplings = [
            [0, 0, 0],
            [0, 0, 0],
            [1, 0, 0]
        ]

        # Assign XYZ (spins 1 and 2 are coupled)
        spin_system.xyz = [
            [0, 0, 0],
            [1, 1, 1],
            [99, 99, 99]
        ]

        # Create a basis set with all coherence orders 
        spin_system.basis.max_spin_order = 3
        spin_system.basis.build()

        # Save the original basis set for further testing
        basis_org = spin_system.basis.basis.copy()

        # Truncate the basis set
        spin_system.basis.truncate_by_coupling()

        # Test each state in the original basis
        for op_def in basis_org:

            # Determine whether a state should exist in the basis
            deleted = op_def[0] == 0 and op_def[1] != 0 and op_def[2] != 0 

            # Check if the state is deleted as it should be
            if deleted:
                with self.assertRaises(ValueError):
                    spin_system.basis.indexof(op_def)

            # Otherwise the state should be in the basis set (no Error)
            else:
                spin_system.basis.indexof(op_def)

    def test_truncate_basis_by_zte(self):
        """
        Test the basis set truncation using ZTE by comparing the generated FID
        to the exact solution.
        """
        # Set the global parameters
        sg.parameters.default()
        sg.parameters.magnetic_field = 1
        sg.parameters.temperature = 295

        # Make the spin system
        ss = sg.SpinSystem(["1H", "1H", "1H", "1H", "1H", "14N"])

        # Create the basis set
        ss.basis.max_spin_order = 3
        ss.basis.build()

        # Define the chemical shifts (in ppm)
        ss.chemical_shifts = [8.56, 8.56, 7.47, 7.47, 7.88, 95.94]

        # Define scalar couplings (in Hz)
        ss.J_couplings = [
            [ 0,     0,      0,      0,      0,      0],
            [-1.04,  0,      0,      0,      0,      0],
            [ 4.85,  1.05,   0,      0,      0,      0],
            [ 1.05,  4.85,   0.71,   0,      0,      0],
            [ 1.24,  1.24,   7.55,   7.55,   0,      0],
            [ 8.16,  8.16,   0.87,   0.87,  -0.19,   0]
        ]

        # Define Cartesian coordinates of nuclei
        ss.xyz = [
            [ 2.0495335, 0.0000000, -1.4916842],
            [-2.0495335, 0.0000000, -1.4916842],
            [ 2.1458878, 0.0000000,  0.9846086],
            [-2.1458878, 0.0000000,  0.9846086],
            [ 0.0000000, 0.0000000,  2.2681296],
            [ 0.0000000, 0.0000000, -1.5987077]
        ]

        # Define shielding tensors
        shielding = np.zeros((6, 3, 3))
        shielding[5] = np.array([
            [-406.20, 0.00,   0.00],
            [ 0.00,   299.44, 0.00],
            [ 0.00,   0.00,  -181.07]
        ])
        ss.shielding = shielding

        # Define electric field gradient tensors
        efg = np.zeros((6, 3, 3))
        efg[5] = np.array([
            [0.3069, 0.0000,  0.0000],
            [0.0000, 0.7969,  0.0000],
            [0.0000, 0.0000, -1.1037]
        ])
        ss.efg = efg

        # Define the relaxation theory
        ss.relaxation.thermalization = True
        ss.relaxation.theory = "redfield"
        ss.relaxation.tau_c = 50e-12

        # Dwell time and number of points
        dt = 1e-3
        npoints = 1000

        # Get the Hamiltonian
        H = sg.hamiltonian(ss)
        
        # Get the Redfield relaxation superoperator
        R = sg.relaxation(ss)

        # Get the Liouvillian
        L = sg.liouvillian(H, R)

        # Transform the Liouvillian to the rotating frame
        L = sg.rotating_frame(ss, L, ["1H", "14N"], [8, 96])

        # Get the thermal equilibrium state
        rho = sg.equilibrium_state(ss)

        # Apply pulse for protons
        indices = np.where(ss.isotopes == "1H")[0]
        op_pulse = "+".join(f"I(y,{i})" for i in indices)
        Px = sg.pulse(ss, op_pulse, 90)
        rho = Px @ rho

        # Truncate the basis set to single-quantum coherence
        L, rho = ss.basis.truncate_by_coherence([1, -1], L, rho)

        # Truncate the basis set using ZTE (make new spin system for this)
        ss_ZTE = deepcopy(ss)
        L_ZTE, rho_ZTE = ss_ZTE.basis.truncate_by_zte(L, rho)

        # Get the time propagators
        P = sg.propagator(L, dt)
        P_ZTE = sg.propagator(L_ZTE, dt)

        # Construct the operator to be measured
        op_measure = "+".join(f"I(-,{i})" for i in indices)

        # Initialize an array for storing results
        fid = np.zeros(npoints, dtype=complex)
        fid_ZTE = np.zeros(npoints, dtype=complex)

        # Perform the time evolution
        for step in range(npoints):
            fid[step] = sg.measure(ss, rho, op_measure)
            fid_ZTE[step] = sg.measure(ss_ZTE, rho_ZTE, op_measure)
            rho = P @ rho
            rho_ZTE = P_ZTE @ rho_ZTE

        # Check that the FIDs match
        self.assertTrue(np.allclose(fid, fid_ZTE))

    def test_truncate_basis_by_indices(self):
        """
        Test the basis set truncation by indices.
        """
        # Example system
        spin_system = sg.SpinSystem(['1H', '1H', '1H'])

        # Create a basis set
        spin_system.basis.max_spin_order = 3
        spin_system.basis.build()

        # Make a copy of the spin system
        spin_system_org = deepcopy(spin_system)

        # Truncate the basis set by retaining only a set of states
        retained_indices = [0, 7, spin_system.basis.dim-1]
        spin_system.basis.truncate_by_indices(retained_indices)

        # Test each state in the original basis
        for op_def in spin_system_org.basis.basis:

            # Determine whether a state should exist in the basis
            idx = spin_system_org.basis.indexof(op_def)
            deleted = idx not in retained_indices

            # Check if the state is deleted as it should be
            if deleted:
                with self.assertRaises(ValueError):
                    spin_system.basis.indexof(op_def)

            # Otherwise the state should be in the basis set (no Error)
            else:
                spin_system.basis.indexof(op_def)