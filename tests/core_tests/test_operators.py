import unittest
import numpy as np
import math
import spinguin as sg

class TestOperators(unittest.TestCase):

    def test_op_S(self):
        """
        Test spin operators (E, Sx, Sy, Sz, Sp, Sm) against hard-coded values 
        and verify commutation relations.
        """

        # Hard-coded operators for different spins
        E = {
            1/2 : np.array([[1, 0],
                            [0, 1]]),
            1 : np.array([[1, 0, 0],
                          [0, 1, 0],
                          [0, 0, 1]]),
            3/2 : np.array([[1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])
        }

        Sx = {
            1/2: np.array([[0, 1/2],
                           [1/2, 0]]),
            1: np.array([[0, 1/math.sqrt(2), 0],
                         [1/math.sqrt(2), 0, 1/math.sqrt(2)],
                         [0, 1/math.sqrt(2), 0]]),
            3/2: np.array([[0, math.sqrt(3)/2, 0, 0],
                           [math.sqrt(3)/2, 0, 1, 0],
                           [0, 1, 0, math.sqrt(3)/2],
                           [0, 0, math.sqrt(3)/2, 0]])
        }

        Sy = {
            1/2: np.array([[0, -1j/2],
                           [1j/2, 0]]),
            1: np.array([[0, 1/(1j*math.sqrt(2)), 0],
                         [-1/(1j*math.sqrt(2)), 0, 1/(1j*math.sqrt(2))],
                         [0, -1/(1j*math.sqrt(2)), 0]]),
            3/2: np.array([[0, math.sqrt(3)/(2j), 0, 0],
                           [-math.sqrt(3)/(2j), 0, -1j, 0],
                           [0, 1j, 0, math.sqrt(3)/(2j)],
                           [0, 0, -math.sqrt(3)/(2j), 0]])
        }

        Sz = {
            1/2: np.array([[1/2, 0],
                           [0, -1/2]]),
            1: np.array([[1, 0, 0],
                         [0, 0, 0],
                         [0, 0, -1]]),
            3/2: np.array([[3/2, 0, 0, 0],
                           [0, 1/2, 0, 0],
                           [0, 0, -1/2, 0],
                           [0, 0, 0, -3/2]])
        }

        Sp = {
            1/2: np.array([[0, 1],
                           [0, 0]]),
            1: np.array([[0, math.sqrt(2), 0],
                         [0, 0, math.sqrt(2)],
                         [0, 0, 0]]),
            3/2: np.array([[0, math.sqrt(3), 0, 0],
                           [0, 0, 2, 0],
                           [0, 0, 0, math.sqrt(3)],
                           [0, 0, 0, 0]])
        }

        Sm = {
            1/2: np.array([[0, 0],
                           [1, 0]]),
            1: np.array([[0, 0, 0],
                         [math.sqrt(2), 0, 0],
                         [0, math.sqrt(2), 0]]),
            3/2: np.array([[0, 0, 0, 0],
                           [math.sqrt(3), 0, 0, 0],
                           [0, 2, 0, 0],
                           [0, 0, math.sqrt(3), 0]])
        }

        # Compare values using the sparse backend
        sg.config.sparse_operator = True
        for spin, op in E.items():
            self.assertTrue(np.allclose(sg.op_E(spin).toarray(), op))
        for spin, op in Sx.items():
            self.assertTrue(np.allclose(sg.op_Sx(spin).toarray(), op))
        for spin, op in Sy.items():
            self.assertTrue(np.allclose(sg.op_Sy(spin).toarray(), op))
        for spin, op in Sz.items():
            self.assertTrue(np.allclose(sg.op_Sz(spin).toarray(), op))
        for spin, op in Sp.items():
            self.assertTrue(np.allclose(sg.op_Sp(spin).toarray(), op))
        for spin, op in Sm.items():
            self.assertTrue(np.allclose(sg.op_Sm(spin).toarray(), op))

        # Compare values using the dense backend
        sg.config.sparse_operator = False
        for spin, op in E.items():
            self.assertTrue(np.allclose(sg.op_E(spin), op))
        for spin, op in Sx.items():
            self.assertTrue(np.allclose(sg.op_Sx(spin), op))
        for spin, op in Sy.items():
            self.assertTrue(np.allclose(sg.op_Sy(spin), op))
        for spin, op in Sz.items():
            self.assertTrue(np.allclose(sg.op_Sz(spin), op))
        for spin, op in Sp.items():
            self.assertTrue(np.allclose(sg.op_Sp(spin), op))
        for spin, op in Sm.items():
            self.assertTrue(np.allclose(sg.op_Sm(spin), op))

        # Define a list of spin quantum numbers to test
        spins = [1/2, 1, 3/2, 2, 5/2, 3, 7/2, 4, 9/2, 5, 11/2]

        # Test the commutation relations using the sparse backend
        sg.config.sparse_operator = True
        for spin in spins:
            self.assertTrue(np.allclose(
                sg.comm(sg.op_E(spin), sg.op_E(spin)).toarray(), 0
            ))
            self.assertTrue(np.allclose(
                sg.comm(sg.op_Sx(spin), sg.op_Sy(spin)).toarray(),
                1j*sg.op_Sz(spin).toarray()
            ))
            self.assertTrue(np.allclose(
                sg.comm(sg.op_Sy(spin), sg.op_Sz(spin)).toarray(),
                1j*sg.op_Sx(spin).toarray()
            ))
            self.assertTrue(np.allclose(
                sg.comm(sg.op_Sz(spin), sg.op_Sx(spin)).toarray(),
                1j*sg.op_Sy(spin).toarray()
            ))
            self.assertTrue(np.allclose(
                sg.comm(sg.op_Sp(spin), sg.op_Sz(spin)).toarray(),
                -sg.op_Sp(spin).toarray()
            ))
            self.assertTrue(np.allclose(
                sg.comm(sg.op_Sm(spin), sg.op_Sz(spin)).toarray(),
                sg.op_Sm(spin).toarray()
            ))

        # Test the commutation relations using the dense backend
        sg.config.sparse_operator = False
        for spin in spins:
            self.assertTrue(np.allclose(
                sg.comm(sg.op_E(spin), sg.op_E(spin)), 0
            ))
            self.assertTrue(np.allclose(
                sg.comm(sg.op_Sx(spin), sg.op_Sy(spin)), 1j*sg.op_Sz(spin)
            ))
            self.assertTrue(np.allclose(
                sg.comm(sg.op_Sy(spin), sg.op_Sz(spin)), 1j*sg.op_Sx(spin)
            ))
            self.assertTrue(np.allclose(
                sg.comm(sg.op_Sz(spin), sg.op_Sx(spin)), 1j*sg.op_Sy(spin)
            ))
            self.assertTrue(np.allclose(
                sg.comm(sg.op_Sp(spin), sg.op_Sz(spin)), -sg.op_Sp(spin)
            ))
            self.assertTrue(np.allclose(
                sg.comm(sg.op_Sm(spin), sg.op_Sz(spin)), sg.op_Sm(spin)
            ))

    def test_op_T(self):
        """
        Test commutation relations for spherical tensor operators.
        """

        # Define a list of spin quantum numbers to test
        spins = [1/2, 1, 3/2, 2, 5/2, 3, 7/2, 4, 9/2, 5, 11/2]

        # Test the commutation relations using the sparse backend
        sg.config.sparse_operator = True
        for spin in spins:

            # Go through all the possible ranks and projections
            for l in range(0, int(2*spin+1)):
                for q in range(-l, l+1):

                    # Check the commutation relations
                    self.assertTrue(np.allclose(
                        sg.comm(sg.op_Sz(spin), sg.op_T(spin, l, q)).toarray(),
                        q*sg.op_T(spin, l, q).toarray()
                    ))
                    self.assertTrue(np.allclose(
                        sg.comm(sg.op_Sx(spin)@sg.op_Sx(spin) + \
                             sg.op_Sy(spin)@sg.op_Sy(spin) + \
                             sg.op_Sz(spin)@sg.op_Sz(spin),
                             sg.op_T(spin, l, q)).toarray(), 0))
                    if not q == -l:
                        self.assertTrue(np.allclose(
                            sg.comm(
                                sg.op_Sm(spin),
                                sg.op_T(spin, l, q)
                            ).toarray(),
                            math.sqrt(l*(l+1) - q*(q-1)) * \
                            sg.op_T(spin, l, q-1).toarray()
                        ))
                    if not q == l:
                        # NOTE: Tolerance of allclose had to be increased
                        self.assertTrue(np.allclose(
                            sg.comm(
                                sg.op_Sp(spin),
                                sg.op_T(spin, l, q)
                            ).toarray(),
                            math.sqrt(l*(l+1) - q*(q+1)) * \
                            sg.op_T(spin, l, q+1).toarray(), atol=1e-7
                        ))

        # Test the commutation relations using the dense backend
        sg.config.sparse_operator = False
        for spin in spins:

            # Go through all the possible ranks and projections
            for l in range(0, int(2*spin+1)):
                for q in range(-l, l+1):

                    # Check the commutation relations
                    self.assertTrue(np.allclose(
                        sg.comm(sg.op_Sz(spin), sg.op_T(spin, l, q)),
                        q*sg.op_T(spin, l, q)
                    ))
                    self.assertTrue(np.allclose(
                        sg.comm(sg.op_Sx(spin)@sg.op_Sx(spin) + \
                                sg.op_Sy(spin)@sg.op_Sy(spin) + \
                                sg.op_Sz(spin)@sg.op_Sz(spin),
                                sg.op_T(spin, l, q)), 0))
                    if not q == -l:
                        self.assertTrue(np.allclose(
                            sg.comm(sg.op_Sm(spin), sg.op_T(spin, l, q)),
                            math.sqrt(l*(l+1) - q*(q-1)) * \
                            sg.op_T(spin, l, q-1)
                        ))
                    if not q == l:
                        self.assertTrue(np.allclose(
                            sg.comm(sg.op_Sp(spin), sg.op_T(spin, l, q)),
                            math.sqrt(l*(l+1) - q*(q+1)) * \
                            sg.op_T(spin, l, q+1)
                        ))

    def test_op_T_coupled(self):
        """
        Test the coupled spherical tensor operators for two spins.
        """
        # Define a list of spin quantum numbers to test
        spins = [1/2, 1, 3/2, 2, 5/2, 3, 7/2, 4, 9/2, 5, 11/2]

        # Using sparse arrays, test the relations given in:
        # Eq. 254-262, Man: Cartesian and Spherical Tensors in NMR Hamiltonian
        sg.config.sparse_operator = True
        for s1 in spins:
            for s2 in spins:

                # Get the two-spin operators
                SxIx = np.kron(sg.op_Sx(s1).toarray(), sg.op_Sx(s2).toarray())
                SxIy = np.kron(sg.op_Sx(s1).toarray(), sg.op_Sy(s2).toarray())
                SxIz = np.kron(sg.op_Sx(s1).toarray(), sg.op_Sz(s2).toarray())
                SyIx = np.kron(sg.op_Sy(s1).toarray(), sg.op_Sx(s2).toarray())
                SyIy = np.kron(sg.op_Sy(s1).toarray(), sg.op_Sy(s2).toarray())
                SyIz = np.kron(sg.op_Sy(s1).toarray(), sg.op_Sz(s2).toarray())
                SzIx = np.kron(sg.op_Sz(s1).toarray(), sg.op_Sx(s2).toarray())
                SzIy = np.kron(sg.op_Sz(s1).toarray(), sg.op_Sy(s2).toarray())
                SzIz = np.kron(sg.op_Sz(s1).toarray(), sg.op_Sz(s2).toarray())

                # Test the relations
                self.assertTrue(np.allclose(
                    sg.op_T_coupled(0, 0, 1, s1, 1, s2).toarray(),
                    -1 / np.sqrt(3) * (SxIx + SyIy + SzIz)
                ))
                self.assertTrue(np.allclose(
                    sg.op_T_coupled(1, 1, 1, s1, 1, s2).toarray(),
                    1 / 2 * (SzIx - SxIz + 1j * (SzIy - SyIz))
                ))
                self.assertTrue(np.allclose(
                    sg.op_T_coupled(1, 0, 1, s1, 1, s2).toarray(),
                    1j / np.sqrt(2) * (SxIy - SyIx)
                ))
                self.assertTrue(np.allclose(
                    sg.op_T_coupled(1, -1, 1, s1, 1, s2).toarray(),
                    1 / 2 * (SzIx - SxIz - 1j * (SzIy - SyIz))
                ))
                self.assertTrue(np.allclose(
                    sg.op_T_coupled(2, 2, 1, s1, 1, s2).toarray(),
                    1 / 2 * (SxIx - SyIy + 1j * (SxIy + SyIx))
                ))
                self.assertTrue(np.allclose(
                    sg.op_T_coupled(2, 1, 1, s1, 1, s2).toarray(),
                    -1 / 2 * (SxIz + SzIx + 1j * (SyIz + SzIy))
                ))
                self.assertTrue(np.allclose(
                    sg.op_T_coupled(2, 0, 1, s1, 1, s2).toarray(),
                    1 / np.sqrt(6) * (-SxIx + 2 * SzIz - SyIy)
                ))
                self.assertTrue(np.allclose(
                    sg.op_T_coupled(2, -1, 1, s1, 1, s2).toarray(),
                    1 / 2 * (SxIz + SzIx - 1j * (SyIz + SzIy))
                ))
                self.assertTrue(np.allclose(
                    sg.op_T_coupled(2, -2, 1, s1, 1, s2).toarray(),
                    1 / 2 * (SxIx - SyIy - 1j * (SxIy + SyIx))
                ))

        # Repeat the test using dense arrays
        sg.config.sparse_operator = False
        for s1 in spins:
            for s2 in spins:

                # Get the two-spin operators
                SxIx = np.kron(sg.op_Sx(s1), sg.op_Sx(s2))
                SxIy = np.kron(sg.op_Sx(s1), sg.op_Sy(s2))
                SxIz = np.kron(sg.op_Sx(s1), sg.op_Sz(s2))
                SyIx = np.kron(sg.op_Sy(s1), sg.op_Sx(s2))
                SyIy = np.kron(sg.op_Sy(s1), sg.op_Sy(s2))
                SyIz = np.kron(sg.op_Sy(s1), sg.op_Sz(s2))
                SzIx = np.kron(sg.op_Sz(s1), sg.op_Sx(s2))
                SzIy = np.kron(sg.op_Sz(s1), sg.op_Sy(s2))
                SzIz = np.kron(sg.op_Sz(s1), sg.op_Sz(s2))

                # Test the relations
                self.assertTrue(np.allclose(
                    sg.op_T_coupled(0, 0, 1, s1, 1, s2),
                    -1 / np.sqrt(3) * (SxIx + SyIy + SzIz)
                ))
                self.assertTrue(np.allclose(
                    sg.op_T_coupled(1, 1, 1, s1, 1, s2),
                    1 / 2 * (SzIx - SxIz + 1j * (SzIy - SyIz))
                ))
                self.assertTrue(np.allclose(
                    sg.op_T_coupled(1, 0, 1, s1, 1, s2),
                    1j / np.sqrt(2) * (SxIy - SyIx)
                ))
                self.assertTrue(np.allclose(
                    sg.op_T_coupled(1, -1, 1, s1, 1, s2),
                    1 / 2 * (SzIx - SxIz - 1j * (SzIy - SyIz))
                ))
                self.assertTrue(np.allclose(
                    sg.op_T_coupled(2, 2, 1, s1, 1, s2),
                    1 / 2 * (SxIx - SyIy + 1j * (SxIy + SyIx))
                ))
                self.assertTrue(np.allclose(
                    sg.op_T_coupled(2, 1, 1, s1, 1, s2),
                    -1 / 2 * (SxIz + SzIx + 1j * (SyIz + SzIy))
                ))
                self.assertTrue(np.allclose(
                    sg.op_T_coupled(2, 0, 1, s1, 1, s2),
                    1 / np.sqrt(6) * (-SxIx + 2 * SzIz - SyIy)
                ))
                self.assertTrue(np.allclose(
                    sg.op_T_coupled(2, -1, 1, s1, 1, s2),
                    1 / 2 * (SxIz + SzIx - 1j * (SyIz + SzIy))
                ))
                self.assertTrue(np.allclose(
                    sg.op_T_coupled(2, -2, 1, s1, 1, s2),
                    1 / 2 * (SxIx - SyIy - 1j * (SxIy + SyIx))
                ))

    def test_operator(self):
        """
        A test for creating the Hilbert-space operators for a spin system.
        """

        # Create a test spin system
        spin_system = sg.SpinSystem(["1H", "14N", "23Na"])

        # Operators to test
        test_opers = ['E', 'x', 'y', 'z', '+', '-']

        # Get the single-spin operators (in dense format)
        sg.config.sparse_operator = False
        opers = {}
        for spin in spin_system.spins:
            opers[('E', spin)] = sg.op_E(spin)
            opers[('x', spin)] = sg.op_Sx(spin)
            opers[('y', spin)] = sg.op_Sy(spin)
            opers[('z', spin)] = sg.op_Sz(spin)
            opers[('+', spin)] = sg.op_Sp(spin)
            opers[('-', spin)] = sg.op_Sm(spin)

        # Try all possible product operator combinations using sparse arrays
        sg.config.sparse_operator = True
        for i in test_opers:
            if i == "E":
                op_i = "E"
            else:
                op_i = f"I({i}, 0)"

            for j in test_opers:
                if j == "E":
                    op_j = "E"
                else:
                    op_j = f"I({j}, 1)"

                for k in test_opers:
                    if k == "E":
                        op_k = "E"
                    else:
                        op_k = f"I({k}, 2)"

                    # Create the operator using inbuilt function
                    op_string = f"{op_i} * {op_j} * {op_k}"
                    oper = sg.operator(spin_system, op_string).toarray()

                    # Create the reference operator
                    oper_ref = np.kron(
                        opers[(i, spin_system.spins[0])], np.kron(
                            opers[(j, spin_system.spins[1])],
                            opers[(k, spin_system.spins[2])]
                        )
                    )

                    # Compare
                    self.assertTrue(np.allclose(oper, oper_ref))
                
        # Try all possible product operator combinations using dense arrays
        sg.config.sparse_operator = False
        for i in test_opers:
            if i == "E":
                op_i = "E"
            else:
                op_i = f"I({i}, 0)"

            for j in test_opers:
                if j == "E":
                    op_j = "E"
                else:
                    op_j = f"I({j}, 1)"

                for k in test_opers:
                    if k == "E":
                        op_k = "E"
                    else:
                        op_k = f"I({k}, 2)"

                    # Create the operator using inbuilt function
                    op_string = f"{op_i} * {op_j} * {op_k}"
                    oper = sg.operator(spin_system, op_string)

                    # Create the reference operator
                    oper_ref = np.kron(
                        opers[(i, spin_system.spins[0])], np.kron(
                            opers[(j, spin_system.spins[1])],
                            opers[(k, spin_system.spins[2])]
                        )
                    )

                    # Compare
                    self.assertTrue(np.allclose(oper, oper_ref))

        # With sparse arrays, test creating a sum operator for all spins using
        # input type 'I(x)'
        sg.config.sparse_operator = True
        for oper in test_opers:
            if oper != "E":      

                op_string = f"I({oper})"
                op_string_ref = f"I({oper},0) + I({oper},1) + I({oper},2)"

                oper = sg.operator(spin_system, op_string).toarray()
                oper_ref = sg.operator(spin_system, op_string_ref).toarray()

                self.assertTrue(np.allclose(oper, oper_ref))

        # With dense arrays, test creating a sum operator for all spins using
        # the input type 'I(x)'
        sg.config.sparse_operator = False
        for oper in test_opers:
            if oper != "E":      

                op_string = f"I({oper})"
                op_string_ref = f"I({oper},0) + I({oper},1) + I({oper},2)"

                oper = sg.operator(spin_system, op_string)
                oper_ref = sg.operator(spin_system, op_string_ref)

                self.assertTrue(np.allclose(oper, oper_ref))

        # With sparse arrays, test creating a sum operator for all spins using
        # the input type T(l,q)
        sg.config.sparse_operator = True
        for l in range(0, 2):
            for q in range(-l, l+1):

                op_string = f"T({l},{q})"
                op_string_ref = f"T({l},{q},0) + T({l},{q},1) + T({l},{q},2)"

                oper = sg.operator(spin_system, op_string).toarray()
                oper_ref = sg.operator(spin_system, op_string_ref).toarray()

                self.assertTrue(np.allclose(oper, oper_ref))

        # With dense arrays, test creating a sum operator for all spins using
        # the input type T(l,q)
        sg.config.sparse_operator = False
        for l in range(0, 2):
            for q in range(-l, l+1):

                op_string = f"T({l},{q})"
                op_string_ref = f"T({l},{q},0) + T({l},{q},1) + T({l},{q},2)"

                oper = sg.operator(spin_system, op_string)
                oper_ref = sg.operator(spin_system, op_string_ref)

                self.assertTrue(np.allclose(oper, oper_ref))
