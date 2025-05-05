import unittest
import numpy as np
import math
from spinguin.system.spin_system import SpinSystem
from spinguin.utils.la import comm
from spinguin.system.basis import idx_to_lq
from spinguin.qm.operators import op_E, op_Sx, op_Sy, op_Sz, op_Sp, op_Sm, op_T, op_prod, operator, op_T_coupled

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

        # Compare values
        for spin, op in E.items():
            self.assertTrue(np.allclose(op_E(spin, sparse=False), op))
            self.assertTrue(np.allclose(op_E(spin, sparse=True).toarray(), op))
        for spin, op in Sx.items():
            self.assertTrue(np.allclose(op_Sx(spin, sparse=False), op))
            self.assertTrue(np.allclose(op_Sx(spin, sparse=True).toarray(), op))
        for spin, op in Sy.items():
            self.assertTrue(np.allclose(op_Sy(spin, sparse=False), op))
            self.assertTrue(np.allclose(op_Sy(spin, sparse=True).toarray(), op))
        for spin, op in Sz.items():
            self.assertTrue(np.allclose(op_Sz(spin, sparse=False), op))
            self.assertTrue(np.allclose(op_Sz(spin, sparse=True).toarray(), op))
        for spin, op in Sp.items():
            self.assertTrue(np.allclose(op_Sp(spin, sparse=False), op))
            self.assertTrue(np.allclose(op_Sp(spin, sparse=True).toarray(), op))
        for spin, op in Sm.items():
            self.assertTrue(np.allclose(op_Sm(spin, sparse=False), op))
            self.assertTrue(np.allclose(op_Sm(spin, sparse=True).toarray(), op))

        # Test the commutation relations and confirm that sparse and dense give same result
        spins = [1/2, 1, 3/2, 2, 5/2, 3, 7/2, 4, 9/2, 5, 11/2]
        for spin in spins:

            # Confirm that sparse and dense give same result
            self.assertTrue(np.allclose(op_E(spin, sparse=False), op_E(spin, sparse=True).toarray()))
            self.assertTrue(np.allclose(op_Sx(spin, sparse=False), op_Sx(spin, sparse=True).toarray()))
            self.assertTrue(np.allclose(op_Sy(spin, sparse=False), op_Sy(spin, sparse=True).toarray()))
            self.assertTrue(np.allclose(op_Sz(spin, sparse=False), op_Sz(spin, sparse=True).toarray()))
            self.assertTrue(np.allclose(op_Sp(spin, sparse=False), op_Sp(spin, sparse=True).toarray()))
            self.assertTrue(np.allclose(op_Sm(spin, sparse=False), op_Sm(spin, sparse=True).toarray()))

            # Test commutation relations
            self.assertTrue(np.allclose(comm(op_E(spin, sparse=False), op_E(spin, sparse=False)), 0))
            self.assertTrue(np.allclose(comm(op_Sx(spin, sparse=False), op_Sy(spin, sparse=False)), 1j*op_Sz(spin, sparse=False)))
            self.assertTrue(np.allclose(comm(op_Sy(spin, sparse=False), op_Sz(spin, sparse=False)), 1j*op_Sx(spin, sparse=False)))
            self.assertTrue(np.allclose(comm(op_Sz(spin, sparse=False), op_Sx(spin, sparse=False)), 1j*op_Sy(spin, sparse=False)))
            self.assertTrue(np.allclose(comm(op_Sp(spin, sparse=False), op_Sz(spin, sparse=False)), -op_Sp(spin, sparse=False)))
            self.assertTrue(np.allclose(comm(op_Sm(spin, sparse=False), op_Sz(spin, sparse=False)), op_Sm(spin, sparse=False)))

    def test_op_T(self):
        """
        Test commutation relations for spherical tensor operators.
        """

        # Test for various spin quantum numbers
        spins = [1/2, 1, 3/2, 2, 5/2, 3, 7/2, 4, 9/2, 5, 11/2]
        for spin in spins:

            # Go through all the possible ranks and projections
            for l in range(0, int(2*spin+1)):
                for q in range(-l, l+1):

                    # Check that sparse and dense representations give same result
                    self.assertTrue(np.allclose(op_T(spin, l, q, sparse=False), op_T(spin, l, q, sparse=True).toarray()))

                    # Check the commutation relations using dense arrays
                    self.assertTrue(np.allclose(comm(op_Sz(spin, sparse=False), op_T(spin, l, q, sparse=False)),
                                                q*op_T(spin, l, q, sparse=False)))
                    self.assertTrue(np.allclose(comm(op_Sx(spin, sparse=False)@op_Sx(spin, sparse=False) + \
                                                     op_Sy(spin, sparse=False)@op_Sy(spin, sparse=False) + \
                                                     op_Sz(spin, sparse=False)@op_Sz(spin, sparse=False),
                                                     op_T(spin, l, q, sparse=False)), 0))
                    if not q == -l:
                        self.assertTrue(np.allclose(comm(op_Sm(spin, sparse=False), op_T(spin, l, q, sparse=False)),
                                                    math.sqrt(l*(l+1) - q*(q-1)) * op_T(spin, l, q-1, sparse=False)))
                    if not q == l:
                        self.assertTrue(np.allclose(comm(op_Sp(spin, sparse=False), op_T(spin, l, q, sparse=False)),
                                                    math.sqrt(l*(l+1) - q*(q+1)) * op_T(spin, l, q+1, sparse=False)))

    def test_op_prod(self):
        """
        Test the construction of product operators for different spin quantum numbers.
        """

        # Create a test spin system
        spins = (1/2, 1, 3/2)
        nspins = len(spins)

        # # Get the Zeeman eigenbasis operators
        # opers = {}
        # for spin in spins:
        #     opers[('E', spin)] = op_E(spin)
        #     opers[('x', spin)] = op_Sx(spin)
        #     opers[('y', spin)] = op_Sy(spin)
        #     opers[('z', spin)] = op_Sz(spin)
        #     opers[('+', spin)] = op_Sp(spin)
        #     opers[('-', spin)] = op_Sm(spin)

        # Try all possible product operator combinations
        for i in range(int(2*spins[0]+1)):
            l_i, q_i = idx_to_lq(i)
            op_i = op_T(spins[0], l_i, q_i, sparse=False)

            for j in range(int(2*spins[1]+1)):
                l_j, q_j = idx_to_lq(j)
                op_j = op_T(spins[1], l_j, q_j, sparse=False)

                for k in range(int(2*spins[2]+1)):
                    l_k, q_k = idx_to_lq(k)
                    op_k = op_T(spins[2], l_k, q_k, sparse=False)

                    # Perform the comparison WITH unit operators included

                    # Create the operator using inbuilt function 
                    op_def = (i, j, k)
                    oper_sparse = op_prod(op_def, spins, include_unit=True, sparse=True)
                    oper_dense = op_prod(op_def, spins, include_unit=True, sparse=False)

                    # Create the reference operator manually
                    oper_ref = np.kron(op_i, np.kron(op_j, op_k))

                    # Compare
                    self.assertTrue(np.allclose(oper_sparse.toarray(), oper_ref))
                    self.assertTrue(np.allclose(oper_dense, oper_ref))

                    # Perform the comparison WITHOUT unit operators included

                    # Create the operator using inbuilt function
                    oper_sparse = op_prod(op_def, spins, include_unit=False, sparse=True)
                    oper_dense = op_prod(op_def, spins, include_unit=False, sparse=False)
                
                    # Create the reference operator manually
                    oper_ref = np.array([[1]])
                    if i != 0:
                        oper_ref = np.kron(oper_ref, op_i)
                    if j != 0:
                        oper_ref = np.kron(oper_ref, op_j)
                    if k != 0:
                        oper_ref = np.kron(oper_ref, op_k)

                    # Compare
                    self.assertTrue(np.allclose(oper_sparse.toarray(), oper_ref))
                    self.assertTrue(np.allclose(oper_dense, oper_ref))

    def test_op_T_coupled(self):
        """
        Test the coupled spherical tensor operators for two spins.
        """

        # Test with different spin quantum numbers
        spins = [1/2, 1, 3/2, 2, 5/2, 3, 7/2, 4, 9/2, 5, 11/2]
        for s1 in spins:
            for s2 in spins:

                # Get the two-spin operators
                SxIx = np.kron(op_Sx(s1, sparse=False), op_Sx(s2, sparse=False))
                SxIy = np.kron(op_Sx(s1, sparse=False), op_Sy(s2, sparse=False))
                SxIz = np.kron(op_Sx(s1, sparse=False), op_Sz(s2, sparse=False))
                SyIx = np.kron(op_Sy(s1, sparse=False), op_Sx(s2, sparse=False))
                SyIy = np.kron(op_Sy(s1, sparse=False), op_Sy(s2, sparse=False))
                SyIz = np.kron(op_Sy(s1, sparse=False), op_Sz(s2, sparse=False))
                SzIx = np.kron(op_Sz(s1, sparse=False), op_Sx(s2, sparse=False))
                SzIy = np.kron(op_Sz(s1, sparse=False), op_Sy(s2, sparse=False))
                SzIz = np.kron(op_Sz(s1, sparse=False), op_Sz(s2, sparse=False))

                # Test relations given in Eq. 254-262, Man: Cartesian and Spherical Tensors in NMR Hamiltonians
                # Using dense arrays
                self.assertTrue(np.allclose(op_T_coupled(0, 0, 1, s1, 1, s2, sparse=False),
                                            -1 / np.sqrt(3) * (SxIx + SyIy + SzIz)))
                self.assertTrue(np.allclose(op_T_coupled(1, 1, 1, s1, 1, s2, sparse=False),
                                            1 / 2 * (SzIx - SxIz + 1j * (SzIy - SyIz))))
                self.assertTrue(np.allclose(op_T_coupled(1, 0, 1, s1, 1, s2, sparse=False),
                                            1j / np.sqrt(2) * (SxIy - SyIx)))
                self.assertTrue(np.allclose(op_T_coupled(1, -1, 1, s1, 1, s2, sparse=False),
                                            1 / 2 * (SzIx - SxIz - 1j * (SzIy - SyIz))))
                self.assertTrue(np.allclose(op_T_coupled(2, 2, 1, s1, 1, s2, sparse=False),
                                            1 / 2 * (SxIx - SyIy + 1j * (SxIy + SyIx))))
                self.assertTrue(np.allclose(op_T_coupled(2, 1, 1, s1, 1, s2, sparse=False),
                                            -1 / 2 * (SxIz + SzIx + 1j * (SyIz + SzIy))))
                self.assertTrue(np.allclose(op_T_coupled(2, 0, 1, s1, 1, s2, sparse=False),
                                            1 / np.sqrt(6) * (-SxIx + 2 * SzIz - SyIy)))
                self.assertTrue(np.allclose(op_T_coupled(2, -1, 1, s1, 1, s2, sparse=False),
                                            1 / 2 * (SxIz + SzIx - 1j * (SyIz + SzIy))))
                self.assertTrue(np.allclose(op_T_coupled(2, -2, 1, s1, 1, s2, sparse=False),
                                            1 / 2 * (SxIx - SyIy - 1j * (SxIy + SyIx))))
                
                # Test the same relations using sparse arrays
                self.assertTrue(np.allclose(op_T_coupled(0, 0, 1, s1, 1, s2, sparse=True).toarray(),
                                            -1 / np.sqrt(3) * (SxIx + SyIy + SzIz)))
                self.assertTrue(np.allclose(op_T_coupled(1, 1, 1, s1, 1, s2, sparse=True).toarray(),
                                            1 / 2 * (SzIx - SxIz + 1j * (SzIy - SyIz))))
                self.assertTrue(np.allclose(op_T_coupled(1, 0, 1, s1, 1, s2, sparse=True).toarray(),
                                            1j / np.sqrt(2) * (SxIy - SyIx)))
                self.assertTrue(np.allclose(op_T_coupled(1, -1, 1, s1, 1, s2, sparse=True).toarray(),
                                            1 / 2 * (SzIx - SxIz - 1j * (SzIy - SyIz))))
                self.assertTrue(np.allclose(op_T_coupled(2, 2, 1, s1, 1, s2, sparse=True).toarray(),
                                            1 / 2 * (SxIx - SyIy + 1j * (SxIy + SyIx))))
                self.assertTrue(np.allclose(op_T_coupled(2, 1, 1, s1, 1, s2, sparse=True).toarray(),
                                            -1 / 2 * (SxIz + SzIx + 1j * (SyIz + SzIy))))
                self.assertTrue(np.allclose(op_T_coupled(2, 0, 1, s1, 1, s2, sparse=True).toarray(),
                                            1 / np.sqrt(6) * (-SxIx + 2 * SzIz - SyIy)))
                self.assertTrue(np.allclose(op_T_coupled(2, -1, 1, s1, 1, s2, sparse=True).toarray(),
                                            1 / 2 * (SxIz + SzIx - 1j * (SyIz + SzIy))))
                self.assertTrue(np.allclose(op_T_coupled(2, -2, 1, s1, 1, s2, sparse=True).toarray(),
                                            1 / 2 * (SxIx - SyIy - 1j * (SxIy + SyIx))))

    def test_operator(self):
        """
        A test for creating the Hilbert-space operators using the operators string.
        """

        # Create a test spin system
        isotopes = np.array(["1H", "14N", "23Na"])
        spin_system = SpinSystem(isotopes)

        # Operators to test
        test_opers = ['E', 'x', 'y', 'z', '+', '-']

        # Get the Zeeman eigenbasis operators
        opers = {}
        for spin in spin_system.spins:
            opers[('E', spin)] = op_E(spin, sparse=False)
            opers[('x', spin)] = op_Sx(spin, sparse=False)
            opers[('y', spin)] = op_Sy(spin, sparse=False)
            opers[('z', spin)] = op_Sz(spin, sparse=False)
            opers[('+', spin)] = op_Sp(spin, sparse=False)
            opers[('-', spin)] = op_Sm(spin, sparse=False)

        # Try all possible product operator combinations
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
                    oper_sparse = operator(spin_system, op_string, sparse=True)
                    oper_dense = operator(spin_system, op_string, sparse=False)

                    # Create the reference operator
                    oper_ref = np.kron(opers[(i, spin_system.spins[0])],
                                       np.kron(opers[(j, spin_system.spins[1])],
                                               opers[(k, spin_system.spins[2])]))

                    # Compare
                    self.assertTrue(np.allclose(oper_sparse.toarray(), oper_ref))
                    self.assertTrue(np.allclose(oper_dense, oper_ref))