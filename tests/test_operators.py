import unittest
import numpy as np
import math
from scipy.sparse import lil_array
from hyppy.spin_system import SpinSystem
from hyppy.la import comm, cartesian_tensor_to_spherical_tensor
from hyppy.basis import idx_to_lq, str_to_op_def, ZQ_basis
from hyppy import operators

class TestOperators(unittest.TestCase):

    def test_op_S(self):

        # Test against hard-coded operators
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
            self.assertTrue(np.allclose(operators.op_E(spin), op))
        for spin, op in Sx.items():
            self.assertTrue(np.allclose(operators.op_Sx(spin), op))
        for spin, op in Sy.items():
            self.assertTrue(np.allclose(operators.op_Sy(spin), op))
        for spin, op in Sz.items():
            self.assertTrue(np.allclose(operators.op_Sz(spin), op))
        for spin, op in Sp.items():
            self.assertTrue(np.allclose(operators.op_Sp(spin), op))
        for spin, op in Sm.items():
            self.assertTrue(np.allclose(operators.op_Sm(spin), op))

        # Test the commutation relations
        spins = [1/2, 1, 3/2, 2, 5/2, 3, 7/2, 4, 9/2, 5, 11/2]
        for spin in spins:
            self.assertTrue(np.allclose(comm(operators.op_E(spin), operators.op_E(spin)), 0))
            self.assertTrue(np.allclose(comm(operators.op_Sx(spin), operators.op_Sy(spin)), 1j*operators.op_Sz(spin)))
            self.assertTrue(np.allclose(comm(operators.op_Sy(spin), operators.op_Sz(spin)), 1j*operators.op_Sx(spin)))
            self.assertTrue(np.allclose(comm(operators.op_Sz(spin), operators.op_Sx(spin)), 1j*operators.op_Sy(spin)))
            self.assertTrue(np.allclose(comm(operators.op_Sp(spin), operators.op_Sz(spin)), -operators.op_Sp(spin)))
            self.assertTrue(np.allclose(comm(operators.op_Sm(spin), operators.op_Sz(spin)), operators.op_Sm(spin)))

    def test_op_T(self):

        # Test commutation relations for spherical tensors
        spins = [1/2, 1, 3/2, 2, 5/2, 3, 7/2, 4, 9/2, 5, 11/2]
        for spin in spins:
            for l in range(0, int(2*spin+1)):
                for q in range(-l, l+1):
                    self.assertTrue(np.allclose(comm(operators.op_Sz(spin), operators.op_T(spin, l, q)), q*operators.op_T(spin, l, q)))
                    self.assertTrue(np.allclose(comm(operators.op_Sx(spin)@operators.op_Sx(spin) + \
                                                     operators.op_Sy(spin)@operators.op_Sy(spin) + \
                                                     operators.op_Sz(spin)@operators.op_Sz(spin), operators.op_T(spin, l, q)), 0))
                    if not q == -l:
                        self.assertTrue(np.allclose(comm(operators.op_Sm(spin), operators.op_T(spin, l, q)), math.sqrt(l*(l+1) - q*(q-1)) * operators.op_T(spin, l, q-1)))
                    if not q == l:
                        self.assertTrue(np.allclose(comm(operators.op_Sp(spin), operators.op_T(spin, l, q)), math.sqrt(l*(l+1) - q*(q+1)) * operators.op_T(spin, l, q+1)))

    def test_op_P(self):

        # Make a test system
        isotopes = np.array(['1H', '14N', '23Na'])
        spin_system = SpinSystem(isotopes)
        spins = spin_system.spins

        # States to test
        states = ['E', 'I_x', 'I_y', 'I_z', 'I_+', 'I_-']

        # Get the Zeeman eigenbasis operators
        opers = {}
        for spin in spins:
            opers[('E', spin)] = operators.op_E(spin)
            opers[('I_x', spin)] = operators.op_Sx(spin)
            opers[('I_y', spin)] = operators.op_Sy(spin)
            opers[('I_z', spin)] = operators.op_Sz(spin)
            opers[('I_+', spin)] = operators.op_Sp(spin)
            opers[('I_-', spin)] = operators.op_Sm(spin)

        # Try all possible state combinations
        for i in states:
            for j in states:
                for k in states:

                    # Create the operators with unit included and compare
                    op_defs, coeffs = str_to_op_def(spin_system, [i, j, k], [0, 1, 2])
                    oper1 = 0
                    for op_def, coeff in zip(op_defs, coeffs):
                        oper1 += coeff * operators.op_P(op_def, spins, include_unit=True)
                    oper2 = np.kron(opers[(i, spins[0])], np.kron(opers[(j, spins[1])], opers[(k, spins[2])]))
                    self.assertTrue(np.allclose(oper1, oper2))

                    # Create the operators without unit included and compare
                    op_defs, coeffs = str_to_op_def(spin_system, [i, j, k], [0, 1, 2])
                    oper1 = 0
                    for op_def, coeff in zip(op_defs, coeffs):
                        oper1 += coeff * operators.op_P(op_def, spins, include_unit=False)
                    oper2 = 1
                    if i != 'E':
                        oper2 = np.kron(oper2, opers[(i, spins[0])])
                    if j!= 'E':
                        oper2 = np.kron(oper2, opers[(j, spins[1])])
                    if k!= 'E':
                        oper2 = np.kron(oper2, opers[(k, spins[2])])
                    self.assertTrue(np.allclose(oper1, oper2))

    def test_structure_coefficients(self):

        # Make a test system
        isotopes = np.array(['1H', '1H'])
        spin_system = SpinSystem(isotopes)
        basis = spin_system.basis.arr
        dim = spin_system.basis.dim
        spins = spin_system.spins

        # Test operator
        op_def_i = (1, 3)

        # Assign empty array for the left and right superoperators
        sop_L = np.zeros((dim, dim), dtype=complex)
        sop_R = np.zeros((dim, dim), dtype=complex)

        # Get the operator to be constructed
        op_i = 1
        for n in range(len(op_def_i)):
            l_i, q_i = idx_to_lq(op_def_i[n])
            op_i = np.kron(op_i, operators.op_T(spins[n], l_i, q_i))

        # Loop over the bras
        for j in range(dim):

            # Get the operator bra
            op_def_j = basis[j]
            op_j = 1
            for n in range(len(op_def_j)):
                l_j, q_j = idx_to_lq(op_def_j[n])
                op_j = np.kron(op_j, operators.op_T(spins[n], l_j, q_j))

            # Loop over the kets
            for k in range(dim):

                # Get the operator ket
                op_def_k = basis[k]
                op_k = 1
                for n in range(len(op_def_k)):
                    l_k, q_k = idx_to_lq(op_def_k[n])
                    op_k = np.kron(op_k, operators.op_T(spins[n], l_k, q_k))

                # Calculate the elements
                norm = np.sqrt((op_j.conj().T @ op_j).trace() * (op_k.conj().T @ op_k).trace())
                sop_L[j, k] = (op_j.conj().T @ op_i @ op_k).trace() / norm
                sop_R[j, k] = (op_j.conj().T @ op_k @ op_i).trace() / norm

        self.assertTrue(np.allclose(sop_L, operators.sop_P(spin_system, op_def_i, "left").toarray()))
        self.assertTrue(np.allclose(sop_R, operators.sop_P(spin_system, op_def_i, "right").toarray()))
    
    def test_sop_P(self):

        # Make the test spin systems
        systems = []

        # Assign isotopes
        systems.append(np.array(['1H']))
        systems.append(np.array(['1H', '14N']))

        # Test all systems
        for isotopes in systems:

            # Test all possible spin orders
            for max_so in range(1, isotopes.size+1):

                # Initialize the spin system
                spin_system = SpinSystem(isotopes, max_spin_order=max_so)

                # Test all possible operators
                for op in spin_system.basis.arr:

                    # Convert to tuple
                    op = tuple(op)

                    # Test left, right and commutation superoperators against the "idiot-proof" function
                    self.assertTrue(np.allclose(operators.sop_P(spin_system, op, "left").toarray(), 
                                                sop_P(spin_system, op, "left").toarray()))
                    self.assertTrue(np.allclose(operators.sop_P(spin_system, op, "right").toarray(), 
                                                sop_P(spin_system, op, "right").toarray()))
                    self.assertTrue(np.allclose(operators.sop_P(spin_system, op, "comm").toarray(), 
                                                sop_P(spin_system, op, "comm").toarray()))

    def test_sop_P_cache(self):

        # Example system
        isotopes = np.array(['1H', '1H', '1H'])
        spin_system = SpinSystem(isotopes)

        # Create an operator, change the basis, and create an operator again
        op_def = (2, 0, 0)
        Iz = operators.sop_P(spin_system, op_def, side='comm')
        ZQ_basis(spin_system)
        Iz_ZQ = operators.sop_P(spin_system, op_def, side='comm')

        # Resulting shapes should be different
        self.assertNotEqual(Iz.shape, Iz_ZQ.shape)

    def test_superoperator(self):

        # Example system
        isotopes = np.array(['1H', '1H'])
        spin_system = SpinSystem(isotopes)

        # Test the superoperator function against sop_P
        self.assertTrue(np.allclose(operators.superoperator(spin_system, 'I_z', 0, "left").toarray(), operators.sop_P(spin_system, (2, 0), "left").toarray()))
        self.assertTrue(np.allclose(operators.superoperator(spin_system, 'I_z', 0, "right").toarray(), operators.sop_P(spin_system, (2, 0), "right").toarray()))
        self.assertTrue(np.allclose(operators.superoperator(spin_system, 'I_z', 0, "comm").toarray(), operators.sop_P(spin_system, (2, 0), "comm").toarray()))
        self.assertTrue(np.allclose(operators.superoperator(spin_system, 'I_z', [0,1], "comm").toarray(),
                                    (operators.sop_P(spin_system, (2, 0), "comm")+operators.sop_P(spin_system, (0, 2), "comm")).toarray()))
        self.assertTrue(np.allclose(operators.superoperator(spin_system, ['I_+', 'I_-'], [0,1], "comm").toarray(),
                                    -2*operators.sop_P(spin_system, (1, 3), "comm").toarray()))
        self.assertTrue(np.allclose(operators.superoperator(spin_system, 'I_x', [0,1], "comm").toarray(),
                                    + (-1/np.sqrt(2)*operators.sop_P(spin_system, (1, 0), "comm") + 1/np.sqrt(2)*operators.sop_P(spin_system, (3, 0), "comm")).toarray()
                                    + (-1/np.sqrt(2)*operators.sop_P(spin_system, (0, 1), "comm") + 1/np.sqrt(2)*operators.sop_P(spin_system, (0, 3), "comm")).toarray()))
        
    def test_op_T_coupled(self):

        spins = [1/2, 1, 3/2, 2, 5/2, 3, 7/2, 4, 9/2, 5, 11/2]

        # Test with all possiblem spin quantum numbers
        for s1 in spins:
            for s2 in spins:

                # Get the two-spin operators
                SxIx = np.kron(operators.op_Sx(s1), operators.op_Sx(s2))
                SxIy = np.kron(operators.op_Sx(s1), operators.op_Sy(s2))
                SxIz = np.kron(operators.op_Sx(s1), operators.op_Sz(s2))
                SyIx = np.kron(operators.op_Sy(s1), operators.op_Sx(s2))
                SyIy = np.kron(operators.op_Sy(s1), operators.op_Sy(s2))
                SyIz = np.kron(operators.op_Sy(s1), operators.op_Sz(s2))
                SzIx = np.kron(operators.op_Sz(s1), operators.op_Sx(s2))
                SzIy = np.kron(operators.op_Sz(s1), operators.op_Sy(s2))
                SzIz = np.kron(operators.op_Sz(s1), operators.op_Sz(s2))

                # Test relations given in Eq. 254-262, Man: Cartesian and Spherical Tensors in NMR Hamiltonians
                self.assertTrue(np.allclose(operators.op_T_coupled(0, 0, 1, s1, 1, s2), -1/np.sqrt(3) * (SxIx + SyIy + SzIz)))
                self.assertTrue(np.allclose(operators.op_T_coupled(1, 1, 1, s1, 1, s2), 1/2 * (SzIx - SxIz + 1j*(SzIy - SyIz))))
                self.assertTrue(np.allclose(operators.op_T_coupled(1, 0, 1, s1, 1, s2), 1j/np.sqrt(2) * (SxIy - SyIx)))
                self.assertTrue(np.allclose(operators.op_T_coupled(1, -1, 1, s1, 1, s2), 1/2 * (SzIx - SxIz - 1j*(SzIy - SyIz))))
                self.assertTrue(np.allclose(operators.op_T_coupled(2, 2, 1, s1, 1, s2), 1/2 * (SxIx - SyIy + 1j*(SxIy + SyIx))))
                self.assertTrue(np.allclose(operators.op_T_coupled(2, 1, 1, s1, 1, s2), -1/2 * (SxIz + SzIx + 1j*(SyIz + SzIy))))
                self.assertTrue(np.allclose(operators.op_T_coupled(2, 0, 1, s1, 1, s2), 1/np.sqrt(6) * (-SxIx + 2*SzIz - SyIy)))
                self.assertTrue(np.allclose(operators.op_T_coupled(2, -1, 1, s1, 1, s2), 1/2 * (SxIz + SzIx - 1j*(SyIz + SzIy))))
                self.assertTrue(np.allclose(operators.op_T_coupled(2, -2, 1, s1, 1, s2), 1/2 * (SxIx - SyIy - 1j*(SxIy + SyIx))))

    def test_sop_T_coupled(self):
        """
        Test creating the Hamiltonian term using "Cartesian" superoperators and the
        coupled spherical tensor superoperator.
        """

        # Example system
        isotopes = np.array(['1H', '1H'])
        spin_system = SpinSystem(isotopes)

        # Get the dimension of the basis
        dim = spin_system.basis.dim

        # Make a random Cartesian interaction tensor
        A = np.random.rand(3,3)

        # Cartesian spin operators
        I = np.array(['I_x', 'I_y', 'I_z'])
        
        # Perform the dot product manually
        left = np.zeros((dim, dim), dtype=complex)
        for i in range(A.shape[0]):
            for s in range(A.shape[1]):
                left += A[i,s] * operators.superoperator(spin_system, [I[i], I[s]], [0, 1], 'comm').toarray()

        # Convert A to spherical tensors
        A = cartesian_tensor_to_spherical_tensor(A)

        # Use spherical tensors
        right = np.zeros((dim, dim), dtype=complex)
        for l in range(0,3):
            for q in range(-l, l+1):
                right += (-1)**(q) * A[(l, q)] * operators.sop_T_coupled(spin_system, l, -q, 0, 1).toarray()

        # Both conventions should give the same result
        self.assertTrue(np.allclose(left, right))
                    
def sop_P(spin_system:SpinSystem, op_def:tuple, side:str):
    """
    An idiot-proof method for calculating the superoperator. Used for testing purposes.
    """

    # If commutation superoperator, calculate left and right superoperators and return their difference
    if side == 'comm':
        sop = sop_P(spin_system, op_def,'left') \
            - sop_P(spin_system, op_def,'right')
        return sop
    
    # Initialize the superoperator
    sop = lil_array((spin_system.basis.dim, spin_system.basis.dim), dtype=complex)

    # Loop over each matrix row j
    for j in range(spin_system.basis.dim):

        # Loop over each matrix column k
        for k in range(spin_system.basis.dim):

            # Initialize the matrix element
            sop_jk = 1

            # Loop over the spin system
            for n in range(spin_system.size):

                # Get the single-spin operator indices
                i_ind = op_def[n]
                j_ind = spin_system.basis.arr[j, n]
                k_ind = spin_system.basis.arr[k, n]

                # Get the structure coefficients for the current spin
                c = operators.structure_coefficients(spin_system.spins[n], side)

                # Add to the product
                sop_jk = sop_jk * c[i_ind, j_ind, k_ind]

            # Add to the superoperator
            sop[j,k] = sop_jk

    return sop
