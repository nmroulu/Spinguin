import unittest
import numpy as np
import spinguin as sg
from typing import Literal
from spinguin._core._la import cartesian_tensor_to_spherical_tensor
from spinguin._core._superoperators import structure_coefficients

class TestSuperoperators(unittest.TestCase):

    def test_superoperator_1(self):
        """
        Test that the operator sparsity does not influence the output.
        """
        # Reset parameters to defaults
        sg.parameters.default()

        # Example system
        ss = sg.SpinSystem(["1H", "14N", "23Na"])
        
        # Build the basis set
        ss.basis.max_spin_order = 2
        ss.basis.build()

        # Build all superoperators from the basis set using the dense operators
        sg.parameters.sparse_operator = False
        sops_dense = []
        for op_def in ss.basis.basis:
            sops_dense.append(sg.superoperator(ss, op_def))

        # Clear the cache
        sg.clear_cache()

        # Build all superoperators from the basis set using the sparse operators
        sg.parameters.sparse_operator = True
        sops_sparse = []
        for op_def in ss.basis.basis:
            sops_sparse.append(sg.superoperator(ss, op_def))

        # Compare
        for sop_dense, sop_sparse in zip(sops_dense, sops_sparse):
            self.assertTrue(np.allclose(
                sop_dense.toarray(),
                sop_sparse.toarray()
            ))

    def test_superoperator_2(self):
        """
        Test that the superoperator sparsity setting works as intended.
        """
        # Reset parameters to defaults
        sg.parameters.default()

        # Example system
        ss = sg.SpinSystem(["1H", "14N", "23Na"])
        
        # Build the basis set
        ss.basis.max_spin_order = 2
        ss.basis.build()

        # Test building dense superoperators
        sg.parameters.sparse_superoperator = False
        sops_dense = []
        for op_def in ss.basis.basis:
            sops_dense.append(sg.superoperator(ss, op_def))

        # Test building sparse superoperators
        sg.parameters.sparse_superoperator = True
        sops_sparse = []
        for op_def in ss.basis.basis:
            sops_sparse.append(sg.superoperator(ss, op_def))

        # Compare
        for sop_dense, sop_sparse in zip(sops_dense, sops_sparse):
            self.assertTrue(np.allclose(
                sop_dense,
                sop_sparse.toarray()
            ))

    def test_superoperator_3(self):
        """
        Test that the superoperator created using the structure coefficients
        matches with the trivial approach.
        """
        # Reset the parameters to defaults
        sg.parameters.default()

        # Create a test spin system
        ss = sg.SpinSystem(["1H", "14N"])
        
        # Build the basis set
        ss.basis.max_spin_order = 2
        ss.basis.build()

        # Test all product operators from the basis set
        for op_def_i in ss.basis.basis:

            # Initialise left and right superoperators (for manual construction)
            sop_L_ref = np.zeros((ss.basis.dim, ss.basis.dim), dtype=complex)
            sop_R_ref = np.zeros((ss.basis.dim, ss.basis.dim), dtype=complex)

            # Construct the operator
            op_i = sg.operator(ss, op_def_i)

            # Loop over the operator bras
            for j in range(ss.basis.dim):

                # Construct the operator bra
                op_def_j = ss.basis.basis[j]
                op_j = sg.operator(ss, op_def_j)

                # Loop over the kets
                for k in range(ss.basis.dim):

                    # Construct the operator ket
                    op_def_k = ss.basis.basis[k]
                    op_k = sg.operator(ss, op_def_k)

                    # Calculate the matrix elements
                    norm = np.sqrt(
                        (op_j.conj().T @ op_j).trace() * \
                        (op_k.conj().T @ op_k).trace()
                    )
                    sop_L_ref[j, k] = (op_j.conj().T @ op_i @ op_k).trace()
                    sop_L_ref[j, k] = sop_L_ref[j, k] / norm
                    sop_R_ref[j, k] = (op_j.conj().T @ op_k @ op_i).trace()
                    sop_R_ref[j, k] = sop_R_ref[j, k] / norm

            # Build left and right superoperators using inbuilt function
            # that uses the structure coefficients
            sop_L = sg.superoperator(ss, op_def_i, "left")
            sop_R = sg.superoperator(ss, op_def_i, "right")

            # Compare
            self.assertTrue(np.allclose(sop_L.toarray(), sop_L_ref))
            self.assertTrue(np.allclose(sop_R.toarray(), sop_R_ref))

    def test_superoperator_4(self):
        """
        Test the construction of superoperators at varying truncated basis sets
        against the reference method.
        """
        # Reset parameters to defaults
        sg.parameters.default()

        # Define test spin systems
        ss1 = sg.SpinSystem(["1H"])
        ss2 = sg.SpinSystem(["1H", "14N"])
        test_systems = [ss1, ss2]

        # Test with both systems
        for ss in test_systems:

            # Test all possible spin orders
            for max_so in range(1, ss.nspins + 1):

                # Create a basis set
                ss.basis.max_spin_order = max_so
                ss.basis.build()

                # Test all possible operators
                for op_def in ss.basis.basis:

                    # Create reference superoperators using an "idiot-proof"
                    # function
                    sop_L_ref = sop_prod_ref(
                        op_def,
                        ss.basis.basis,
                        ss.spins,
                        'left'
                    )
                    sop_R_ref = sop_prod_ref(
                        op_def,
                        ss.basis.basis,
                        ss.spins,
                        'right'
                    )
                    sop_comm_ref = sop_prod_ref(
                        op_def,
                        ss.basis.basis,
                        ss.spins,
                        'comm'
                    )

                    # Create superoperators using the inbuilt function
                    sop_L = sg.superoperator(ss, op_def, "left")
                    sop_R = sg.superoperator(ss, op_def, "right")
                    sop_comm = sg.superoperator(ss, op_def, "comm")

                    # Compare
                    self.assertTrue(np.allclose(
                        sop_L.toarray(),
                        sop_L_ref
                    ))
                    self.assertTrue(np.allclose(
                        sop_R.toarray(),
                        sop_R_ref
                    ))
                    self.assertTrue(np.allclose(
                        sop_comm.toarray(),
                        sop_comm_ref
                    ))

    def test_superoperator_5(self):
        """
        Test caching behavior of the superoperator function when the basis
        changes.
        """
        # Reset parameters to defaults
        sg.parameters.default()

        # Example system
        ss = sg.SpinSystem(["1H", "1H", "1H"])

        # Define an operator to be created
        op_def = np.array([2, 0, 0])

        # Construct the full basis
        ss.basis.max_spin_order = 3
        ss.basis.build()
        
        # Create the superoperator in the full basis
        Iz = sg.superoperator(ss, op_def, "comm")

        # Construct a truncated basis
        ss.basis.max_spin_order = 2
        ss.basis.build()

        # Create the superoperator in the truncated basis
        Iz_tr = sg.superoperator(ss, op_def, "comm")

        # Resulting shapes should be different
        self.assertNotEqual(Iz.shape, Iz_tr.shape)

    def test_superoperator_6(self):
        """
        Test creating the superoperator from a string against creating the same
        superoperator from an array.
        """
        # Reset to default parameters
        sg.parameters.default()

        # Example system
        ss = sg.SpinSystem(["1H", "1H"])

        # Build a basis set
        ss.basis.max_spin_order = 2
        ss.basis.build()

        # Create the same superoperator using the string and array inputs
        self.assertTrue(np.allclose(
            sg.superoperator(ss, "I(z,0)", "left").toarray(),
            sg.superoperator(ss, [2, 0], "left").toarray()
        ))
        self.assertTrue(np.allclose(
            sg.superoperator(ss, "I(z,0)", "right").toarray(),
            sg.superoperator(ss, [2, 0], "right").toarray()
        ))
        self.assertTrue(np.allclose(
            sg.superoperator(ss, "I(z,0)", "comm").toarray(),
            sg.superoperator(ss, [2, 0], "comm").toarray()
        ))
        self.assertTrue(np.allclose(
            sg.superoperator(ss, "I(z,0) + I(z,1)", "comm").toarray(),
            sg.superoperator(ss, [2, 0], "comm").toarray() + \
            sg.superoperator(ss, [0, 2], "comm").toarray()
        ))
        self.assertTrue(np.allclose(
            sg.superoperator(ss, "I(+,0) * I(-,1)", "comm").toarray(),
            -2 * sg.superoperator(ss, [1, 3], "comm").toarray()
        ))
        self.assertTrue(np.allclose(
            sg.superoperator(ss, "I(x,0) + I(x,1)", "comm").toarray(),
            (
                - 1 / np.sqrt(2) * sg.superoperator(ss, [1, 0], "comm") \
                + 1 / np.sqrt(2) * sg.superoperator(ss, [3, 0], "comm") \
                - 1 / np.sqrt(2) * sg.superoperator(ss, [0, 1], "comm") \
                + 1 / np.sqrt(2) * sg.superoperator(ss, [0, 3], "comm")
            ).toarray()
        ))
        
    def test_sop_T_coupled(self):
        """
        Test creating the Hamiltonian term using "Cartesian" superoperators and
        the coupled spherical tensor superoperator.
        """
        # Set parameters
        sg.parameters.default()
        sg.parameters.sparse_superoperator = False

        # Example system
        ss = sg.SpinSystem(["1H", "1H"])
        
        # Build the basis set
        ss.basis.max_spin_order = 2
        ss.basis.build()

        # Make a random Cartesian interaction tensor
        A = np.random.rand(3, 3)

        # Cartesian spin operators
        I = np.array(['x', 'y', 'z'])
        
        # Perform the dot product manually
        left = np.zeros((ss.basis.dim, ss.basis.dim), dtype=complex)
        for i in range(A.shape[0]):
            for s in range(A.shape[1]):
                left += A[i, s]*sg.superoperator(ss, f"I({I[i]},0)*I({I[s]},1)")

        # Convert A to spherical tensors
        A = cartesian_tensor_to_spherical_tensor(A)

        # Use spherical tensors
        right = np.zeros((ss.basis.dim, ss.basis.dim), dtype=complex)
        for l in range(0, 3):
            for q in range(-l, l + 1):
                right += (-1)**(q)*A[(l, q)] * sg.sop_T_coupled(ss, l, -q, 0, 1)

        # Both conventions should give the same result
        self.assertTrue(np.allclose(left, right))

    def test_sop_to_truncated_basis(self):
        """
        Test the transformation of superoperators to truncated basis.
        """
        # Reset to defaults
        sg.parameters.default()

        # Example system
        ss = sg.SpinSystem(["1H", "1H", "1H", "1H", "14N"])

        # Create the basis set
        ss.basis.max_spin_order = 3
        ss.basis.build()

        # Operators to test
        operators = ['E', 'x', 'y', 'z', '+', '-']

        # Create the superoperators in the original basis set
        sops_original = []
        for i in operators:
            if i == "E":
                op_i = "E"
            else:
                op_i = f"I({i}, 0)"

            for j in operators:
                if j == "E":
                    op_j = "E"
                else:
                    op_j = f"I({j}, 1)"

                for k in operators:
                    if k == "E":
                        op_k = "E"
                    else:
                        op_k = f"I({k}, 2)"

                    # Create the operator string
                    op_string = f"{op_i} * {op_j} * {op_k}"

                    # Create the superoperator
                    sop = sg.superoperator(ss, op_string)
                    sops_original.append(sop)

        # Truncate the basis set and superoperators
        sops_trunc = ss.basis.truncate_by_coherence([-2, 0, 2], *sops_original)

        # Create the superoperators in the truncated basis directly
        sops_trunc_ref = []
        for i in operators:
            if i == "E":
                op_i = "E"
            else:
                op_i = f"I({i}, 0)"

            for j in operators:
                if j == "E":
                    op_j = "E"
                else:
                    op_j = f"I({j}, 1)"

                for k in operators:
                    if k == "E":
                        op_k = "E"
                    else:
                        op_k = f"I({k}, 2)"

                    # Create the operator string
                    op_string = f"{op_i} * {op_j} * {op_k}"

                    # Create the superoperator
                    sop = sg.superoperator(ss, op_string)
                    sops_trunc_ref.append(sop)

        # Compare
        for sop_trunc, sop_trunc_ref in zip(sops_trunc, sops_trunc_ref):
            self.assertTrue(np.allclose(
                sop_trunc.toarray(),
                sop_trunc_ref.toarray()
            ))

def sop_prod_ref(
    op_def: np.ndarray,
    basis: np.ndarray,
    spins: np.ndarray,
    side: Literal["comm", "left", "right"]
) -> np.ndarray:
    """
    A reference method for calculating the superoperator.
    
    NOTE:
    This implementation is very slow and should be used for testing purposes
    only.

    Parameters
    ----------
    op_def : ndarray
        Specifies the product operator to be generated. For example,
        input `(0, 2, 0, 1)` will generate `E*T_10*E*T_11`. The indices are
        given by `N = l^2 + l - q`, where `l` is the rank and `q` is the
        projection.
    basis : ndarray
        A two-dimensional array where each row contains integers that represent
        a Kronecker product of single-spin irreducible spherical tensors.
    spins : ndarray
        A sequence of floats describing the spin quantum numbers of the spin
        system.
    side : {'comm', 'left', 'right'}
        Specifies the type of superoperator:
        - 'comm' -- commutation superoperator
        - 'left' -- left superoperator
        - 'right' -- right superoperator

    Returns
    -------
    sop : ndarray
        Superoperator defined by `op_def`.
    """

    # If commutation superoperator, calculate left and right superoperators and
    # return their difference
    if side == 'comm':
        sop = sop_prod_ref(op_def, basis, spins, 'left') \
            - sop_prod_ref(op_def, basis, spins, 'right')
        return sop
    
    # Obtain the basis dimension and number of spins
    dim = basis.shape[0]
    nspins = spins.shape[0]
    
    # Initialize the superoperator
    sop = np.zeros((dim, dim), dtype=complex)

    # Loop over each matrix row j
    for j in range(dim):

        # Loop over each matrix column k
        for k in range(dim):

            # Initialize the matrix element
            sop_jk = 1

            # Loop over the spins
            for n in range(nspins):

                # Get the single-spin operator indices
                i_ind = op_def[n]
                j_ind = basis[j, n]
                k_ind = basis[k, n]

                # Get the structure coefficients for the current spin
                c = structure_coefficients(spins[n], side)

                # Add to the product
                sop_jk = sop_jk * c[i_ind, j_ind, k_ind]

            # Add to the superoperator
            sop[j, k] = sop_jk

    return sop