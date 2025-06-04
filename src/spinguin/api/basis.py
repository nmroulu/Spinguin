"""
basis.py

This module provides the Basis class.
"""

# Referencing SpinSystem class
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from spinguin.api.spin_system import SpinSystem

# Imports
import numpy as np
import scipy.sparse as sp
import warnings
from spinguin.core.states import state_to_truncated_basis
from spinguin.core.superoperators import sop_to_truncated_basis
from spinguin.core.basis import make_basis, truncate_basis_by_coherence
from spinguin.core.la import isvector

class Basis:
    """
    TODO
    """

    # Basis set properties
    _basis: np.ndarray = None
    _max_spin_order: int = None
    _spin_system: SpinSystem = None

    def __init__(self, spin_system: SpinSystem):
        print("Basis set has been initialized with the following defaults:")
        print(f"max_spin_order: {self.max_spin_order}")

        # Store a reference to the SpinSystem
        self._spin_system = spin_system

    @property
    def dim(self) -> int:
        """Dimension of the basis set."""
        return self.basis.shape[0]

    @property
    def max_spin_order(self) -> int:
        return self._max_spin_order
    
    @property
    def basis(self) -> np.ndarray:
        return self._basis
    
    @max_spin_order.setter
    def max_spin_order(self, max_spin_order):
        """
        Specifies the maximum number of a active spins that are included in the
        product operators that constitute the basis set. Must be at least 1 and
        not larger than the number of spins in the system.
        """
        if max_spin_order < 1:
            raise ValueError("Maximum spin order must be at least 1.")
        if max_spin_order > self._spin_system.nspins:
            raise ValueError("Maximum spin order must not be larger than "
                             "the number of spins in the system.")
        self._max_spin_order = max_spin_order
        print(f"Maximum spin order set to: {self.max_spin_order}\n")

    def build(self):
        """
        Builds the basis set for the spin system. Prior to building the basis,
        the isotopes must be defined.
        """
        # If maximum spin order is not specified, raise a warning and set it
        # equal to the number of spins
        if self.max_spin_order is None:
            warnings.warn("Maximum spin order not specified.\
                          Defaulting to the number of spins.")
            self.max_spin_order = self._spin_system.nspins

        # Build the basis
        self._basis = make_basis(spins = self._spin_system.spins,
                                 max_spin_order = self.max_spin_order)
        
    def truncate_by_coherence(self,
                              coherence_orders: list,
                              *objs: np.ndarray | sp.csc_array
                              ) -> None | tuple[np.ndarray | sp.csc_array]:
        """
        Truncates the basis set by retaining only the product operators that
        correspond to coherence orders specified in the `coherence_orders` list.

        Optionally, superoperators or state vectors can be given as input. These
        will be converted to the truncated basis.

        Parameters
        ----------
        coherence_orders : list
            List of coherence orders to be retained in the basis.

        Returns
        -------
        objs_transformed : tuple
            Superoperators and state vectors transformed into the truncated
            basis.
        """
        # Truncate the basis and obtain the index map
        truncated_basis, index_map = truncate_basis_by_coherence(
            basis = self.basis,
            coherence_orders = coherence_orders
        )

        # Update the basis
        self._basis = truncated_basis

        # Optionally, convert the superoperators and state vectors to the
        # truncated basis
        if objs:
            objs_transformed = []
            for obj in objs:

                # Consider state vectors
                if isvector(obj):
                    objs_transformed.append(state_to_truncated_basis(
                        index_map=index_map,
                        rho=obj))
                    
                # Consider superoperators
                else:
                    objs_transformed.append(sop_to_truncated_basis(
                        index_map=index_map,
                        sop=obj
                    ))

            # Convert to tuple
            objs_transformed = tuple(objs_transformed)

            return objs_transformed
