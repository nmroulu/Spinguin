# Changelog

## 0.0.4
- It is possible to create a subsystem from a SpinSystem object.

## 0.0.3
- Add possibility to enquire the index of a state from the basis set.
- Add functions for converting between operator indices & rank and projection.
- Add function for calculating the coherence order of a state
- Fix bug when creating a state
- Functions obey the dense/sparse rule faithfully irrespective of the input
type.
- Removed sparsity options for Hamiltonian, relaxation superoperator and pulse.
These use the sparsity option of the superoperator.
- Better tests.
- Add cache clear functionality.
- Add possibility to create single-spin and coupled spherical tensor operators.

## 0.0.2
- Re-structure the program internally.
- Improved documentation.
- Pulse sequences moved to their own subpackage.
- Add possibility to reset the parameters to defaults.

## 0.0.1
- Initial release of the Spinguin package.