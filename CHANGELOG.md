# Changelog

## 0.1.1
- Added quadrupolar moments to several isotopes.

## 0.1.0
- Improved basis set handling.
- Refined the implementation of zero-track elimination.
- Added anisotropic rotational diffusion for Redfield relaxation theory.
- Added a verbosity option to parameters (to disable printing).
- General documentation and comment updates.
- Added more general rotating frame implementation.
- Improved basis set truncation options.

## 0.0.4
- Added support for creating a subsystem from a `SpinSystem` object.

## 0.0.3
- Added the possibility to enquire the index of a state from the basis
	set.
- Added functions for converting between operator indices, rank, and
	projection.
- Added a function for calculating the coherence order of a state.
- Fixed a bug in state creation.
- Ensured that functions obey the dense/sparse rule irrespective of the
	input type.
- Removed separate sparsity options for the Hamiltonian, relaxation
	superoperator, and pulse operators. These now follow the sparsity option
	of the superoperator.
- Improved the test suite.
- Added cache-clearing functionality.
- Added the possibility to create single-spin and coupled spherical tensor
	operators.

## 0.0.2
- Restructured the program internally.
- Improved documentation.
- Moved pulse sequences to their own subpackage.
- Added the possibility to reset the parameters to their default values.

## 0.0.1
- Initial release of the Spinguin package.