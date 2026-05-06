# Changelog

## 0.2.0
- Re-write the documentation using AI.
- Dynamic frequency shifts are included by default.
- Antisymmetric part of CSA is included by default.
- Thermalisation of the relaxation superoperator is applied by default.
- Added a possibility of adding own isotopes.
- Dipole-dipole coupling constants and tensors can now be requested for a spin
system.

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