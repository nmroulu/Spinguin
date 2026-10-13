# Changelog

## Unreleased
- The automatic rotational correlation times use the van der Waals solid of
the molecule with Bondi radii. The isotropic model uses the sphere of equal
volume and the anisotropic model uses an ellipsoid with the inertia axis ratios
of the solid, scaled to the same volume. This replaces the uniform 1.2 Å atomic
radius and the 1 Å correction of the semi-axes.
- The principal axes of anisotropic rotational diffusion are the principal axes
of inertia of the van der Waals solid instead of those of the nuclear masses.
- Added `Molecule.elements`, `Molecule.vdw_radii` (custom radii can be
assigned), `Molecule.principal_axes` and `Molecule.semi_axes`.
- The Perrin integrals are evaluated exactly with Carlson elliptic integrals.
- Fixed the frame transformation of anisotropic rotational diffusion: the
interaction tensors were rotated with the inverse rotation, and a left-handed
set of principal axes could not be reached by a proper rotation. Anisotropic
results were correct only for molecules given in their principal-axis frame.
Added a test of the invariance of the relaxation rates to the molecular
orientation.
- Documented `scaling_factor` of `auto_tau_c` as an empirical, solvent-dependent
parameter.

## 0.2.1
- Accept input of the form "I(component, isotope)" for constructing operators,
superoperators and states.
- Added a function for performing elementary chemical reactions.
- CSA and EFG tensors are assigned to zero arrays by default.
- Added a function for returning the spin order of a product operator.
- Hydrodynamic radius can now be estimated from the molecule geometry when
calculating the isotropic rotational correlation time.

## 0.2.0
- Re-write the documentation using AI.
- Dynamic frequency shifts are included by default.
- Antisymmetric part of CSA is included by default.
- Thermalisation of the relaxation superoperator is applied by default.
- Added a possibility of adding own isotopes.
- Dipole-dipole coupling constants and tensors can now be requested for a spin
system.
- Quadrupole coupling tensors can now be requested for a spin system.
- Shielding interaction tensors (in rad/s) can now be requested for a spin
system.
- Added a function for returning resonance frequencies of a spin system.

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