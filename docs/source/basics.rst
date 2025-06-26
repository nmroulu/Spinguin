Basics
======

Spinguin is meant to be easy to use, even for those who are not very familiar
with Python nor spin dynamics simulations. The package is designed to for
running spin dynamics simulations in liquid state. The main features of
Spinguin are:

* Versatile numerical spin-dynamics simulations in liquid state.
* Support for restricted basis sets, enabling the use of large spin systems
  with more than 10 spins on consumer-level hardware.
* Simulation of coherent dynamics, relaxation, and chemical exchange processes.

Usage
-----
Spinguin is designed to be used directly under the ``spinguin`` namespace. To
start, simply import the package::

    import spinguin as sg

You can then access its functionality through the ``sg`` namespace.

Global parameters
-----------------
Spinguin makes use of global parameters that can be set to control the behavior
of the simulations. These parameters can be set by calling
``sg.parameters.PARAMETERNAME``. For example, to set the magnetic field and
temperature, you can use the following commands::

    sg.parameters.magnetic_field = 9.4
    sg.parameters.temperature = 295

Creating a spin system
----------------------

A ``SpinSystem`` is the core of any spin dynamics simulation in Spinguin. It
represents a collection of spins, to which properties such as chemical shifts,
*J*-coupling constants, and relaxation rates can be assigned.

You can create a spin system by providing a list of spin types (e.g., '1H',
'13C', etc.) to the ``SpinSystem`` class. The spin types are specified as
strings, and you can include as many spins as needed::

    spin_system = sg.SpinSystem(['1H', '1H', '1H', '1H', '1H', '14N'])

As an example, we are going to assign the chemical shifts to the spin system
created above. The chemical shifts are specified as a list of floats in the
units of ppm (parts per million)::

    spin_system.chemical_shifts = [8.56, 8.56, 7.47, 7.47, 7.88, 95.94]

Building a basis set
--------------------

Simulations in Spinguin are performed in Liouville space using the Kronecker
products of irreducible spherical tensors (ISTs) as the basis set. This allows
the use of a restricted basis set, which is essential for simulating large spin
systems efficiently.

To build the basis, you must first define the maximum spin order that is going
to be used in the basis. This is specified as an attribute to the basis of the
``SpinSystem`` instance. Next, you can call the ``build_basis()`` method to
build the basis set::

    spin_system.basis.max_spin_order = 3
    spin_system.basis.build()

Performing the simulations
--------------------------

Once the global parameters, spin system, and basis set have been defined, the
simulations can be performed. The user can choose to create their own simulation
by calculating the required superoperators, or by calculating the Hamiltonian
and relaxation superoperator. Another option is to use the predefined simulation
functions provided in Spinguin, such as the pulse and acquire and inversion
recovery sequences.