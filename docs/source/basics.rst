.. _basics:

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

Below, the main functionality of Spinguin is introduced by simulating a simple
NMR spectrum of a pyridine molecule.

Usage
-----
Spinguin is designed to be used directly under the ``spinguin`` namespace. To
start, simply import the package::

    import spinguin as sg

You can then access its functionality through the ``sg`` namespace. For the
available functionality, please refer to :ref:`spinguin`.

Global parameters
-----------------
Spinguin makes use of global parameters that can be set to control the behavior
of the simulations. These parameters can be set by calling
``sg.parameters.PARAMETERNAME`` after importing the package. For example, to set
the magnetic field and temperature, you can use the following commands::

    sg.parameters.magnetic_field = 9.4
    sg.parameters.temperature = 295

Essentially, the ``parameters`` is an instance of ``Parameters`` class, and the
behaviour of the Spinguin package can be managed by changing the attributes of
the object. For the available parameters, please refer to the ``Parameters``
class in :ref:`internals`.

Creating a spin system
----------------------

A ``SpinSystem`` is the core of any spin dynamics simulation in Spinguin. It
represents a collection of spins, to which properties, such as chemical shifts,
*J*-coupling constants, and relaxation rates can be assigned.

You can create a spin system by providing a list of spin types (e.g., '1H',
'13C', etc.) to the ``SpinSystem`` class. The spin types are specified as
strings, and you can include as many spins as needed::

    spin_system = sg.SpinSystem(['1H', '1H', '1H', '1H', '1H', '14N'])

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

The ``basis`` is an instance of ``Basis`` class which is automatically assigned
as an attribute to the ``SpinSystem`` object. Other functionality available in
the ``Basis`` class is documented in :ref:`internals`.

Defining spin system properties
-------------------------------

The properties of the spin system are assigned as attributes to the system.

First, we are going to assign the chemical shifts to the spin system created
above. The chemical shifts are specified as a list of floats in the units of ppm
(parts per million)::

    spin_system.chemical_shifts = [8.56, 8.56, 7.47, 7.47, 7.88, 95.94]

Next, we assign the *J*-coupling constants. These are given as a two-dimensional
list in the units of Hz::

    spin_system.J_couplings = [
        [ 0,     0,      0,      0,      0,      0],
        [-1.04,  0,      0,      0,      0,      0],
        [ 4.85,  1.05,   0,      0,      0,      0],
        [ 1.05,  4.85,   0.71,   0,      0,      0],
        [ 1.24,  1.24,   7.55,   7.55,   0,      0],
        [ 8.16,  8.16,   0.87,   0.87,  -0.19,   0]
    ]

Define the relaxation theory
----------------------------

The goal is to simulate the NMR spectrum of pyridine. Therefore, relaxation must
be defined. We use the `phenomenological` relaxation here, `i.e.`, we give the
relaxation times to the software.

First, the relaxation theory must be defined::
    
    spin_system.relaxation.theory = "phenomenological"

Then, the relaxation times in seconds can be given as a list::

    spin_system.relaxation.T1 = [20.0, 17.5, 15.0, 17.5, 20.0, 0.001]
    spin_system.relaxation.T2 = [5.0, 4.2, 3.1, 4.2, 5.0, 0.0002]

Here, the ``relaxation`` is an instance of ``RelaxationProperties`` class, which
is used to store the relaxation-theory settigns. The other available settings
are documented in the class description in :ref:`internals`.

Performing the pulse-and-acquire experiment
-------------------------------------------

Once the global parameters, spin system, basis set, spin system properties, and 
relaxation theory have been defined, the simulations can be performed. We
simulate the NMR spectrum using the pulse-and-acquire experiment. The available
pulse sequences are documented in :ref:`sequences` and are accessible by
``sg.sequences.SEQUENCENAME``. The pulse-and-acquire experiment requires five
additional parameters to be defined:

* isotope
* center frequency (ppm)
* number of points
* dwell time (s)
* pulse angle

Next, we define these parameters::

    isotope = "1H"
    center_frequency = 8
    npoints = 12500
    dwell_time = 1e-3
    angle = 90

Finally, we perform the experiment, which outputs the free induction decay::

    fid = sg.sequences.pulse_and_acquire(
        spin_system = spin_system,
        isotope = isotope,
        center_frequency = center_frequency,
        npoints = npoints,
        dwell_time = dwell_time,
        angle = angle
    )

Congratulations! You have now sucessfully performed a spin dynamics simulation
using Spinguin.

Visualising the FID
-------------------

We plot the results using ``matplotlib``. First, make sure that ``matplotlib``
is installed and then import the package::

    import matplotlib.pyplot as plt

Next, calculate the time axis using the number of points and dwell time::

    time_axis = sg.time_axis(npoints, dwell_time)

Now we can plot the FID::

    plt.plot(time_axis, fid)
    plt.xlabel("Time (s)")
    plt.ylabel("Intensity (arb. units)")
    plt.show()
    plt.clf()

Obtaining the spectrum
----------------------

The ultimate goal is to plot the NMR spectrum of pyridine, which is obtained by
performing the Fourier transform. The Fourier transform returns the frequencies
and their corresponding intensities. The frequencies are converted to the ppm
scale::

    freqs, spec = sg.spectrum(fid, dwell_time)
    shift_axis = freqs/sg.resonance_frequency(isotope) * 1e6 + center_frequency

Finally, we plot the spectrum of pyridine::

    plt.plot(shift_axis, spec)
    plt.xlabel("Chemical shift (ppm)")
    plt.ylabel("Intensity (arb. units)")
    plt.xlim(8.8, 7.2)
    plt.show()
    plt.clf()

Congratulations! You have now successfully simulated and plotted the NMR
spectrum of pyridine.