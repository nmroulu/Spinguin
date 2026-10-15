.. _internals:

Internal classes
================

Spinguin contains a number of internal classes that organise common
functionality and support the overall workflow of the package. These classes
are not intended to be instantiated directly by the user. Instead, they are
part of the internal structure of the program and should be accessed through
the exposed functionality of the public interface.

.. autoclass:: spinguin._core._basis.Basis
   :members:

.. autoclass:: spinguin._core._parameters.Parameters
   :members:

.. autoclass:: spinguin._core._relaxation_properties.RelaxationProperties
   :members: