.. _internals:

Internal classes
================

Spinguin contains some internal classes, which are used to improve the workflow
by organising common functionality under a specific class. These classes are
not meant to be instantiated by the user; rather they are already a part of the
program and should be accessed through the exposed functionality in the public
interface.

.. autoclass:: spinguin._core._basis.Basis
   :members:

.. autoclass:: spinguin._core._parameters.Parameters
   :members:

.. autoclass:: spinguin._core._relaxation_properties.RelaxationProperties
   :members: