Advanced usage
==============

The user-facing funcionality in Spinguin is designed to be used with the
``SpinSystem`` instance, as it simplifies the function calls by encapsulating
most of the information required by the calculations into one object. However,
the core functions in Spinguin have been developed for reusability in mind.
This means that these functions do not depend on the ``SpinSystem`` object at
all. Instead, they take all the necessary information separately as input.

To access the advanced functionality of Spinguin, the user may import these
functions directly from the core of the package. For example, to import the
function responsible for calculating the Redfield relaxation superoperator,
you may call::

    from spinguin.core.relaxation import sop_R_redfield