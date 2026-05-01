"""
Validation utilities for ensuring prerequisites are met before executing
operations.
"""

from typing import Any

_PRE_MESSAGES = {
    "basis.basis": "Basis must be built before",
    "relaxation.theory": "Relaxation theory must be specified before",
    "relaxation.T1": "T1 relaxation times must be set before",
    "relaxation.T2": "T2 relaxation times must be set before",
    "relaxation.tau_c": "The correlation time must be set before",
    "relaxation.thermalization": "Thermalization must be set to True before",
    "magnetic_field": "Magnetic field must be set before",
    "temperature": "Temperature must be set before",
}

def require(obj: Any, properties: str | list[str], operation: str) -> None:
    """
    Check that the requested properties have been set on the given object.

    Parameters
    ----------
    obj : Any
        The object whose properties are to be checked.
    properties : str or list of str
        A dot-separated property path or a list of such paths.
    operation : str
        Description of the operation that requires these properties.

    Raises
    ------
    ValueError
        Raised if any of the requested properties evaluate to None.
    """
    if isinstance(properties, str):
        properties = [properties]

    for prop in properties:
        current = obj
        attr_path = prop.split(".")
        for attr in attr_path:
            current = getattr(current, attr)

        if current is None:
            pre_message = _PRE_MESSAGES.get(
                prop,
                f"'{prop}' must be set before"
            )
            raise ValueError(f"{pre_message} {operation}.")
