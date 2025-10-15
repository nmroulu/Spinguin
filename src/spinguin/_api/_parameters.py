"""
This module provides the Parameters class which contains all the necessary
global parameters required for spin dynamics simulations. It is instantiated
when the Spinguin package is imported and can be accessed by::

    import spinguin as sg
    sg.parameters.PARAMETERNAME = VALUE
"""

class Parameters:
    """
    Parameters class contains all the global parameters for the Spinguin
    package.
    """

    # Experimental conditions
    _magnetic_field: float = None
    _temperature: float = None

    # Rotating frame setting
    _rotating_frame_order: int = 5

    @property
    def magnetic_field(self) -> float:
        """
        External magnetic field (in Tesla).
        """
        return self._magnetic_field

    @magnetic_field.setter
    def magnetic_field(self, magnetic_field: float):
        self._magnetic_field = magnetic_field
        print(f"Magnetic field set to: {self.magnetic_field} T\n")

    @property
    def temperature(self) -> float:
        """
        Temperature of the sample (in Kelvin).
        """
        return self._temperature
    
    @temperature.setter
    def temperature(self, temperature: float):
        self._temperature = temperature
        print(f"Temperature set to: {self.temperature} K\n")

    @property
    def rotating_frame_order(self) -> int:
        """
        Order of the rotating frame approximation. Default is 5.
        """
        return self._rotating_frame_order
    
    @rotating_frame_order.setter
    def rotating_frame_order(self, rotating_frame_order: int):
        self._rotating_frame_order = rotating_frame_order
        print(f"Rotating frame order set to: {self.rotating_frame_order}\n")

# Instantiate the Parameters object
parameters = Parameters()