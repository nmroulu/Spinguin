"""
parameters.py

This module provides the Parameters class which contains all the necessary
global parameters required for spin dynamics simulations.
"""

# Imports
import numpy as np
from spinguin.core.la import arraylike_to_array

class Parameters:
    """
    Parameters class contains all the necessary global parameters required for
    spin dynamics simulations.

    Attributes
    ----------
    magnetic_field : float, default=None
        External magnetic field in the units of T.
    temperature : float, default=None
        Temperature in the units of K.
    center_frequency : dict, default=None
        Defines the central point of the frequency window for data acquisition.
        Center frequency, defined in ppm, can be set for each isotope
        inividually.
    dwell_time : ndarray, default=None
        Defines the sampling interval in seconds. If there are multiple
        dimensions, the direct dimension is always specified last. Dwell time
        can also be set by changing the spectral width.
    spectral_width : ndarray, default=None
        Defines the spectral width in Hz. If there are multiple dimensions, the
        direct dimension is always specified last. Spectral width can also be
        set by changing the dwell time.
    npoints : int, default=None
        Defines the number of points to acquire in the FID.
    """

    # Experimental conditions
    _magnetic_field: float = None
    _temperature: float = None

    # Spectrometer settings
    _center_frequency: dict = None
    _dwell_time: np.ndarray = None
    _spectral_width: np.ndarray = None
    _npoints: int = None
    
    def __init__(self):
        print("Global simulation parameters have been initialized to the "
              "following defaults:")
        print(f"magnetic_field: {self.magnetic_field}")
        print(f"temperature: {self.temperature}")
        print(f"dwell_time: {self.dwell_time}")
        print(f"npoints: {self.npoints}")
        print(f"center_frequency: {self.center_frequency}")
        print()

    @property
    def magnetic_field(self) -> float:
        return self._magnetic_field
    
    @magnetic_field.setter
    def magnetic_field(self, magnetic_field: float):
        """
        External magnetic field in the units of T.
        """
        self._magnetic_field = magnetic_field
        print(f"Magnetic field set to: {self.magnetic_field} T\n")

    @property
    def temperature(self) -> float:
        return self._temperature
    
    @temperature.setter
    def temperature(self, temperature: float):
        """
        Temperature in the units of K.
        """
        self._temperature = temperature
        print(f"Temperature set to: {self.temperature} K\n")

    @property
    def center_frequency(self) -> dict:
        return self._center_frequency
    
    @center_frequency.setter
    def center_frequency(self, center_frequency: dict):
        """
        Defines the central point of the frequency window for data acquisition.
        Center frequency, defined in ppm, can be set for each isotope
        inividually. The input should be a dictionary.
        """
        if not isinstance(center_frequency, dict):
            raise ValueError("Center frequencies should be given as "
                             "dictionary.")
        if self._center_frequency is None:
            self._center_frequency = center_frequency
        else:
            self._center_frequency.update(center_frequency)
        print(f"Center frequencies set to: {self.center_frequency} (ppm)\n")

    @property
    def dwell_time(self) -> np.ndarray:
        return self._dwell_time
    
    @dwell_time.setter
    def dwell_time(self, dwell_time: float | list | tuple | np.ndarray):
        """
        Defines the sampling interval in seconds. If multiple dimensions are
        used, an array must be specified where the direct dimension corresponds
        to the latest value.
        """
        # Set the dwell time
        self._dwell_time = arraylike_to_array(dwell_time)

        # Set the spectral width accordingly
        self._spectral_width = 1/self.dwell_time

        print("Dwell time has been set to:")
        for dim, dt in enumerate(self.dwell_time):
            print(f"Dimension {dim}: {dt} s")

        print("Corresponding spectral widths are:")
        for dim, sw in enumerate(self.spectral_width):
            print(f"Dimension {dim}: {sw} Hz")

    @property
    def spectral_width(self) -> np.ndarray:
        return self._spectral_width
    
    @spectral_width.setter
    def spectral_width(self, spectral_width: float | list | tuple | np.ndarray):
        """
        Defines the spectral width in Hz. If multiple dimensions are used, an
        array must be specified where the latest element corresponds to the
        direct dimension.
        """
        # Set the spectral width
        self._spectral_width = arraylike_to_array(spectral_width)

        # Set the dwell time accordingly
        self._dwell_time = 1 / self.spectral_width

        print("Spectral width has been set to:")
        for dim, sw in enumerate(self.spectral_width):
            print(f"Dimension {dim}: {sw} Hz")

        print("Corresponding dwell times are:")
        for dim, dt in enumerate(self.dwell_time):
            print(f"Dimension {dim}: {dt} s")

    @property
    def npoints(self) -> int:
        return self._npoints
    
    @npoints.setter
    def npoints(self, npoints: int):
        """
        Defines the number of points to acquire in the FID.
        """
        self._npoints = npoints
        print(f"Number of points to acquire set to: {self.npoints}\n")

# Instantiate the Parameters object
parameters = Parameters()