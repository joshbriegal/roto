import inspect
import numpy as np


from typing import Optional

from src.methods.periodfinder import PeriodFinder
from src.methods.lombscargle import LombScarglePeriodFinder

class RoTo:

    METHODS = {
        "lombscargle": LombScarglePeriodFinder
    }

    def __init__(self, timeseries: np.ndarray, flux: np.ndarray, flux_errors: Optional[np.ndarray] = None, methods_parameters: Optional[dict] = None):

        self.timeseries = timeseries
        self.flux = flux
        self.flux_errors = flux_errors

        self.methods = self._parse_constructor_parameters(methods_parameters)


    def _parse_constructor_parameters(self, methods_parameters: Optional[dict]) -> list:
        if methods_parameters is None:
            return [method(self.timeseries, self.flux, self.flux_errors) for method in self.METHODS.values()]
        
        methods = []
        for method, kwargs in methods_parameters.items():
            methods.append(self.METHODS[method](self.timeseries, self.flux, self.flux_errors, **kwargs))

        return methods

    def __call__(self, **kwargs):
        
        periods = [method(**kwargs) for method in self.methods()]


    