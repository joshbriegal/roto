import numpy as np

from abc import ABC, abstractmethod
from typing import Optional

class PeriodFinder(ABC):
    """ Abstract PeriodFinder Class Interface """

    def __init__(self, timeseries: np.ndarray, flux: np.ndarray, flux_errors: Optional[np.ndarray] = None):

        self.timeseries = timeseries
        self.flux = flux
        self.flux_errors = flux_errors

    @abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError

