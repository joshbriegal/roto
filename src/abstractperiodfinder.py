from abc import ABC, abstractmethod

class PeriodFinder(ABC):

    def __init__(self, timeseries, flux, flux_errors):

        self.timeseries = timeseries
        self.flux = flux
        self.flux_errors = flux_errors

    @abstractmethod
    def __call__(self, *args):
        pass
