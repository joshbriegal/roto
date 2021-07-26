import numpy as np

from abc import ABC, abstractmethod
from typing import Optional
from dataclasses import dataclass
from peakutils import indexes


class Periodogram:
    """
    Periodogram dataclass
    """

    def __init__(self, frequency_axis: np.ndarray, power_axis: np.ndarray):
        """[summary]

        Args:
            frequency_axis (np.ndarray): [description]
            power_axis (np.ndarray): [description]
        """
        self.frequency_axis = frequency_axis
        self.power_axis = np.divide(power_axis, power_axis.max())


@dataclass
class PeriodResult:
    period: float
    neg_error: float = 0
    pos_error: float = 0
    method: str = None

    def __repr__(self):
        return f"[{self.period:.4f}+{self.pos_error:.4f}-{self.neg_error:.4f} ({self.method})]"


class PeriodFinder(ABC):
    """ Abstract PeriodFinder Class Interface """

    def __init__(
        self,
        timeseries: np.ndarray,
        flux: np.ndarray,
        flux_errors: Optional[np.ndarray] = None,
        min_ratio_of_maximum_peak_size=0.2,
        samples_per_peak=3,
    ):

        self.timeseries = timeseries
        self.flux = flux
        self.flux_errors = flux_errors

        self.min_ratio_of_maximum_peak_size = min_ratio_of_maximum_peak_size
        self.samples_per_peak = samples_per_peak

    def __call__(self, **kwargs) -> PeriodResult:
        periodogram = self.calculate_periodogram(**kwargs)
        peak_indices = self.calculate_peak_indexes(periodogram.power_axis)
        if peak_indices.size > 0:
            peak_frequencies = periodogram.frequency_axis[peak_indices]
            period = 1.0 / peak_frequencies[0]  # just output largest peak in periodogram for now.
            return PeriodResult(period=period, method=self.__class__.__name__)
        else:
            return None
        

    @abstractmethod
    def calculate_periodogram(self, **kwargs) -> Periodogram:
        raise NotImplementedError

    def calculate_peak_indexes(self, data: np.ndarray) -> np.ndarray:
        """
        :param data: input data
        :return: indexes within x, sorted largest to smallest
        """
        peak_indices = indexes(
            data,
            thres=self.min_ratio_of_maximum_peak_size,
            min_dist=self.samples_per_peak,
        )

        if peak_indices.size > 0:

            sort_indexes = data[peak_indices].argsort()

            return peak_indices[sort_indexes[::-1]]
        else:
            return np.array([])
