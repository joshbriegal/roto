from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np
from peakutils import indexes


@dataclass
class Periodogram:
    """
    Periodogram dataclass.
    Stores frequency axis and normalised power axis.

    Args:
        frequency_axis (np.ndarray): [description]
        power_axis (np.ndarray): [description]
    """

    frequency_axis: np.ndarray
    power_axis: np.ndarray
    normalize: bool = True

    def __post_init__(self):
        """Normalises the power axis of the periodogram"""
        if self.normalize:
            self.power_axis = np.divide(self.power_axis, self.power_axis.max())


@dataclass
class PeriodResult:
    """Period Result Dataclass.

    Args:
        period: period value
        neg_error: error below period value
        pos_error: error above period value
        method: name of method used to calculate period (e.g. LombScarglePeriodFinder)
        value_distribution: = None
        value_probabilities: = None
    """

    period: float
    neg_error: float = 0
    pos_error: float = 0
    method: str = None
    period_distribution: Optional[np.ndarray] = None

    def __repr__(self):
        return f"[{self.period:.4f}+{self.pos_error:.4f}-{self.neg_error:.4f} ({self.method})]"

    def __add__(self, other: "PeriodResult"):
        """Add together two PeriodResult objects.
        Will add upper and lower errors in quadrature.
        """
        return PeriodResult(
            period=(self.period + other.period),
            neg_error=np.sqrt(self.neg_error ** 2 + other.neg_error ** 2),
            pos_error=np.sqrt(self.pos_error ** 2 + other.pos_error ** 2),
            method="CombinedPeriodResult",
        )

    def __sub__(self, other: "PeriodResult"):
        """Subtract two PeriodResult objects and find absolute difference.
        Will add upper and lower errors in quadrature.
        """
        return PeriodResult(
            period=np.abs(self.period - other.period),
            neg_error=np.sqrt(self.neg_error ** 2 + other.neg_error ** 2),
            pos_error=np.sqrt(self.pos_error ** 2 + other.pos_error ** 2),
            method="CombinedPeriodResult",
        )

    def __truediv__(self, other: float):
        relative_neg_error = self.neg_error / self.period
        relative_pos_error = self.pos_error / self.period

        new_period = self.period / other

        return PeriodResult(
            period=new_period,
            neg_error=(relative_neg_error * new_period),
            pos_error=(relative_pos_error * new_period),
            method="CombinedPeriodResult",
        )


class PeriodFinder(ABC):
    """Abstract PeriodFinder Class Interface"""

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

        self.periodogram = None

    def __call__(self, **kwargs) -> PeriodResult:
        """Call the PeriodFinder object to return a PeriodResult object.

        Returns:
            PeriodResult: PeriodResult contains period, error and method information.
        """
        self.periodogram = self.calculate_periodogram(**kwargs)
        peak_indices = self.calculate_peak_indexes(self.periodogram.power_axis)
        if peak_indices.size > 0:
            peak_frequencies = self.periodogram.frequency_axis[peak_indices]
            period = (
                1.0 / peak_frequencies[0]
            )  # just output largest peak in periodogram for now.
            return PeriodResult(period=period, method=self.__class__.__name__)

        return None

    @abstractmethod
    def calculate_periodogram(self, **kwargs) -> Periodogram:
        """Abstract method for calculating periodogram of data.

        Returns:
            Periodogram
        """
        raise NotImplementedError

    def calculate_peak_indexes(self, data: np.ndarray, sort=True) -> np.ndarray:
        """Calculate the indexes of peaks within a data array.

        Args:
            data (np.ndarray): input data values
            sort (bool, default True): if True, return peaks sorted by height

        Returns:
            np.ndarray: indexes within data, sorted largest to smallest
        """
        peak_indices = indexes(
            data,
            thres=self.min_ratio_of_maximum_peak_size,
            min_dist=self.samples_per_peak,
        )

        if peak_indices.size > 0:
            if sort:
                sort_indexes = data[peak_indices].argsort()
                return peak_indices[sort_indexes[::-1]]

            return peak_indices

        return np.array([])

    @abstractmethod
    def plot(self, ax, period: PeriodResult) -> None:
        """Given a figure and an axis plot the interesting output of the object.

        Args:
            ax ([type]): Matplotlib axis
            period (PeriodResult): Outputted period to plot around
        """
        raise NotImplementedError

    def plot_periodogram(self, ax, period: PeriodResult) -> None:
        """Plot a periodgram generated by calling the PeriodFinder.

        Args:
            ax ([type]): Matplotlib axis
            period (PeriodResult): Outputted period to plot around
        """
        if self.periodogram is None:
            self()

        ax.semilogx(
                    np.divide(1.0, self.periodogram.frequency_axis),
                    self.periodogram.power_axis,
                    lw=1,
                )

        ax.axvline(period.period, color="orange", lw=1)
        ax.axvspan(period.period - period.neg_error, period.period + period.pos_error, color='orange', alpha=0.5)

        ax.set_xlabel("Period")
        ax.set_ylabel("Power")

        

        

