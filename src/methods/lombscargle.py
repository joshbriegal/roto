from typing import Optional

import numpy as np
from astropy.timeseries import LombScargle
from matplotlib.axes import Axes

from src.methods.periodfinder import PeriodFinder, Periodogram, PeriodResult


class LombScarglePeriodFinder(PeriodFinder):
    """LombScargle method to find periods.
    Conforms to PeriodFinder interface.
    """

    def __init__(
        self,
        timeseries: np.ndarray,
        flux: np.ndarray,
        flux_errors: Optional[np.ndarray] = None,
        min_ratio_of_maximum_peak_size: float = 0.2,
        samples_per_peak: int = 3,
        units: str = "days",
        fit_mean: Optional[bool] = True,
        center_data: Optional[bool] = True,
        nterms: Optional[bool] = 1,
        normalization: Optional[bool] = "standard",
    ):
        """
        Args:
            timeseries (np.ndarray): array like time series.
            flux (np.ndarray): array like flux values
            flux_errors (Optional[np.ndarray], optional): array like errors on flux values. Defaults to None.
            fit_mean (Optional[bool], optional): [description]. Defaults to None.
            center_data (Optional[bool], optional): [description]. Defaults to None.
            nterms (Optional[bool], optional): [description]. Defaults to None.
            normalization (Optional[bool], optional): [description]. Defaults to None.
        """
        super().__init__(timeseries, flux, flux_errors, min_ratio_of_maximum_peak_size, samples_per_peak, units)

        self._lombscargle = LombScargle(
            self.timeseries,
            self.flux,
            dy=self.flux_errors,
            fit_mean=fit_mean,
            center_data=center_data,
            nterms=nterms,
            normalization=normalization,
        )

    def calculate_periodogram(self, **kwargs) -> Periodogram:
        """Calculate LS Periodogram of data

        Args:
            method (str, optional): [description]. Defaults to "auto".
            method_kwds ([type], optional): [description]. Defaults to None.
            normalization ([type], optional): [description]. Defaults to None.
            samples_per_peak (int, optional): [description]. Defaults to 5.
            nyquist_factor (int, optional): [description]. Defaults to 5.
            minimum_frequency ([type], optional): [description]. Defaults to None.
            maximum_frequency ([type], optional): [description]. Defaults to None.

        Returns:
            Tuple[np.ndarray, np.ndarray]: The frequency and Lomb-Scargle power
        """

        method = kwargs.get("method", "auto")
        method_kwds = kwargs.get("method_kwds", None)
        normalization = kwargs.get("normalization", None)
        samples_per_peak = kwargs.get("samples_per_peak", 5)
        nyquist_factor = kwargs.get("nyquist_factor", 5)
        minimum_frequency = kwargs.get("minimum_frequency", None)
        maximum_frequency = kwargs.get("maximum_frequency", None)

        if maximum_frequency is None:
            # set max frequency to nyquist limit to prevent small spurious periods.
            min_timestamp_diff = np.min(np.diff(self.timeseries))
            maximum_frequency = 1.0 / (nyquist_factor * min_timestamp_diff)

        return Periodogram(
            *self._lombscargle.autopower(
                method=method,
                method_kwds=method_kwds,
                normalization=normalization,
                samples_per_peak=samples_per_peak,
                nyquist_factor=nyquist_factor,
                minimum_frequency=minimum_frequency,
                maximum_frequency=maximum_frequency,
            )
        )

    def plot(self, ax: Axes, period: PeriodResult, colour: Optional[str] = "orange") -> Axes:
        """Given a figure and an axis plot the interesting output of the object.

        Args:
            ax ([type]): Matplotlib axis
            period (PeriodResult): Outputted period to plot around
        """
        ax = self.plot_periodogram(ax, period, colour=colour)
        ax.set_title("Lomb Scargle Periodogram")
        return ax
