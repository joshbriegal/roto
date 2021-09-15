import logging
from typing import Optional

import numpy as np
import progressbar
from astropy.timeseries import LombScargle
from matplotlib.axes import Axes

from roto.methods.periodfinder import PeriodFinder, Periodogram, PeriodResult

logger = logging.getLogger(__name__)


class LombScarglePeriodFinder(PeriodFinder):
    """LombScargle method to find periods.
    Conforms to PeriodFinder interface.
    """

    def __init__(
        self,
        timeseries: np.ndarray,
        flux: np.ndarray,
        flux_errors: Optional[np.ndarray] = None,
        min_ratio_of_maximum_peak_size: float = 0.2, samples_per_peak: int = 3, time_units: str = "days",
        flux_units: str = "relative flux units",
        fit_mean: Optional[bool] = True,
        center_data: Optional[bool] = True,
        nterms: Optional[bool] = 1,
        normalization: Optional[bool] = "standard",
        sliding: Optional[bool] = True,
    ):
        """
        Args:
            timeseries (np.ndarray): array like time series.
            flux (np.ndarray): array like flux values
            flux_errors (Optional[np.ndarray], optional): array like errors on flux values. Defaults to None.
            fit_mean (Optional[bool]): if True, include a constant offset as part of the model at each frequency. This can lead to more accurate results, especially in the case of incomplete phase coverage.
            center_data (Optional[bool]): if True, pre-center the data by subtracting the weighted mean of the input data. This is especially important if fit_mean = False.
            nterms (Optional[bool]): number of terms to use in the Fourier fit. {‘standard’, ‘model’, ‘log’, ‘psd’},
            normalization (Optional[bool]): Normalization to use for the periodogram.
            sliding (Optional[bool]): Use a sliding window to generate an error on the period.
        """
        super().__init__(
            timeseries,
            flux,
            flux_errors,
            min_ratio_of_maximum_peak_size,
            samples_per_peak,
            time_units,
            flux_units,
        )

        self._lombscargle = LombScargle(
            self.timeseries,
            self.flux,
            dy=self.flux_errors,
            fit_mean=fit_mean,
            center_data=center_data,
            nterms=nterms,
            normalization=normalization,
        )

        self.sliding = sliding
        if self.sliding:
            self.ls_kwargs = {
                "fit_mean": fit_mean,
                "center_data": center_data,
                "nterms": nterms,
                "normalization": normalization,
            }

    def __call__(self, **kwargs) -> PeriodResult:
        """Call the PeriodFinder object to return a PeriodResult object.
        If sliding, will run first run the standard period finder to find a period,
         and then generate a set of PeriodResults using a sliding window over periods.

        Returns:
            PeriodResult: PeriodResult contains period, error and method information.
        """
        period_result = super().__call__(**kwargs)

        if not self.sliding:
            return period_result

        return self._sliding_ls_periodogram(period_result, **kwargs)

    def _sliding_ls_periodogram(
        self,
        period_result_estimate: PeriodResult,
        n_periods: int = 5,
        sliding_aggregation: str = "median",
        **autopower_kwargs,
    ):

        methods = ["mean", "median"]
        if sliding_aggregation not in methods:
            raise ValueError(
                f"method must be on of {methods}, not {sliding_aggregation}"
            )

        period_estimate = period_result_estimate.period

        periods = []
        epoch = self.timeseries.min()
        number_of_windows = (
            int(
                (self.timeseries.max() - (period_estimate * n_periods))
                / period_estimate
            )
            + 1
        )

        if number_of_windows < 3:
            logger.warning(
                "Sliding window too large to generate good estimate, returning regular lombscargle"
            )
            return period_result_estimate

        count = 0
        with progressbar.ProgressBar(
            maxval=number_of_windows,
            widgets=[
                "Sliding LombScargle Window: ",
                progressbar.Counter(),
                " windows (",
                progressbar.Timer(),
                ")",
            ],
        ) as bar:
            while epoch <= self.timeseries.max() - (period_estimate * n_periods):
                idxs = np.logical_and(
                    self.timeseries >= epoch,
                    self.timeseries < epoch + (period_estimate * n_periods),
                )

                if len(self.timeseries[idxs]) == 0:
                    continue

                ls_periodfinder = LombScarglePeriodFinder(
                    self.timeseries[idxs],
                    self.flux[idxs],
                    self.flux_errors[idxs] if self.flux_errors is not None else None,
                    **self.ls_kwargs,
                    sliding=False,
                )
                period_result = ls_periodfinder(**autopower_kwargs)

                if period_result is not None:
                    periods.append(period_result.period)

                epoch += period_estimate
                count += 1
                bar.update(count)

        if sliding_aggregation == "median":
            percentiles = np.percentile(periods, [10, 50, 90])
            ave_period = percentiles[1]
            std_period = percentiles[2] - percentiles[0]
        elif sliding_aggregation == "mean":
            ave_period = np.nanmean(periods)
            std_period = np.nanstd(periods)

        return PeriodResult(
            ave_period, std_period, std_period, method=self.__class__.__name__
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

    def plot(
        self, ax: Axes, period: PeriodResult, colour: Optional[str] = "orange"
    ) -> Axes:
        """Given a figure and an axis plot the interesting output of the object.

        Args:
            ax ([type]): Matplotlib axis
            period (PeriodResult): Outputted period to plot around
        """
        ax = self.plot_periodogram(ax, period, colour=colour)
        ax.set_title("Lomb Scargle Periodogram")
        return ax
