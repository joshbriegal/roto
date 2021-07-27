from gacf import GACF
from typing import Optional


import numpy as np


from src.methods.periodfinder import PeriodFinder, PeriodResult
from src.methods.periodfinder import Periodogram
from src.methods.fft import FFTPeriodFinder


class GACFPeriodFinder(PeriodFinder):
    """Generalised Autocorrelation Function (G-ACF) method to find periods.
    Conforms to PeriodFinder interface.
    """

    def __init__(
        self,
        timeseries: np.ndarray,
        flux: np.ndarray,
        flux_errors: Optional[np.ndarray] = None,
    ):
        """
        Args:
            timeseries (np.ndarray): array like time series.
            flux (np.ndarray): array like flux values
            flux_errors (Optional[np.ndarray], optional): array like errors on flux values. Defaults to None.
        """
        super().__init__(timeseries, flux, flux_errors)
        self._gacf = GACF(self.timeseries, self.flux, self.flux_errors)

    def calculate_periodogram(self) -> None:
        """A "periodogram" does not exist for an ACF

        Returns:
            None
        """
        return None

    def calculate_autocorrelation(self, **kwargs) -> Periodogram:
        """Calculate G-ACF of data.
        It is recommended to leave selection_function and weight_function as default for speed.

        Args:
            min_lag (float, optional): min lag in units of time. Defaults to None.
            max_lag (float, optional): max lag in units of time. Defaults to None.
            lag_resolution (float, optional): lag resolution in units of time. Defaults to None.
            alpha (float, optional): weight function characteristic length scale, default is t.median_time. Defaults to None.
            selection_function (str, optional): 'fast' or 'natural' - see paper for more details. Defaults to "natural".
            weight_function: (str, optional) 'gaussian' or 'fractional' see paper for more details. Defaults to "fractional".

        Returns:
            Tuple[np.ndarray, np.ndarray]: G-ACF lag timeseries and correlations
        """

        min_lag = kwargs.get("min_lag", None)
        max_lag = kwargs.get("max_lag", None)
        lag_resolution = kwargs.get("lag_resolution", None)
        selection_function = kwargs.get("selection_function", "natural")
        weight_function = kwargs.get("weight_function", "fractional")
        alpha = kwargs.get("alpha", None)

        lag_timeseries, correlations = self._gacf.autocorrelation(
            min_lag=min_lag,
            max_lag=max_lag,
            lag_resolution=lag_resolution,
            selection_function=selection_function,
            weight_function=weight_function,
            alpha=alpha,
        )

        return lag_timeseries, correlations

    def __call__(self, **kwargs) -> PeriodResult:
        """Overrides parent call method to allow 2-stage period extraction."""

        lag_timeseries, correlations = self.calculate_autocorrelation(**kwargs)
        fft = FFTPeriodFinder(lag_timeseries, correlations)
        fft_period = fft(**kwargs)
        return PeriodResult(
            period=fft_period.period,
            neg_error=fft_period.neg_error,
            pos_error=fft_period.pos_error,
            method=self.__class__.__name__,
        )
