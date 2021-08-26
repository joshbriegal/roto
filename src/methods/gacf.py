from typing import Optional, Tuple

import numpy as np
from astropy.convolution import Gaussian1DKernel, convolve
from gacf import GACF
from scipy.stats import median_abs_deviation

from src.methods.fft import FFTPeriodFinder
from src.methods.periodfinder import PeriodFinder, PeriodResult


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
        self.lag_timeseries = None
        self.correlations = None

    def calculate_periodogram(self, **kwargs) -> None:
        """A "periodogram" does not exist for an ACF
        Returns:
            None
        """
        return None

    def calculate_autocorrelation(self, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
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

        min_lag = kwargs.get("min_lag", 0)
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

        return np.array(lag_timeseries), np.array(correlations)

    def __call__(self, gacf_method="fft", **kwargs) -> PeriodResult:
        """Overrides parent call method to allow 2-stage period extraction.

        Args:
            method (str, optional): Method used to get final period values. Defaults to "fft".
                - "fft" will use an FFT on the G-ACF
                - "peaks" will find peaks within the G-ACF itself.

        Returns:
            PeriodResult: [description]
        """

        self.lag_timeseries, self.correlations = self.calculate_autocorrelation(**kwargs)
        if gacf_method == "fft":
            fft = FFTPeriodFinder(self.lag_timeseries, self.correlations)
            fft_period = fft(**kwargs)
            return PeriodResult(
                period=fft_period.period,
                neg_error=fft_period.neg_error,
                pos_error=fft_period.pos_error,
                method=self.__class__.__name__,
            )
        elif gacf_method == "peaks":
            return self.find_acf_peaks(self.lag_timeseries, self.correlations)

    def find_acf_peaks(
        self, lag_timeseries: np.ndarray, correlation: np.ndarray
    ) -> PeriodResult:
        """Method taken from McQuillan 2013:
            Convolve ACF with Gaussian Kernel
            Identify peaks in ACF
            Select peak associated with mean rotation period
            Evaluate uncertainty on error

        Args:
            lag_timeseries (np.ndarray): Lag time series. Must be positive side only.
            correlation (np.ndarray): Correlations

        Returns:
            PeriodResult: [description]
        """
        gaussian_fwhm = lag_timeseries[18] - lag_timeseries[0]
        gauss_kernel = Gaussian1DKernel(
            gaussian_fwhm, x_size=(np.ceil(gaussian_fwhm * (57 / 18)) // 2 * 2 + 1)
        )
        smoothed_correlations = convolve(correlation, gauss_kernel)

        acf_peak_indexes = self.calculate_peak_indexes(
            smoothed_correlations, sort=False
        )

        # Remove zero point as not a real peak
        acf_peak_indexes = np.delete(acf_peak_indexes, 0)

        if len(acf_peak_indexes) <= 1:
            # just one peak, use width of Gaussian as stdev
            # find left min:
            central_index = acf_peak_indexes[0]
            left_idx = acf_peak_indexes[0]
            value = smoothed_correlations[left_idx]
            while value > 0.5 * smoothed_correlations[central_index]:
                try:
                    value = smoothed_correlations[left_idx]
                    left_idx -= 1
                except IndexError:
                    left_idx = None
                    break
            # find right min:
            right_idx = acf_peak_indexes[0]
            value = smoothed_correlations[right_idx]
            while value > 0.5 * smoothed_correlations[central_index]:
                try:
                    value = smoothed_correlations[right_idx]
                    right_idx += 1
                except IndexError:
                    right_idx = None
                    break
            sigma_p = 0
            if left_idx and right_idx:
                sigma_p = lag_timeseries[right_idx] - lag_timeseries[left_idx]

            return PeriodResult(
                period=lag_timeseries[acf_peak_indexes[0]],
                neg_error=sigma_p,
                pos_error=sigma_p,
                method=self.__class__.__name__,
            )

        peak_lags = lag_timeseries[acf_peak_indexes]
        local_heights = np.zeros(len(acf_peak_indexes))

        for i, peak_idx in enumerate(acf_peak_indexes):

            # find left min:
            left_idx = peak_idx
            diff = smoothed_correlations[left_idx] - smoothed_correlations[left_idx - 1]
            while diff > 0:
                try:
                    diff = (
                        smoothed_correlations[left_idx]
                        - smoothed_correlations[left_idx - 1]
                    )
                    left_idx -= 1
                except IndexError:
                    left_idx = None
                    break
            if left_idx:
                left_height = correlation[peak_idx] - correlation[left_idx]

            # find right min:
            right_idx = peak_idx
            diff = (
                smoothed_correlations[right_idx] - smoothed_correlations[right_idx + 1]
            )
            while diff > 0:
                try:
                    diff = (
                        smoothed_correlations[right_idx]
                        - smoothed_correlations[right_idx + 1]
                    )
                    right_idx += 1
                except IndexError:
                    right_idx = None
                    break
            if right_idx:
                right_height = correlation[peak_idx] - correlation[right_idx]
            if left_height and right_height:
                local_heights[i] = (left_height + right_height) / 2
            elif right_height:
                local_heights[i] = right_height
            elif left_height:
                local_heights[i] = left_height
            else:
                local_heights[i] = np.nan

        first_lag = peak_lags[0]
        second_lag = peak_lags[1]
        p_start = 0

        if not (2 * first_lag * 0.8) <= second_lag <= (2 * first_lag * 1.2):
            if local_heights[1] > local_heights[0]:
                p_start = 1

        valid_peaks = [peak_lags[p_start]]
        valid_peak_indexes = [p_start]
        gap = 0
        peak_number = 2
        for i in range(1, len(peak_lags)):
            if (i + p_start) >= len(peak_lags):
                break
            if len(valid_peaks) >= 10:
                break
            if i + p_start - 1 >= 0:
                gap = peak_lags[i + p_start] - peak_lags[i + p_start - 1]
            gap_ratio = gap / peak_lags[p_start]

            if (
                (peak_lags[p_start] * 0.8 * peak_number)
                <= peak_lags[i + p_start]
                <= (peak_lags[p_start] * 1.2 * peak_number)
            ):
                if gap_ratio > 0.3:
                    valid_peaks.append(peak_lags[i + p_start] / peak_number)
                    valid_peak_indexes.append(i + p_start)
                    peak_number += 1

        # use median / MAD estimate from multiple peaks.
        mad = median_abs_deviation(valid_peaks)
        sigma_p = 1.483 * mad / np.sqrt(len(valid_peaks) - 1)
        med_p = np.median(valid_peaks)

        return PeriodResult(
            period=med_p,
            neg_error=sigma_p,
            pos_error=sigma_p,
            method=self.__class__.__name__,
        )

    def plot(self, ax, period: PeriodResult) -> None:
        """Given a figure and an axis plot the interesting output of the object.

        Args:
            ax ([type]): Matplotlib axis
            period (PeriodResult): Outputted period to plot around
        """
        if (self.lag_timeseries is None) or (self.correlations is None):
            self()

        ax.scatter(self.lag_timeseries, self.correlations, s=1)

        ax.axvline(period.period, color="orange", lw=1)
        ax.axvspan(period.period - period.neg_error, period.period + period.pos_error, color='orange', alpha=0.5)

        ax.set_xlim([0, min(5 * period.period, self.lag_timeseries[-1])])

        ax.set_xlabel("Lag time")
        ax.set_ylabel("G-ACF Power")
        ax.set_title("G-ACF")



