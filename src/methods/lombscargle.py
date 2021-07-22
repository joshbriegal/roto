import numpy as np

from astropy.timeseries import LombScargle
from typing import Optional, Tuple

from src.methods.abstractperiodfinder import PeriodFinder


class LombScarglePeriodFinder(PeriodFinder):
    """LombScargle method to find periods.
    Conforms to PeriodFinder interface.
    """

    def __init__(
        self,
        timeseries: np.ndarray,
        flux: np.ndarray,
        flux_errors: Optional[np.ndarray] = None,
        fit_mean: Optional[bool] = None,
        center_data: Optional[bool] = None,
        nterms: Optional[bool] = None,
        normalization: Optional[bool] = None,
    ):
        """
        Args:
            timeseries (np.ndarray): [description]
            flux (np.ndarray): [description]
            flux_errors (Optional[np.ndarray], optional): [description]. Defaults to None.
            fit_mean (Optional[bool], optional): [description]. Defaults to None.
            center_data (Optional[bool], optional): [description]. Defaults to None.
            nterms (Optional[bool], optional): [description]. Defaults to None.
            normalization (Optional[bool], optional): [description]. Defaults to None.
        """
        super().__init__(timeseries, flux, flux_errors)

        self._lombscargle = LombScargle(
            self.timeseries,
            self.flux,
            dy=self.flux_errors,
            fit_mean=fit_mean,
            center_data=center_data,
            nterms=nterms,
            normalization=normalization,
        )

    def __call__(
        self,
        method="auto",
        method_kwds=None,
        normalization=None,
        samples_per_peak=5,
        nyquist_factor=5,
        minimum_frequency=None,
        maximum_frequency=None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """[summary]

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
        return self._lombscargle.autopower(
            method=method,
            method_kwds=method_kwds,
            normalization=normalization,
            samples_per_peak=samples_per_peak,
            nyquist_factor=nyquist_factor,
            minimum_frequency=minimum_frequency,
            maximum_frequency=maximum_frequency,
        )
