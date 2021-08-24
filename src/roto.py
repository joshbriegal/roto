import inspect
from typing import List, Optional

import numpy as np
import pandas as pd

from src.methods.fft import FFTPeriodFinder
from src.methods.gacf import GACFPeriodFinder
from src.methods.gaussianprocess import GPPeriodFinder
from src.methods.lombscargle import LombScarglePeriodFinder
from src.methods.periodfinder import PeriodFinder, PeriodResult


class RoTo:

    METHODS = {
        "lombscargle": LombScarglePeriodFinder,
        "fft": FFTPeriodFinder,
        "gacf": GACFPeriodFinder,
        "gp": GPPeriodFinder,
    }

    def __init__(
        self,
        timeseries: np.ndarray,
        flux: np.ndarray,
        flux_errors: Optional[np.ndarray] = None,
        methods_parameters: Optional[dict] = None,
    ):

        self.timeseries = timeseries
        self.flux = flux
        self.flux_errors = flux_errors

        self.methods = self._parse_constructor_parameters(methods_parameters)

        self.periods = None

    def _parse_constructor_parameters(self, methods_parameters: Optional[dict]) -> list:
        if methods_parameters is None:
            return [
                method(self.timeseries, self.flux, self.flux_errors)
                for method in self.METHODS.values()
            ]

        methods = []
        for method, kwargs in methods_parameters.items():
            methods.append(
                self.METHODS[method](
                    self.timeseries, self.flux, self.flux_errors, **kwargs
                )
            )

        return methods

    def __call__(self, **kwargs):

        self.periods = [method(**kwargs) for method in self.methods]

    def periods_to_table(self):

        columns = {"period": [], "neg_error": [], "pos_error": [], "method": []}

        if not self.periods:
            return pd.DataFrame()

        for period_result in self.periods:
            columns["period"].append(period_result.period)
            columns["neg_error"].append(period_result.neg_error)
            columns["pos_error"].append(period_result.pos_error)
            columns["method"].append(period_result.method)

        period_df = pd.DataFrame.from_dict(columns)

        return period_df

    def __str__(self):
        return self.periods_to_table().to_string(index=False)
