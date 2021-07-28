from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import median_abs_deviation

from src.methods.fft import FFTPeriodFinder
from src.methods.gacf import GACFPeriodFinder
from src.methods.gaussianprocess import GPPeriodFinder
from src.methods.lombscargle import LombScarglePeriodFinder
from src.methods.periodfinder import PeriodResult


class RoTo:

    METHODS = {
        "lombscargle": LombScarglePeriodFinder,
        "fft": FFTPeriodFinder,
        "gacf": GACFPeriodFinder,
        "gp": GPPeriodFinder,  # keep at end of dictionary to allow seed period generation from other methods.
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

        self.periods = []

    def _parse_constructor_parameters(self, methods_parameters: Optional[dict]) -> dict:
        if methods_parameters is None:
            return {
                name: method(self.timeseries, self.flux, self.flux_errors)
                for name, method in self.METHODS.items()
            }

        methods = {}
        if list(methods_parameters.keys()) == ["gp"]:
            # if just a GP, use a lomb scargle also to seed GP period.
            methods_parameters = {"lombscargle": {}, **methods_parameters}

        for method, kwargs in methods_parameters.items():
            methods[method] = self.METHODS[method](
                self.timeseries, self.flux, self.flux_errors, **kwargs
            )

        return methods

    def __call__(self, **kwargs):

        for name, method in self.methods.items():
            if name == "gp":
                if "gp_seed_period" not in kwargs:
                    average_period = np.median(
                        [period_result.period for period_result in self.periods]
                    )
                    kwargs["gp_seed_period"] = average_period

            self.periods.append(method(**kwargs))

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

    def best_period(self, method: str = "mean") -> PeriodResult:
        if not self.periods:
            return None

        if method == "mean":
            mean = np.mean([p.period for p in self.periods])
            std = np.std([p.period for p in self.periods]) / np.sqrt(len(self.periods))
            return PeriodResult(
                period=mean, neg_error=std, pos_error=std, method="CombinedPeriodResult"
            )
        elif method == "median":
            median = np.median([p.period for p in self.periods])
            std = (
                1.4826
                * median_abs_deviation([p.period for p in self.periods])
                / np.sqrt(len(self.periods))
            )
            return PeriodResult(
                period=median,
                neg_error=std,
                pos_error=std,
                method="CombinedPeriodResult",
            )
        elif method in self.METHODS:
            for periodresult in self.periods:
                if periodresult.method == self.METHODS[method].__name__:
                    return periodresult
        else:
            raise ValueError(
                f"Parameter 'method' must be one of ['mean', 'median'] or {list(self.METHODS.keys())}]"
            )
