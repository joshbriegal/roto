from typing import List, Optional

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

    def best_period(self, method: str = "mean", include: Optional[List] =  [], exclude: Optional[List] = []) -> PeriodResult:
        """Calculate best period based on methods already run. If called before 
        running the period finding methods, will return None.

        Args:
            method (str, optional): [description]. Defaults to "mean".
            include (Optional[List], optional): [description]. Defaults to [].
            exclude (Optional[List], optional): [description]. Defaults to [].

        Raises:
            ValueError: [description]

        Returns:
            PeriodResult: [description]
        """
        if not self.periods:
            return None

        periods_to_use = self.periods

        try:
            if include:
                include_classes = [self.METHODS[method_to_include].__name__ for method_to_include in include]
                periods_to_use = [period_result for period_result in periods_to_use if period_result.method in include_classes]
            if exclude:
                exclude_classes = [self.METHODS[method_to_exclude].__name__ for method_to_exclude in exclude]
                periods_to_use = [period_result for period_result in periods_to_use if period_result.method not in exclude_classes]

            if not periods_to_use:
                raise ValueError("Provided incompatible list of include / exclude values. No best period calculated. \n include: {include} \n exclude: {exclude}")

        except KeyError:
            raise ValueError(f"Unable to parse include / exclude values given. \n include: {include} \n exclude: {exclude}")

        if method == "mean":
            mean = np.mean([p.period for p in periods_to_use])
            std = np.std([p.period for p in periods_to_use]) / np.sqrt(len(periods_to_use))
            return PeriodResult(
                period=mean, neg_error=std, pos_error=std, method="CombinedPeriodResult"
            )
        elif method == "median":
            median = np.median([p.period for p in periods_to_use])
            std = (
                1.4826
                * median_abs_deviation([p.period for p in periods_to_use])
                / np.sqrt(len(periods_to_use))
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

        raise ValueError(
            f"Parameter 'method' must be one of ['mean', 'median'] or {list(self.METHODS.keys())}]. Did you specify a period extraction method not run?"
        )
