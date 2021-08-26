from typing import List, Optional

import numpy as np
import pandas as pd
from scipy.stats import median_abs_deviation, gaussian_kde
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from itertools import cycle

from src.methods.fft import FFTPeriodFinder
from src.methods.gacf import GACFPeriodFinder
from src.methods.gaussianprocess import GPPeriodFinder
from src.methods.lombscargle import LombScarglePeriodFinder
from src.methods.periodfinder import PeriodResult
from src.plotting.plotting_tools import split_phase, calculate_phase


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
        name: Optional[str] = "Unnamed RoTo Object",
    ):
        self.name = name

        self.timeseries = timeseries
        self.flux = flux
        self.flux_errors = flux_errors

        timeseries_diffs = np.diff(self.timeseries)
        self.regular_sampling = (timeseries_diffs.max() - timeseries_diffs.min()) < 1e-5

        self.methods = self._parse_constructor_parameters(methods_parameters)

        self.periods = {}

    def _parse_constructor_parameters(self, methods_parameters: Optional[dict]) -> dict:
        if methods_parameters is None:
            return {
                name: method(self.timeseries, self.flux, self.flux_errors)
                for name, method in self.METHODS.items()
                if (name != "fft") or (self.regular_sampling)
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
                        [
                            period_result.period
                            for period_result in self.periods.values()
                        ]
                    )
                    kwargs["gp_seed_period"] = average_period

            self.periods[name] = method(**kwargs)

    def periods_to_table(self):

        columns = {"period": [], "neg_error": [], "pos_error": [], "method": []}

        if not self.periods:
            return pd.DataFrame()

        for period_result in self.periods.values():
            columns["period"].append(period_result.period)
            columns["neg_error"].append(period_result.neg_error)
            columns["pos_error"].append(period_result.pos_error)
            columns["method"].append(period_result.method)

        period_df = pd.DataFrame.from_dict(columns)

        return period_df

    def __str__(self):
        return self.periods_to_table().to_string(index=False)

    def best_period(
        self,
        method: str = "mean",
        include: Optional[List] = None,
        exclude: Optional[List] = None,
    ) -> PeriodResult:
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

        periods_to_use = self.periods.values()

        try:
            if include:
                include_classes = [
                    self.METHODS[method_to_include].__name__
                    for method_to_include in include
                ]
                periods_to_use = [
                    period_result
                    for period_result in periods_to_use
                    if period_result.method in include_classes
                ]
            if exclude:
                exclude_classes = [
                    self.METHODS[method_to_exclude].__name__
                    for method_to_exclude in exclude
                ]
                periods_to_use = [
                    period_result
                    for period_result in periods_to_use
                    if period_result.method not in exclude_classes
                ]

            if not periods_to_use:
                raise ValueError(
                    "Provided incompatible list of include / exclude values. No best period calculated. \n include: {include} \n exclude: {exclude}"
                )

        except KeyError:
            raise ValueError(
                f"Unable to parse include / exclude values given. \n include: {include} \n exclude: {exclude}"
            )

        if method == "mean":
            mean = np.mean([p.period for p in periods_to_use])
            std = np.std([p.period for p in periods_to_use]) / np.sqrt(
                len(periods_to_use)
            )
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

    def plot(
        self,
        savefig: bool = False,
        filename: Optional[str] = None,
        include: Optional[List] = None,
        exclude: Optional[List] = None,
        plot_gp: bool = True,
        show: bool = True,
    ):

        if savefig and not filename:
            filename = f"{self.name}.pdf"

        if (not include) or (not self.periods):
            include = list(self.periods.keys())

        fig, ax_dict = self._setup_figure(include=include, exclude=exclude)

        self.plot_data(ax_dict["data"])
        self.plot_periods(ax_dict["distributions"])
        self.plot_phase_folded_data(
            ax_dict["phase_fold"], self.best_period().period
        )

        for method_name, method in self.methods.items():
            if method_name == "gp":
                if plot_gp:
                    pass
                    # method.plot_gp_predictions(ax_dict["data"])
            method.plot(
                ax_dict[method_name]["method"],
                self.periods[method_name],
            )
            self.plot_phase_folded_data(
                ax_dict[method_name]["phase_fold"],
                self.periods[method_name].period,
            )

        fig.tight_layout()
        if savefig:
            fig.savefig(filename)
        if show:
            plt.show()

    def plot_periods(self, ax):
        colours = cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])
        for name, period in self.periods.items():
            color = next(colours)
            if period.period_distribution is not None:
                # plot as distribution
                density = gaussian_kde(period.period_distribution)
                pmin = max(0, period.period - 5 * period.neg_error)
                pmax = period.period + 5 * period.pos_error
                xs = np.linspace(pmin, pmax, 100)
                kde_plot = density(xs)
                kde_plot *= 1.0 / kde_plot.max()
                ax.plot(xs, kde_plot, color=color)
            ax.axvline(period.period, label=name, color=color)
            ax.axvspan(
                period.period - period.neg_error,
                period.period + period.pos_error,
                color=color,
                alpha=0.2,
            )

        # plot best period as a single point with error bars
        best_period = self.best_period()
        ax.errorbar(
            best_period.period,
            0.5,
            xerr=[[best_period.neg_error], [best_period.pos_error]],
            ms=10,
            marker="s",
            c="k",
            capsize=10,
        )

        ax.set_xlim(
            [
                best_period.period - 5 * best_period.neg_error,
                best_period.period + 5 * best_period.pos_error,
            ]
        )
        ax.set_ylim([0, 1])

        ax.get_yaxis().set_visible(False)
        ax.set_xlabel("Period")
        ax.set_title(str(best_period))

        ax.legend()

    def plot_data(self, ax):
        ax.scatter(self.timeseries, self.flux, s=1)
        ax.set_xlabel("Time")
        ax.set_ylabel("Flux")

    def plot_phase_folded_data(self, ax, period: float, epoch: float = 0):
        phased_timeseries = calculate_phase(self.timeseries, period, epoch)
        split_phases, split_flux = split_phase(phased_timeseries, self.flux)
        colours = iter(cm.viridis(np.r_[0 : 1 : len(split_phases) * 1j]))

        for phase, flux in zip(split_phases, split_flux):
            ax.scatter(phase, flux, color=next(colours), s=1)

        ax.set_title(f"Phase folded on {period:.2f}")
        ax.set_xlim([0, 1])
        ax.set_xlabel("Phase")
        ax.set_ylabel("Flux")

    def _setup_figure(self, include: Optional[List] = [], exclude: Optional[List] = []):

        unit_grid_width = 5
        unit_grid_height = 1

        data_plot_size = (2, 2)  # in units of grid width, height
        distributions_plot_size = (1, 3)
        phase_fold_plot_size = (1, 3)
        method_plot_size = (1, 3)

        methods = {name: method for name, method in self.methods.items()}
        if include:
            methods = {
                name: method for name, method in methods.items() if name in include
            }
        if exclude:
            methods = {
                name: method for name, method in methods.items() if name not in exclude
            }

        n_grid_units_width = 2
        n_grid_units_height = (
            data_plot_size[1]
            + distributions_plot_size[1]
            + method_plot_size[1] * len(methods)
        )

        figsize = (
            unit_grid_width * n_grid_units_width,
            unit_grid_height * n_grid_units_height,
        )

        fig = plt.figure(figsize=figsize)
        gridspec = fig.add_gridspec(n_grid_units_height, n_grid_units_width)

        axes = {}
        height = 0
        axes["data"] = fig.add_subplot(gridspec[height : height + data_plot_size[1], :])
        height += data_plot_size[1]
        axes["distributions"] = fig.add_subplot(
            gridspec[height : height + distributions_plot_size[1], 0]
        )
        axes["phase_fold"] = fig.add_subplot(
            gridspec[height : height + phase_fold_plot_size[1], 1]
        )
        height += phase_fold_plot_size[1]

        for method in methods:
            axes[method] = {
                "method": fig.add_subplot(
                    gridspec[height : height + method_plot_size[1], 0]
                ),
                "phase_fold": fig.add_subplot(
                    gridspec[height : height + method_plot_size[1], 1]
                ),
            }
            height += method_plot_size[1]

        fig.suptitle(self.name)

        return fig, axes
