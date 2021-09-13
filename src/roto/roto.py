import logging
from itertools import cycle
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import ScalarFormatter
from scipy.stats import gaussian_kde, median_abs_deviation

from roto.methods.fft import FFTPeriodFinder
from roto.methods.gacf import GACFPeriodFinder
from roto.methods.gaussianprocess import GPPeriodFinder
from roto.methods.lombscargle import LombScarglePeriodFinder
from roto.methods.periodfinder import PeriodResult
from roto.plotting.plotting_tools import (
    calculate_phase,
    create_axis_with_formatter,
    split_phase,
)

DEFAULT_COLOUR_CYCLE = cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])

logger = logging.getLogger(__name__)


class RoTo:

    METHODS = {
        "lombscargle": LombScarglePeriodFinder,
        "fft": FFTPeriodFinder,
        "gacf": GACFPeriodFinder,
        "gp": GPPeriodFinder,  # keep at end of dictionary to allow seed period generation from other methods.
    }

    METHOD_NAMES = {
        "lombscargle": "Lomb-Scargle",
        "fft": "Fast Fourier Transform",
        "gacf": "G-ACF",
        "gp": "Gaussian Process Regression",
    }

    PLOTTING_COLOURS = {method: next(DEFAULT_COLOUR_CYCLE) for method in METHODS}

    def __init__(
        self,
        timeseries: np.ndarray,
        flux: np.ndarray,
        flux_errors: Optional[np.ndarray] = None,
        methods_parameters: Optional[dict] = None,
        name: str = "Unnamed RoTo Object",
        time_units: str = "days",
        flux_units: str = "relative flux units",
    ):
        self.name = name

        self.timeseries = timeseries
        self.flux = flux
        self.flux_errors = flux_errors

        timeseries_diffs = np.diff(self.timeseries)
        self.regular_sampling = (timeseries_diffs.max() - timeseries_diffs.min()) < 1e-5
        self.time_units = time_units
        if self.time_units != "days":
            logger.warning(
                "GP prior scaled to expect timeseries data in days. Check prior or convert units."
            )
        self.flux_units = flux_units
        if self.flux_units != "relative flux units":
            logger.warning(
                "GP prior scaled to expect flux data in relative flux units. Check prior or convert units."
            )

        self.methods = self._parse_constructor_parameters(methods_parameters)

        self.periods = {}

    def _parse_constructor_parameters(
        self,
        methods_parameters: Optional[dict] = None,
    ) -> dict:
        if methods_parameters is None:
            return {
                name: method(
                    self.timeseries,
                    self.flux,
                    self.flux_errors,
                    time_units=self.time_units,
                    flux_units=self.flux_units,
                )
                for name, method in self.METHODS.items()
                if (name != "fft") or (self.regular_sampling)
            }

        methods = {}
        if list(methods_parameters.keys()) == ["gp"]:
            # if just a GP, use a lomb scargle also to seed GP period.
            methods_parameters = {"lombscargle": {}, **methods_parameters}

        for method, kwargs in methods_parameters.items():
            methods[method] = self.METHODS[method](
                self.timeseries,
                self.flux,
                self.flux_errors,
                time_units=self.time_units,
                flux_units=self.flux_units,
                **kwargs,
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
            try:
                self.periods[name] = method(**kwargs)
            except Exception as e:
                logger.error("Unable to run method %s" % name)
                logger.error(e, exc_info=True)
                continue

    def periods_to_table(self) -> pd.DataFrame:
        """Convert roto.periods into a DataFrame for display.

        Returns:
            pd.DataFrame: Dataframe with all outputted periods
        """

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
            method (str, optional): method should be one of 'mean', 'median' or a period finding method. Defaults to "mean".
            include (Optional[List], optional): Method outputs to include. Defaults to [].
            exclude (Optional[List], optional): Method outputs to exclude. Defaults to [].

        Raises:
            ValueError: If method specified incorrect.

        Returns:
            PeriodResult: CombinedPeriodResult.
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
        elif method in self.periods:
            return self.periods[method]

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
        summary: bool = False,
        scientific: bool = False,
        return_fig_ax: bool = False,
    ) -> Union[None, Tuple[Figure, Dict]]:
        """Generate summary plot of RoTo object run.

        Args:
            savefig (bool, optional): Save figure to pdf. Defaults to False.
            filename (Optional[str], optional): Name of pdf. Defaults to None.
            include (Optional[List], optional): Methods to include. Defaults to None (all methods).
            exclude (Optional[List], optional): Methods to exclude. Defaults to None (no methods).
            plot_gp (bool, optional): Plot Gaussian Process prediction & residuals. Defaults to True.
            show (bool, optional): Show interactive plot. Defaults to True.
            summary (bool, optional): Just plot summary, no methods. Defaults to False.
            scientific (bool, optional): Scientific formatting of numbers vs linear scale. Defaults to False.
            return_fig_ax (bool, optional): Return figure and axis tuples for further processing. Defaults to False.

        Returns:
            Union[None, Tuple(Figure, Dict)]: None or a tuple (matplotlib figure, dictionary of matplotlib axes)
        """

        if savefig and not filename:
            filename = f"{self.name}.pdf"

        if (not include) or (not self.periods):
            include = list(self.periods.keys())

        fig, ax_dict = self._setup_figure(
            include=include,
            exclude=exclude,
            summary=summary,
            scientific=scientific,
            plot_gp=plot_gp,
        )

        self.plot_data(ax_dict["data"])
        self.plot_periods(ax_dict["distributions"])
        self.plot_phase_folded_data(ax_dict["phase_fold"], self.best_period().period)

        if not summary:
            for method_name, method in self.methods.items():
                if method_name == "gp":
                    if plot_gp:
                        ax_dict["data"].get_xaxis().set_visible(False)
                        method.plot_gp_predictions(
                            ax_dict["data"], colour=self.PLOTTING_COLOURS[method_name]
                        )
                        method.plot_gp_residuals(
                            ax_dict["residuals"],
                            colour=self.PLOTTING_COLOURS[method_name],
                        )

                method.plot(
                    ax_dict[method_name]["method"],
                    self.periods[method_name],
                    colour=self.PLOTTING_COLOURS[method_name],
                )
                self.plot_phase_folded_data(
                    ax_dict[method_name]["phase_fold"],
                    self.periods[method_name].period,
                )

        if savefig:
            fig.savefig(filename, bbox_inches="tight", pad_inches=0.25)
        if show:
            plt.show()

        if return_fig_ax:
            return fig, ax_dict

    def plot_summary(
        self,
        savefig: bool = False,
        filename: Optional[str] = None,
        plot_gp: bool = True,
        show: bool = True,
    ):
        """Helper function to create just summary plots, same as calling self.plot(summary=True)

        Args:
            savefig (bool, optional): Save figure to pdf. Defaults to False.
            filename (Optional[str], optional): Name of pdf. Defaults to None.
            plot_gp (bool, optional): Plot Gaussian Process prediction & residuals. Defaults to True.
            show (bool, optional): Show interactive plot. Defaults to True.
        """
        self.plot(
            savefig=savefig, filename=filename, plot_gp=plot_gp, show=show, summary=True
        )

    def plot_periods(self, ax: Axes) -> Axes:
        """Plot figure comparing outputted periods and errors.

        Args:
            ax (Axes): Matplotlib axis

        Returns:
            Axes: Matplotlib axis
        """
        for name, period in self.periods.items():
            if period.period_distribution is not None:
                # plot as distribution
                density = gaussian_kde(period.period_distribution)
                pmin = max(0, period.period - 5 * period.neg_error)
                pmax = period.period + 5 * period.pos_error
                xs = np.linspace(pmin, pmax, 100)
                kde_plot = density(xs)
                kde_plot *= 1.0 / kde_plot.max()
                ax.plot(xs, kde_plot, color=self.PLOTTING_COLOURS[name])
            ax.axvline(
                period.period,
                label=self.METHOD_NAMES[name],
                color=self.PLOTTING_COLOURS[name],
            )
            ax.axvspan(
                period.period - period.neg_error,
                period.period + period.pos_error,
                color=self.PLOTTING_COLOURS[name],
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
        two_sided_error = np.average([best_period.neg_error, best_period.pos_error])
        ax.set_title(
            f"Adopted Period: {best_period.period:.2f} ± {two_sided_error:.2f} {self.time_units}"
        )

        ax.legend()

        return ax

    def plot_gp_diagnostics(
        self,
        show: bool = True,
        savefig: bool = False,
        filename: str = "",
        fileext: str = "pdf",
    ):
        """Plot Gaussian Process diagnostic outputs figures.

        Args:
            show (bool, optional): Show interactive plot. Defaults to True.
            savefig (bool, optional): Save figure to pdf. Defaults to False.
            filename (Optional[str], optional): Name of pdf. Defaults to None.
            fileext (str, optional): File extension to save figure. Defaults to "pdf".

        Raises:
            RuntimeError: If no GP found.
        """
        if "gp" not in self.methods:
            raise RuntimeError("Cannot plot GP diagnostics, no GP method found.")

        if savefig and not filename:
            filename = f"{self.name}.pdf"

        try:
            self.methods["gp"].plot_trace(
                show=show, savefig=savefig, filename=filename, fileext=fileext
            )
        except RuntimeError as trace_err:
            logger.error("Unable to plot trace")
            logger.error(trace_err, exc_info=True)
        try:
            self.methods["gp"].plot_distributions(
                show=show, savefig=savefig, filename=filename, fileext=fileext
            )
        except (RuntimeError, ValueError) as dist_err:
            logger.error("Unable to plot GP distributions")
            logger.error(dist_err, exc_info=True)

    def plot_data(self, ax: Axes) -> Axes:
        """Scatter plot of input data.

        Args:
            ax (Axes): Matplotlib axis

        Returns:
            Axes: Matplotlib axis
        """
        ax.errorbar(
            self.timeseries,
            self.flux,
            self.flux_errors,
            markersize=1,
            errorevery=2,
            linestyle=" ",
            marker=".",
            color="k",
            capsize=1,
            elinewidth=1,
        )
        ax.set_xlabel(f"Time / {self.time_units}")
        ax.set_ylabel(f"Flux / {self.flux_units}")

        ymin = np.min(self.flux - self.flux_errors)
        ymax = np.max(self.flux + self.flux_errors)
        yextent = ymax - ymin
        ax.set_ylim([ymin - (yextent * 0.01), ymax + (yextent * 0.01)])

        return ax

    def plot_phase_folded_data(self, ax: Axes, period: float, epoch: float = 0) -> Axes:
        """Plot data phase folded on period and epoch.
        Colour scale incremented for each period.

        Args:
            ax (Axes): Matplotlib axis
            period (float): Period on which to phase fold.
            epoch (float, optional): Epoch on which to phase fold. Defaults to 0.

        Returns:
            Axes: Matplotlib axis
        """
        phased_timeseries = calculate_phase(self.timeseries, period, epoch)
        split_phases, split_flux = split_phase(phased_timeseries, self.flux)
        colours = iter(cm.viridis(np.r_[0 : 1 : len(split_phases) * 1j]))

        for phase, flux in zip(split_phases, split_flux):
            ax.scatter(phase, flux, color=next(colours), s=1)

        ax.set_title(f"Period: {period:.4f} {self.time_units}")
        ax.set_xlim([0, 1])
        ax.set_xlabel("Phase")
        ax.set_ylabel(f"Flux / {self.flux_units}")

        return ax

    def _setup_figure(
        self,
        include: Optional[List] = [],
        exclude: Optional[List] = [],
        summary: bool = False,
        scientific: bool = False,
        plot_gp: bool = False,
    ):

        unit_grid_width = 5
        unit_grid_height = 1

        data_plot_size = (2, 3)  # in units of grid width, height
        residuals_plot_size = (2, 1)
        distributions_plot_size = (1, 3)
        phase_fold_plot_size = (1, 3)
        method_plot_size = (1, 3)
        spacer_plot_size = (2, 1)

        if summary:
            # just plot summary stats, no method plots.
            methods = {}
        else:
            methods = {name: method for name, method in self.methods.items()}
            if include:
                methods = {
                    name: method for name, method in methods.items() if name in include
                }
            if exclude:
                methods = {
                    name: method
                    for name, method in methods.items()
                    if name not in exclude
                }

        n_grid_units_width = 2
        n_grid_units_height = (
            data_plot_size[1]
            + (residuals_plot_size[1] * int(plot_gp))
            + distributions_plot_size[1]
            + method_plot_size[1] * len(methods)
            + spacer_plot_size[1] * (1 + len(methods))
        )

        figsize = (
            unit_grid_width * n_grid_units_width,
            unit_grid_height * n_grid_units_height,
        )

        fig = plt.figure(figsize=figsize)
        gridspec = fig.add_gridspec(n_grid_units_height, n_grid_units_width)
        plt.subplots_adjust(hspace=0.0, wspace=0.2)

        axes = {}
        formatter = ScalarFormatter()
        formatter.set_scientific(scientific)
        height = 0
        axes["data"] = create_axis_with_formatter(
            fig, gridspec[height : height + data_plot_size[1], :], formatter
        )
        height += data_plot_size[1]
        if plot_gp:
            axes["residuals"] = create_axis_with_formatter(
                fig,
                gridspec[height : height + residuals_plot_size[1], :],
                formatter,
                sharex=axes["data"],
            )
            height += residuals_plot_size[1]
        height += spacer_plot_size[1]

        axes["distributions"] = create_axis_with_formatter(
            fig, gridspec[height : height + distributions_plot_size[1], 0], formatter
        )
        axes["phase_fold"] = create_axis_with_formatter(
            fig, gridspec[height : height + phase_fold_plot_size[1], 1], formatter
        )
        height += phase_fold_plot_size[1]
        height += spacer_plot_size[1]

        for method in methods:
            axes[method] = {
                "method": create_axis_with_formatter(
                    fig, gridspec[height : height + method_plot_size[1], 0], formatter
                ),
                "phase_fold": create_axis_with_formatter(
                    fig, gridspec[height : height + method_plot_size[1], 1], formatter
                ),
            }
            height += method_plot_size[1]
            height += spacer_plot_size[1]

        axes["data"].set_title(self.name)

        return fig, axes
