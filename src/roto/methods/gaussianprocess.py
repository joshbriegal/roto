import logging
from math import ceil
from os.path import splitext
from typing import Optional, Tuple

import aesara_theano_fallback.tensor as tt
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc3 as pm
import pymc3_ext as pmx
from celerite2.theano import GaussianProcess, terms
from corner import corner
from matplotlib.axes import Axes

from roto.methods.periodfinder import PeriodFinder, PeriodResult
from roto.plotting.plotting_tools import ppt_to_rel_flux, rel_flux_to_ppt

logger = logging.getLogger(__name__)


class GPPeriodFinder(PeriodFinder):
    """Gaussian Process (GP) regression method to find periods.
    Conforms to PeriodFinder interface.
    """

    def __init__(
        self,
        timeseries: np.ndarray,
        flux: np.ndarray,
        flux_errors: Optional[np.ndarray] = None,
        min_ratio_of_maximum_peak_size: float = 0.2,
        samples_per_peak: int = 3,
        time_units: str = "days",
        flux_units: str = "relative flux units",
        gp_seed_period: Optional[float] = None,
    ):
        """
        Args:
            timeseries (np.ndarray): array like time series.
            flux (np.ndarray): array like flux values
            flux_errors (Optional[np.ndarray], optional): array like errors on flux values. Defaults to None.
            gp_seed_period (Optional, float): Starting period for GP. Can also be specified at run time.
        """

        self.model: pm.model = None
        self.solution: pm.Point = None
        self.gp: GaussianProcess = None
        self.trace: pm.backends.base.MultiTrace = None

        self.gp_seed_period = gp_seed_period
        super().__init__(
            timeseries,
            flux,
            flux_errors,
            min_ratio_of_maximum_peak_size,
            samples_per_peak,
            time_units,
            flux_units,
        )
        self.mask = np.ones(len(self.timeseries), dtype=bool)
        self.median_flux = np.nanmedian(self.flux)

        # convert data into ppt format if input units correct
        if self.flux_units == "relative flux units":
            if self.median_flux != 0:
                self.flux_ppt = rel_flux_to_ppt(
                    self.flux,
                    normalise=True,
                    normalisation_value=self.median_flux,
                    center_around=1.0,
                )
                self.flux_errors_ppt = rel_flux_to_ppt(
                    self.flux_errors,
                    normalise=True,
                    normalisation_value=self.median_flux,
                    center_around=0.0,
                )
            else:
                self.flux_ppt = rel_flux_to_ppt(self.flux)
                self.flux_errors_ppt = rel_flux_to_ppt(self.flux_errors)

        elif self.flux_units == "ppt":
            self.flux_ppt = self.flux
            self.flux_errors_ppt = self.flux_errors
        else:
            print(
                "Warning: Not converting units as cannot handle anything other than relative flux units"
            )
            self.flux_ppt = self.flux
            self.flux_errors_ppt = self.flux_errors

    def calculate_periodogram(self, **kwargs) -> None:
        """A "periodogram" does not exist for a GP
        Returns:
            None
        """
        return None

    def __call__(self, **kwargs) -> PeriodResult:
        """Overrides parent call method to allow MAP and MCMC period extraction.

        Args:

        Returns:
            PeriodResult: [description]
        """
        return self.calcuate_gp_period(**kwargs)

    def calcuate_gp_period(self, **kwargs):
        """Calculate the period using a Gaussian Process.

        Args:
            gp_seed_period (Optional, float): Starting period for GP. Can also be specified at instantiation.
            remove_outliers (bool, optional): Defaults to False.
            rms_sigma (float, optional): Defaults to 3.
            do_mcmc (bool, optional): Defaults to False.
            tune (int, optional): Defaults to 500.
            draws (int, optional): Defaults to 500.
            cores (int, optional): Defaults to 1.
            chains (int, optional): Defaults to 2
            target_accept (float, optional): Defaults to 0.9.

        Returns:
            PeriodResult: [description]
        """
        # unpack **kwargs
        self.gp_seed_period = kwargs.get("gp_seed_period", self.gp_seed_period)
        remove_outliers = kwargs.get("remove_outliers", False)
        rms_sigma = kwargs.get("rms_sigma", 3)
        do_mcmc = kwargs.get("do_mcmc", False)
        tune = kwargs.get("tune", 500)
        draws = kwargs.get("draws", 500)
        cores = kwargs.get("cores", 1)
        chains = kwargs.get("chains", 2)
        target_accept = kwargs.get("target_accept", 0.9)

        # compute initial MAP solution
        model, map_soln = self.build_model()

        # (optionally) recompute MAP solution with outliers masked
        if remove_outliers:
            residuals = self.flux_ppt - map_soln["pred"]
            rms = np.sqrt(np.nanmedian(residuals ** 2))
            self.mask = np.abs(residuals) < rms_sigma * rms
            model, map_soln = self.build_model(start=map_soln)

        logger.info("MAP Solution found")
        logger.debug(map_soln)

        # (optionally) run MCMC to sample from posterior
        if do_mcmc and model:
            with model:
                trace = pmx.sample(
                    tune=tune,
                    draws=draws,
                    start=map_soln,
                    cores=cores,
                    chains=chains,
                    target_accept=target_accept,
                    return_inferencedata=True,  # returns an arviz.InferenceData object
                    discard_tuned_samples=True,
                )

            # estimate period and uncertainty
            period_samples = np.asarray(trace.posterior["period"]).flatten()
            percentiles = np.percentile(period_samples, [15.87, 50.0, 84.14])
            med_p = float("{:.5f}".format(percentiles[1]))
            sigma_n = float("{:.5f}".format(percentiles[1] - percentiles[0]))
            sigma_p = float("{:.5f}".format(percentiles[2] - percentiles[1]))

            self.trace = trace
            self.model = model

            return PeriodResult(
                period=med_p,
                neg_error=sigma_n,
                pos_error=sigma_p,
                method=self.__class__.__name__,
                period_distribution=period_samples,
            )

        return PeriodResult(
            period=float("{:.5f}".format(map_soln["period"])),
            neg_error=0.0,
            pos_error=0.0,
            method=self.__class__.__name__,
        )

    def build_model(
        self, start: Optional[pm.Point] = None
    ) -> Tuple[pm.Model, pm.Point]:
        """Build a stellar variability Gaussian Process Model.
        Optimise starting point by finding maximum a posteriori parameters.

        Args:
            start (pymc3 Point, optional): Starting point for initial solution. Defaults to None (model.test_point).

        Returns:
            Tuple[pm.Model, pm.Point]: Tuple containing the model and an optimised starting point.
        """

        with pm.Model() as model:
            mean = pm.Normal("mean", mu=0.0, sigma=10.0)

            # White noise jitter term
            log_jitter = pm.Normal(
                "log_jitter",
                mu=np.log(np.nanmean(self.flux_errors_ppt[self.mask])),
                sigma=np.subtract(
                    *np.percentile(self.flux_errors_ppt[self.mask], [90, 10])
                ),
            )

            # SHOTerm kernel parameters for non-periodic variability (defaults adopted from exoplanet examples)
            sigma = pm.InverseGamma(
                "sigma", **pmx.estimate_inverse_gamma_parameters(1.0, 5.0)
            )
            rho = pm.InverseGamma(
                "rho", **pmx.estimate_inverse_gamma_parameters(0.5, 2.0)
            )

            # RotationTerm kernel parameters
            sigma_rot = pm.InverseGamma(
                "sigma_rot", **pmx.estimate_inverse_gamma_parameters(1.0, 5.0)
            )
            log_period = pm.Normal(
                "log_period", mu=np.log(self.gp_seed_period), sigma=2.0
            )
            period = pm.Deterministic("period", tt.exp(log_period))
            log_Q0 = pm.HalfNormal("log_Q0", sigma=2.0)
            log_dQ = pm.Normal("log_dQ", mu=0.0, sigma=2.0)
            f = pm.Uniform("f", lower=0.1, upper=1.0)

            # Define GP (Rotation + SHO) model
            kernel = terms.SHOTerm(sigma=sigma, rho=rho, Q=1 / 3.0)
            kernel += terms.RotationTerm(
                sigma=sigma_rot,
                period=period,
                Q0=tt.exp(log_Q0),
                dQ=tt.exp(log_dQ),
                f=f,
            )
            gp = GaussianProcess(
                kernel,
                t=self.timeseries[self.mask],
                diag=tt.add(
                    self.flux_errors_ppt[self.mask] ** 2, tt.exp(2 * log_jitter)
                ),
                mean=mean,
                quiet=True,
            )

            # Compute the GP likelihood and add it into the PyMC3 model as a "potential"
            gp.marginal("gp", observed=self.flux_ppt[self.mask])

            # Compute the mean model prediction for plotting purposes
            pm.Deterministic("pred", gp.predict(self.flux_ppt[self.mask]))

            # Optimize to find the maximum a posteriori parameters
            if start is None:
                start = model.test_point
            map_soln = pmx.optimize(start=start)

            self.gp = gp
            self.model = model
            self.solution = map_soln

            return model, map_soln

    def plot(self, ax, period: PeriodResult, colour: Optional[str] = "orange") -> None:
        """Given a figure and an axis plot the interesting output of the object.

        Args:
            ax ([type]): Matplotlib axis
            period (PeriodResult): Outputted period to plot around
        """

        if period.period_distribution is not None:

            ax.hist(
                period.period_distribution,
                histtype="step",
                bins=21,
                # bins=np.linspace(xmin - period.neg_error, xmax + period.pos_error),
                color=colour,
            )

        else:
            ax.axvline(period.period, color=colour)
            ax.axvspan(
                period.period - period.neg_error,
                period.period + period.pos_error,
                color=colour,
                alpha=0.2,
            )

        ax2 = ax.twinx()
        ax2.set_ylim([0, 1])
        ax2.get_yaxis().set_visible(False)

        ax2.errorbar(
            period.period,
            0.5,
            xerr=[[period.neg_error], [period.pos_error]],
            ms=10,
            marker="s",
            c="k",
            capsize=10,
        )

        ax.set_xlabel(f"Period / {self.time_units}")
        ax.set_yticks([])
        ax.set_ylabel("Period Posterior")
        ax.set_title("Gaussian Process Model")

    def _generate_plotting_predictions(
        self, timeseries: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        mu, var = pmx.eval_in_model(
            self.gp.predict(self.flux_ppt[self.mask], t=timeseries, return_var=True),
            point=self.solution,
            model=self.model,
        )
        mu += self.solution["mean"]
        std = np.sqrt(var)

        # convert data from ppt back into rel flux
        if self.flux_units == "relative flux units":
            if self.median_flux != 0:
                mu_rel = ppt_to_rel_flux(
                    mu,
                    normalise=True,
                    normalisation_value=self.median_flux,
                    center_around=1.0,
                )
                std_rel = ppt_to_rel_flux(
                    std,
                    normalise=True,
                    normalisation_value=self.median_flux,
                    center_around=0.0,
                )
            else:
                mu_rel = ppt_to_rel_flux(mu)
                std_rel = ppt_to_rel_flux(std)
        elif self.flux_units == "ppt":
            mu_rel = mu
            std_rel = std
        else:
            print(
                "Warning: Not converting units as cannot handle anything other than relative flux units"
            )
            mu_rel = mu
            std_rel = std

        return mu_rel, std_rel

    def plot_gp_predictions(self, ax: Axes, colour: Optional[str] = "orange") -> Axes:
        """Plot GP model predictions.

        Args:
            ax (Axes):  Matplotlib axis
        """
        time_extent = self.timeseries.max() - self.timeseries.min()
        model_timeseries = np.linspace(
            self.timeseries.min() - (time_extent * 0.05),
            self.timeseries.max() + (time_extent * 0.05),
            2000,
        )

        mu, std = self._generate_plotting_predictions(model_timeseries)

        line = ax.fill_between(
            model_timeseries, mu + std, mu - std, color=colour, alpha=0.3, zorder=1
        )
        line.set_edgecolor("none")

        ax.plot(model_timeseries, mu, color=colour, zorder=10)

    def plot_gp_residuals(
        self,
        ax: Axes,
        colour: Optional[str] = "orange",
        max_number_of_points: float = 2000,
    ) -> Axes:
        """Plot GP model predictions.

        Args:
            ax (Axes):  Matplotlib axis
        """

        if len(self.timeseries) > max_number_of_points:
            # downsample for plotting
            n_times_over = len(self.timeseries[self.mask]) / max_number_of_points
            plotting_timeseries = self.timeseries[self.mask][:: ceil(n_times_over)]
            plotting_flux = self.flux[self.mask][:: ceil(n_times_over)]
        else:
            plotting_timeseries = self.timeseries[self.mask]
            plotting_flux = self.flux[self.mask]

        mu, std = self._generate_plotting_predictions(plotting_timeseries)

        residuals = plotting_flux - mu

        ax.fill_between(plotting_timeseries, std, -std, color=colour, alpha=0.2)
        ax.scatter(plotting_timeseries, residuals, color="k", s=1)

        ax.axhline(y=0, lw=1, alpha=0.2, c='gray')
        ax.set_xlabel(f"Time / {self.time_units}")
        ax.set_ylabel("Residuals")

        return ax

    def plot_trace(
        self,
        show: bool = True,
        savefig: bool = False,
        filename: str = "",
        fileext: str = "pdf",
    ):
        """Plot trace using arviZ plot_trace.

        Args:
            show (bool, optional): Show using e.g. interactive backend. Defaults to True.
            savefig (bool, optional): Save figure. Defaults to False.
            filename (str, optional): Filename to save figure. Defaults to "" (saves as '_trace.pdf')
            fileext (str, optional): File extension to save figure. Defaults to "pdf".

        Raises:
            RuntimeError: If no trace, will raise RuntimeError
        """

        if self.trace:
            with self.model:
                trace_ax = az.plot_trace(
                    self.trace, show=show, combined=True, compact=True
                )

            if savefig:
                plt.savefig(splitext(filename)[0] + "_trace." + fileext)
        else:
            raise RuntimeError("Cannot plot trace as no trace generated")

    def plot_distributions(
        self,
        show: bool = True,
        savefig: bool = False,
        filename: str = "",
        fileext: str = "pdf",
    ):
        """Plot outputted parameter distributions corner plot.

        Args:
            show (bool, optional): Show using e.g. interactive backend. Defaults to True.
            savefig (bool, optional): Save figure. Defaults to False.
            filename (str, optional): Filename to save figure. Defaults to "" (saves as '_distributions.pdf')
            fileext (str, optional): File extension to save figure. Defaults to "pdf".

        Raises:
            RuntimeError: If no solution, will raise RuntimeError
        """
        if self.trace and self.solution:

            names = [
                k for k in self.solution.keys() if (k[-2:] != "__") and (k != "pred")
            ]

            fig = corner(
                self.trace,
                names=names,
                quantiles=[0.15865, 0.5, 0.84135],
                show_titles=True,
                max_n_ticks=4,
                var_names=names,
            )
            if show:
                plt.show()
            if savefig:
                fig.savefig(splitext(filename)[0] + "_distributions." + fileext)
        else:
            raise RuntimeError("Cannot plot distributions as no trace found.")
