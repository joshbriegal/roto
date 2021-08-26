from typing import Optional, Tuple

import aesara_theano_fallback.tensor as tt
import numpy as np
import pymc3 as pm
import pymc3_ext as pmx
from celerite2.theano import GaussianProcess, terms

from src.methods.periodfinder import PeriodFinder, PeriodResult


class GPPeriodFinder(PeriodFinder):
    """Gaussian Process (GP) regression method to find periods.
    Conforms to PeriodFinder interface.
    """

    def __init__(
        self,
        timeseries: np.ndarray,
        flux: np.ndarray,
        flux_errors: Optional[np.ndarray] = None,
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
        super().__init__(timeseries, flux, flux_errors)

        # convert data into ppt format
        fmed = np.nanmedian(self.flux)
        self.flux_ppt = (self.flux / fmed - 1) * 1.0e3
        self.flux_errors_ppt = (self.flux_errors / fmed) * 1.0e3

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
            mask = np.abs(residuals) < rms_sigma * rms
            model, map_soln = self.build_model(mask=mask, start=map_soln)

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
                    return_inferencedata=True,
                    random_seed=[13089739, 13089740],
                )

            # estimate period and uncertainty
            period_samples = np.asarray(trace.posterior["period"]).flatten()
            percentiles = np.percentile(period_samples, [15.87, 50.0, 84.14])
            med_p = float("{:.5f}".format(percentiles[1]))
            sigma_n = float("{:.5f}".format(percentiles[1] - percentiles[0]))
            sigma_p = float("{:.5f}".format(percentiles[2] - percentiles[1]))

            self.trace = trace

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
        self, mask: Optional[np.ndarray] = None, start: Optional[pm.Point] = None
    ) -> Tuple[pm.Model, pm.Point]:
        """Build a stellar variability Gaussian Process Model.
        Optimise starting point by finding maximum a posteriori parameters.

        Args:
            mask (np.ndarray, optional): Masking array to apply to timeseries. Defaults to None (unmasked).
            start (pymc3 Point, optional): Starting point for initial solution. Defaults to None (model.test_point).

        Returns:
            Tuple[pm.Model, pm.Point]: Tuple containing the model and an optimised starting point.
        """
        if mask is None:
            mask = np.ones(len(self.timeseries), dtype=bool)

        with pm.Model() as model:
            mean = pm.Normal("mean", mu=0.0, sigma=10.0)

            # White noise jitter term
            log_jitter = pm.Normal(
                "log_jitter",
                mu=np.log(np.nanmean(self.flux_errors_ppt[mask])),
                sigma=2.0,
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
                t=self.timeseries[mask],
                diag=tt.add(self.flux_ppt[mask] ** 2, tt.exp(2 * log_jitter)),
                mean=mean,
                quiet=True,
            )

            # Compute the GP likelihood and add it into the PyMC3 model as a "potential"
            gp.marginal("gp", observed=self.flux_ppt[mask])

            # Compute the mean model prediction for plotting purposes
            pm.Deterministic("pred", gp.predict(self.flux_ppt[mask]))

            # Optimize to find the maximum a posteriori parameters
            if start is None:
                start = model.test_point
            map_soln = pmx.optimize(start=start)

            self.gp = gp
            self.model = model
            self.solution = map_soln

            return model, map_soln

    def plot(self, ax, period: PeriodResult) -> None:
        """Given a figure and an axis plot the interesting output of the object.

        Args:
            ax ([type]): Matplotlib axis
            period (PeriodResult): Outputted period to plot around
        """
        nperiods = 10
        xmin = 0
        xmax = 1

        if period.period_distribution is not None:
            xmin = period.period - 5 * period.neg_error
            xmax = min(
                period.period + 5 * period.pos_error, max(period.period_distribution)
            )

            bin_size = (period.neg_error + period.pos_error) / 5

            ax.hist(
                period.period_distribution,
                histtype='step',
                bins=np.linspace(
                    xmin - period.neg_error, xmax + period.pos_error
                ),
            )

        ax.set_xlim([xmin, xmax])
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

        ax.set_xlabel("Period")
        ax.set_yticks([])
        ax.set_ylabel("Period Posterior")
        ax.set_title("Gaussian Process Model")

    def plot_gp_predictions(self, ax) -> None:
        """Plot GP model predictions.

        Args:
            ax ([type]):  Matplotlib axis
        """
        model_timeseries = np.linspace(self.timeseries.min()-5, self.timeseries.max()+5, 2000)
        mu, var = self.gp.predict(self.flux, t=model_timeseries, return_var=True)
        mu += self.solution["mean"]
        std = np.sqrt(var)

        line = ax.fill_between(model_timeseries, mu+std, mu-std, color="orange", alpha=0.3, zorder=1)
        line.set_edgecolor("none")

        ax.plot(model_timeseries, mu, color="orange", zorder=2)



