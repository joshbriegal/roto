import logging

import lightkurve as lk
import numpy as np

from roto import RoTo

logging.basicConfig(
    level=logging.ERROR,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logging.getLogger("roto").setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)


if __name__ == "__main__":

    name = "TIC 10863087"

    lcf = lk.search_lightcurve(name, mission="TESS", author="SPOC").download_all(
        quality_bitmask="hardest", flux_column="pdcsap_flux"
    )
    lc = lcf.stitch().remove_nans().remove_outliers()
    lc = lc[:5000]
    _, mask = lc.flatten().remove_outliers(sigma=3.0, return_mask=True)
    lc = lc[~mask]

    x = np.ascontiguousarray(lc.time.value, dtype=np.float64)
    y = np.ascontiguousarray(lc.flux, dtype=np.float64)
    yerr = np.ascontiguousarray(lc.flux_err, dtype=np.float64)
    mu = np.mean(y)
    y /= mu
    yerr /= mu

    roto = RoTo(
        x,
        y,
        yerr,
        name=name,
        # methods_parameters={
        #     "lombscargle": {},
        #     "fft": {},
        #     "gacf": {},
        #     "gp": {"gp_seed_period": 3.853},
        # },
    )
    roto(
        gacf_method="peaks",
        do_mcmc=False,
        remove_outliers=True,
        draws=2000,
        cores=4,
        chains=4,
    )
    print(roto)
    roto.plot(plot_gp=True, savefig=True, show=False)

    # roto.plot_gp_diagnostics(show=False, savefig=True, fileext="png")
