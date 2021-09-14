import logging

import pandas as pd

from roto import RoTo

logging.basicConfig(
    level=logging.ERROR,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logging.getLogger("roto").setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)

if __name__ == "__main__":

    # import data
    df = pd.read_csv(
        "example/NGTS-Object-NG1827+0636.1439611.csv",
        engine="python",
        comment="#",
        # names=["timeseries", "flux", "flux_errors"],
    )

    print("Imported the following light curve: ")
    print(df)

    roto = RoTo(
        df.timeseries,
        df.flux,
        df.flux_errors,
        # {"lombscargle": {}, "gacf": {}},
        name="NGTS-Object-NG1827+0636.1439611",
    )
    roto(
        gacf_method="fft",
        do_mcmc=False,
        remove_outliers=True,
        chains=8,
        cores=8,
        draws=500,
    )
    print(roto)
    roto.plot(savefig=True, show=False)
    # roto.plot_gp_diagnostics(show=False, savefig=True, fileext='png')
