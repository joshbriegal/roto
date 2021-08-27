import pandas as pd

from roto import RoTo

if __name__ == "__main__":

    # import data
    df = pd.read_csv(
        "example/KIC_5110407_LC_with_gaps.dat",
        sep=r"\s+",
        header=None,
        engine="python",
        thousands=",",
        comment="#",
        names=["timeseries", "flux", "flux_errors"],
    )

    print("Imported the following light curve: ")
    print(df)

    roto = RoTo(
        df.timeseries,
        df.flux,
        df.flux_errors,
        # {"lombscargle": {}, "gacf": {}},
        name="KIC_5110407_LC_with_gaps",
    )
    roto(gacf_method="peaks", do_mcmc=True, remove_outliers=True)
    print(roto)
    roto.plot(savefig=True, show=False)
