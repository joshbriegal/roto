from roto import RoTo

import pandas as pd

if __name__ == "__main__":

    # import data
    df = pd.read_csv(
        "example/KIC_5110407_LC_with_gaps.dat",
        sep="\s+",
        header=None,
        engine="python",
        thousands=",",
        comment="#",
        names=["timeseries", "flux", "flux_errors"],
    )

    print("Imported the following light curve: ")
    print(df)

    roto = RoTo(df.timeseries, df.flux, df.flux_errors)
    roto()
    print(roto)
