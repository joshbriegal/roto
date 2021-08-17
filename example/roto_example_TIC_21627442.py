import pandas as pd

from roto import RoTo

if __name__ == "__main__":

    # import data
    df = pd.read_csv(
        "example/TIC_21627442_2min_LC_sector_6.dat",
        sep=r"\s+",
        header=None,
        engine="python",
        thousands=",",
        comment="#",
        names=["timeseries", "flux", "flux_errors"],
    )

    print("Imported the following light curve: ")
    print(df)

    mask = (df.quality==0)

    roto = RoTo(df.timeseries[mask], df.flux[mask], df.flux_errors[mask])
    roto()
    print(roto)
