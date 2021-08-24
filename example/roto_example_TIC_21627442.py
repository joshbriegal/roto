from astropy.io import ascii

from src.roto import RoTo

if __name__ == "__main__":

    # import tab
    tab = ascii.read(
        "example/TIC_21627442_2min_LC_sector_6.dat",
    )

    # print("Imported the following light curve: ")
    # print(tab)

    mask = tab["quality"].data == 0
    timeseries = tab["time"].data[mask]
    flux = tab["flux"].data[mask]
    flux_errors = tab["flux_err"].data[mask]

    roto = RoTo(
        timeseries,
        flux,
        flux_errors,
        methods_parameters={
            "lombscargle": {},
            "fft": {},
            "gacf": {},
            "gp": {"gp_seed_period": 3.853},
        },
    )
    roto()
    print(roto)
