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
        name="TIC_21627442_2min_LC_sector_6.dat",
        # methods_parameters={
        #     "lombscargle": {},
        #     "fft": {},
        #     "gacf": {},
        #     "gp": {"gp_seed_period": 3.853},
        # },
    )
    roto(gacf_method="peaks", do_mcmc=False, remove_outliers=False)
    print(roto)
    roto.plot(plot_gp=True, savefig=True, show=False)
    # roto.plot_gp_diagnostics(show=False, savefig=True)
