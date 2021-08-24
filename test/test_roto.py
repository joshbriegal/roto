import pytest

from src.methods.periodfinder import PeriodFinder
from src.roto import RoTo

inverse_methods_dictionary = {method: name for name, method in RoTo.METHODS.items()}


@pytest.mark.parametrize(
    "method_parameters",
    [
        None,
        {"lombscargle": {"normalization": True, "fit_mean": False}},
        {"gacf": {}},
        {"fft": {}},
        {"gp": {}},
    ],
)
def test_create_roto(method_parameters, timeseries, flux, flux_errors):
    roto = RoTo(timeseries, flux, flux_errors, method_parameters)

    for method in roto.methods:
        assert isinstance(method, PeriodFinder)
        if method_parameters:
            assert inverse_methods_dictionary[method.__class__] in method_parameters
