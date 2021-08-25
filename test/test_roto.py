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
    ],
)
def test_create_roto(method_parameters, timeseries, flux, flux_errors):
    roto = RoTo(timeseries, flux, flux_errors, method_parameters)

    for name, method in roto.methods.items():
        assert isinstance(method, PeriodFinder)
        assert name in roto.METHODS.keys()
        if method_parameters:
            assert inverse_methods_dictionary[method.__class__] in method_parameters


def test_create_roto_only_gp(timeseries, flux, flux_errors):
    method_parameters = {"gp": {}}
    roto = RoTo(timeseries, flux, flux_errors, method_parameters)

    for name, method in roto.methods.items():
        assert isinstance(method, PeriodFinder)
        assert name in roto.METHODS.keys()

    assert list(roto.methods.keys()) == ["lombscargle", "gp"]
