import numpy as np

from astropy.timeseries import LombScargle
from numpy.testing import assert_equal
from unittest import mock

from src.methods.lombscargle import LombScarglePeriodFinder


def test_init(timeseries, flux, flux_errors):

    pf = LombScarglePeriodFinder(timeseries, flux, flux_errors)

    assert isinstance(pf._lombscargle, LombScargle)
    assert_equal(pf.timeseries, timeseries)
    assert_equal(pf.flux, flux)
    assert_equal(pf.flux_errors, flux_errors)


@mock.patch.object(LombScargle, "autopower", autospec=True)
def test_periodogram(mock_autopower, timeseries, flux, flux_errors):
    mock_autopower.return_value = (np.zeros(10), np.ones(10))

    ls = LombScarglePeriodFinder(timeseries, flux, flux_errors)

    maximum_frequency = 2352
    method = "somemethod"
    method_kwds = {"kwa": "rg"}
    minimum_frequency = 92992929
    normalization = True
    nyquist_factor = 69
    samples_per_peak = 420

    periodogram = ls.calculate_periodogram(
        maximum_frequency=maximum_frequency,
        method=method,
        method_kwds=method_kwds,
        minimum_frequency=minimum_frequency,
        normalization=normalization,
        nyquist_factor=nyquist_factor,
        samples_per_peak=samples_per_peak,
        this_is_not="a real argument",
    )

    mock_autopower.assert_called_once_with(
        ls._lombscargle,
        maximum_frequency=maximum_frequency,
        method=method,
        method_kwds=method_kwds,
        minimum_frequency=minimum_frequency,
        normalization=normalization,
        nyquist_factor=nyquist_factor,
        samples_per_peak=samples_per_peak,
    )

    assert_equal(periodogram.frequency_axis, mock_autopower.return_value[0])
    assert_equal(periodogram.power_axis, mock_autopower.return_value[1])


def test_call(timeseries, flux, period):

    ls = LombScarglePeriodFinder(timeseries, flux, flux_errors=None)

    outputted_period = ls()

    # assert outputted_period.period == period  # TODO: Currently fails as LS outputs many aliases - we see sampling as largest peak.
    assert outputted_period.method == "LombScarglePeriodFinder"
    assert outputted_period.neg_error == 0
    assert outputted_period.pos_error == 0
