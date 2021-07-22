import numpy as np
import pytest

from astropy.timeseries import LombScargle
from numpy.testing import assert_equal
from unittest import mock

from src.methods.lombscargle import LombScarglePeriodFinder


@pytest.fixture
def timeseries():
    return np.linspace(0, 1, 1000)


@pytest.fixture
def flux(timeseries):
    return np.sin(timeseries)


@pytest.fixture
def flux_errors(timeseries):
    return np.random.rand(*timeseries.shape)


def testLSPeriodFinderInit(timeseries, flux, flux_errors):

    pf = LombScarglePeriodFinder(timeseries, flux, flux_errors)

    assert isinstance(pf._lombscargle, LombScargle)
    assert_equal(pf.timeseries, timeseries)
    assert_equal(pf.flux, flux)
    assert_equal(pf.flux_errors, flux_errors)


@mock.patch.object(LombScargle, "autopower", autospec=True)
def testLSPeriodFinderFind(mock_autopower, timeseries, flux, flux_errors):
    mock_autopower.return_value = (np.zeros(10), np.ones(10))

    pf = LombScarglePeriodFinder(timeseries, flux, flux_errors)

    maximum_frequency = 2352
    method = "somemethod"
    method_kwds = {"kwa": "rg"}
    minimum_frequency = 92992929
    normalization = True
    nyquist_factor = 69
    samples_per_peak = 420

    freq, power = pf(
        maximum_frequency=maximum_frequency,
        method=method,
        method_kwds=method_kwds,
        minimum_frequency=minimum_frequency,
        normalization=normalization,
        nyquist_factor=nyquist_factor,
        samples_per_peak=samples_per_peak,
    )

    mock_autopower.assert_called_once_with(
        pf._lombscargle,
        maximum_frequency=maximum_frequency,
        method=method,
        method_kwds=method_kwds,
        minimum_frequency=minimum_frequency,
        normalization=normalization,
        nyquist_factor=nyquist_factor,
        samples_per_peak=samples_per_peak
        )

    assert_equal(freq, mock_autopower.return_value[0])
    assert_equal(power, mock_autopower.return_value[1])
