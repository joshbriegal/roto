from src.methods.periodfinder import PeriodResult
import numpy as np

from gacf import GACF
from numpy.testing import assert_equal
from unittest import mock

from src.methods.gacf import GACFPeriodFinder
from src.methods.fft import FFTPeriodFinder


def test_init(timeseries, flux, flux_errors):

    pf = GACFPeriodFinder(timeseries, flux, flux_errors)

    assert isinstance(pf._gacf, GACF)
    assert_equal(pf.timeseries, timeseries)
    assert_equal(pf.flux, flux)
    assert_equal(pf.flux_errors, flux_errors)


def test_periodogram(timeseries, flux, flux_errors):
    pf = GACFPeriodFinder(timeseries, flux, flux_errors)

    assert pf.calculate_periodogram() is None


@mock.patch.object(GACF, "autocorrelation", autospec=True)
def test_autocorrelation(mock_gacf, timeseries, flux, flux_errors):
    mock_gacf.return_value = (np.zeros(10), np.ones(10))

    pf = GACFPeriodFinder(timeseries, flux, flux_errors)

    min_lag = 69
    max_lag = 420
    lag_resolution = 0.9
    selection_function = "someselectionfunction"
    weight_function = "someweightfunction"
    alpha = None

    lag_timeseries, correlations = pf.calculate_autocorrelation(
        min_lag=min_lag,
        max_lag=max_lag,
        lag_resolution=lag_resolution,
        selection_function=selection_function,
        weight_function=weight_function,
        alpha=alpha,
        this_is="not a real argument",
    )

    mock_gacf.assert_called_once_with(
        pf._gacf,
        min_lag=min_lag,
        max_lag=max_lag,
        lag_resolution=lag_resolution,
        selection_function=selection_function,
        weight_function=weight_function,
        alpha=alpha,
    )

    assert_equal(lag_timeseries, mock_gacf.return_value[0])
    assert_equal(correlations, mock_gacf.return_value[1])


@mock.patch("src.methods.gacf.FFTPeriodFinder", autospec=True)
def test_call(mock_fft, timeseries, flux, period):

    mock_fft_object = mock.Mock(return_value=PeriodResult(period))
    mock_fft.return_value = mock_fft_object

    kwargs = {"some": "random", "inputs": [8]}

    pf = GACFPeriodFinder(timeseries, flux, flux_errors=None)

    outputted_period = pf(**kwargs)

    mock_fft_object.assert_called_once_with(**kwargs)

    assert outputted_period.period == period
    assert outputted_period.method == "GACFPeriodFinder"
    assert outputted_period.neg_error == 0
    assert outputted_period.pos_error == 0
