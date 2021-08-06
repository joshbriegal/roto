from unittest import mock

import numpy as np
from gacf import GACF
from numpy.testing import assert_equal

from src.methods.fft import FFTPeriodFinder
from src.methods.gacf import GACFPeriodFinder
from src.methods.periodfinder import PeriodResult


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


@mock.patch("src.methods.gacf.median_abs_deviation", autospec=True)
@mock.patch("src.methods.gacf.np.median", autospec=True)
def test_find_acf_peaks_sine(mock_median, mock_mad):

    period = 1.23

    mock_mad.return_value = 0.1
    mock_median.return_value = period

    timeseries = np.linspace(0, 10, 100)
    flux = np.sin(np.pi * timeseries * 2 / period)

    pf = GACFPeriodFinder(timeseries, flux)
    lag_timeseries, correlations = pf.calculate_autocorrelation(min_lag=0)

    outputted_period = pf.find_acf_peaks(lag_timeseries, correlations)

    valid_peaks = mock_mad.call_args[0][0]

    assert mock_mad.call_count == 1
    assert mock_median.call_count == 1

    assert outputted_period.period == period
    assert outputted_period.method == "GACFPeriodFinder"
    assert outputted_period.neg_error == outputted_period.pos_error
    assert outputted_period.pos_error == 0.1 * 1.483 / np.sqrt(len(valid_peaks) - 1)


@mock.patch("src.methods.gacf.median_abs_deviation", autospec=True)
@mock.patch("src.methods.gacf.np.median", autospec=True)
def test_find_acf_peaks_sin_short_period(mock_median, mock_mad):

    period = 0.04

    mock_mad.return_value = 0.1
    mock_median.return_value = period

    timeseries = np.linspace(0, 10, 100)
    flux = np.sin(np.pi * timeseries * 2 / period)

    pf = GACFPeriodFinder(timeseries, flux)
    lag_timeseries, correlations = pf.calculate_autocorrelation(min_lag=0)

    outputted_period = pf.find_acf_peaks(lag_timeseries, correlations)

    valid_peaks = mock_mad.call_args[0][0]

    assert len(valid_peaks) == 10

    assert mock_mad.call_count == 1
    assert mock_median.call_count == 1

    assert outputted_period.period == period
    assert outputted_period.method == "GACFPeriodFinder"
    assert outputted_period.neg_error == outputted_period.pos_error
    assert outputted_period.pos_error == 0.1 * 1.483 / np.sqrt(len(valid_peaks) - 1)


@mock.patch("src.methods.gacf.median_abs_deviation", autospec=True)
@mock.patch("src.methods.gacf.np.median", autospec=True)
def test_find_acf_peaks_sine_long_period(mock_median, mock_mad):

    period = 7.5

    timeseries = np.linspace(0, 10, 100)
    flux = np.sin(np.pi * timeseries * 2 / period)

    pf = GACFPeriodFinder(timeseries, flux)
    lag_timeseries, correlations = pf.calculate_autocorrelation(min_lag=0)

    outputted_period = pf.find_acf_peaks(lag_timeseries, correlations)

    mock_mad.assert_not_called()
    mock_median.assert_not_called()

    # values not exact, but check it roughly works.
    assert period * 0.8 <= outputted_period.period <= period * 1.2
    assert outputted_period.method == "GACFPeriodFinder"
    assert outputted_period.neg_error == outputted_period.pos_error
    assert 2.4 <= outputted_period.pos_error <= 2.5
