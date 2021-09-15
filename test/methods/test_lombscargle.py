from unittest import mock
import pytest

import numpy as np
from astropy.timeseries import LombScargle
from numpy.testing import assert_equal

from roto.methods.lombscargle import LombScarglePeriodFinder
from roto.methods.periodfinder import PeriodFinder, PeriodResult


@pytest.fixture
def period_result():
    return PeriodResult(1, 1, 1, "mock")

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


def test_call_non_sliding(timeseries, flux, period):

    ls = LombScarglePeriodFinder(timeseries, flux, flux_errors=None, sliding=False)

    outputted_period = ls()

    assert period * 0.8 <= outputted_period.period <= period * 1.2
    assert outputted_period.method == "LombScarglePeriodFinder"
    assert outputted_period.neg_error == 0
    assert outputted_period.pos_error == 0


@mock.patch.object(LombScarglePeriodFinder, "_sliding_ls_periodogram", autospec=True)
@mock.patch.object(PeriodFinder, "__call__", autospec=True)
def test_call_sliding(mock_call, mock_sliding, timeseries, flux, period_result):

    mock_call.return_value = period_result

    ls = LombScarglePeriodFinder(timeseries, flux, flux_errors=None, sliding=True)
    
    ls(n_periods=100, sliding_aggregation="agg_method")

    mock_call.assert_called_once_with(ls, n_periods=100, sliding_aggregation="agg_method")
    mock_sliding.assert_called_once_with(ls, period_result, n_periods=100, sliding_aggregation="agg_method")


@mock.patch("roto.methods.lombscargle.np.nanmean", autospec=True)
@mock.patch("roto.methods.lombscargle.np.nanstd", autospec=True)
def test_sliding_ls_periodogram_method_mean(mock_std, mock_mean, timeseries, flux, period_result):
    mock_mean.return_value = 420
    mock_std.return_value = 69

    ls = LombScarglePeriodFinder(timeseries, flux, flux_errors=None, sliding=True)

    mock_ls = mock.MagicMock()
    mock_ls.return_value = period_result

    with mock.patch("roto.methods.lombscargle.LombScarglePeriodFinder", autospec=True, return_value=mock_ls) as mock_ls_create:

        pr = ls._sliding_ls_periodogram(period_result, sliding_aggregation="mean", some="kwarg")

        assert pr.period == 420
        assert pr.neg_error == pr.pos_error == 69
        assert pr.method == "LombScarglePeriodFinder"

        mock_mean.assert_called_once_with([1] * 96)
        mock_std.assert_called_once_with([1] * 96)

        assert mock_ls_create.call_count == 96

        first_mock_call = mock_ls_create.mock_calls[0]
        assert_equal(timeseries[0:50], first_mock_call[1][0])
        assert_equal(flux[0:50], first_mock_call[1][1])
        assert first_mock_call[1][2] is None

        assert first_mock_call[2] == {
            "fit_mean":True, "center_data":True, "nterms":1, "normalization":'standard', "sliding":False
        }

        assert mock_ls.call_count == 96
        mock_ls.assert_called_with(some="kwarg")


@mock.patch("roto.methods.lombscargle.np.percentile", autospec=True)
def test_sliding_ls_periodogram_method_median(mock_perc, timeseries, flux, period_result):
    mock_perc.return_value = [385.5, 420, 454.5]

    ls = LombScarglePeriodFinder(timeseries, flux, flux_errors=None, sliding=True)

    mock_ls = mock.MagicMock()
    mock_ls.return_value = period_result

    with mock.patch("roto.methods.lombscargle.LombScarglePeriodFinder", autospec=True, return_value=mock_ls) as mock_ls_create:

        pr = ls._sliding_ls_periodogram(period_result, sliding_aggregation="median", some="kwarg")

        assert pr.period == 420
        assert pr.neg_error == pr.pos_error == 69
        assert pr.method == "LombScarglePeriodFinder"

        mock_perc.assert_called_once_with([1] * 96, [10, 50, 90])

        assert mock_ls_create.call_count == 96

        first_mock_call = mock_ls_create.mock_calls[0]
        assert_equal(timeseries[0:50], first_mock_call[1][0])
        assert_equal(flux[0:50], first_mock_call[1][1])
        assert first_mock_call[1][2] is None

        assert first_mock_call[2] == {
            "fit_mean":True, "center_data":True, "nterms":1, "normalization":'standard', "sliding":False
        }

        assert mock_ls.call_count == 96
        mock_ls.assert_called_with(some="kwarg")

@pytest.mark.parametrize(
    "period",
    [100, 50, 40, 33.3333, 20, 34/5],
)
@mock.patch("roto.methods.lombscargle.np.percentile", autospec=True)
def test_sliding_ls_periodogram_period_too_long(mock_perc, timeseries, flux, period_result):
    period_result.period = 50

    ls = LombScarglePeriodFinder(timeseries, flux, flux_errors=None, sliding=True)

    mock_ls = mock.MagicMock()
    mock_ls.return_value = period_result

    with mock.patch("roto.methods.lombscargle.LombScarglePeriodFinder", autospec=True, return_value=mock_ls) as mock_ls_create:

        pr = ls._sliding_ls_periodogram(period_result)

        assert pr == period_result

        mock_perc.assert_not_called()
        mock_ls.assert_not_called()
        mock_ls_create.assert_not_called()