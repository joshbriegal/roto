from src.methods.periodfinder import PeriodFinder
import numpy as np

from numpy.testing import assert_equal, assert_almost_equal
from unittest import mock

from src.methods.fft import FFTPeriodFinder


def test_init(timeseries, flux, flux_errors):

    pf = FFTPeriodFinder(timeseries, flux, flux_errors)

    assert_equal(pf.timeseries, timeseries)
    assert_equal(pf.flux, flux)
    assert_equal(pf.flux_errors, flux_errors)


@mock.patch("src.methods.fft.fft.rfft", autospec=True)
@mock.patch("src.methods.fft.fft.rfftfreq", autospec=True)
def test_periodogram(mock_rfftfreq, mock_rfft, timeseries, flux, flux_errors):

    mock_rfft.return_value = np.ones(10)
    mock_rfftfreq.return_value = np.zeros(10)

    ls = FFTPeriodFinder(timeseries, flux, flux_errors)

    n_times_data = 69
    len_ft = 420
    power_of_two = False
    pad_both_sides = True

    periodogram = ls.calculate_periodogram(
        n_times_data=n_times_data,
        len_ft=len_ft,
        power_of_two=power_of_two,
        pad_both_sides=pad_both_sides,
        this_is_not="a real argument",
    )

    assert mock_rfft.call_count == 1
    assert_equal(mock_rfft.call_args[0][0], flux)
    assert mock_rfft.call_args[1]["n"] == len_ft

    mock_rfftfreq.assert_called_once_with(
        len_ft, timeseries[1] - timeseries[0]
    )

    assert_equal(periodogram.frequency_axis, mock_rfftfreq.return_value)
    assert_equal(periodogram.power_axis, mock_rfft.return_value)


def test_call(timeseries, flux, period):

    fft = FFTPeriodFinder(timeseries, flux, flux_errors=None)

    outputted_period = fft()

    assert_almost_equal(outputted_period.period, period, decimal=1)  # TODO: Currently fails as LS outputs many aliases - we see sampling as largest peak.
    assert outputted_period.method == "FFTPeriodFinder"
    assert outputted_period.neg_error == 0
    assert outputted_period.pos_error == 0
