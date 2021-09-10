from unittest import mock

import numpy as np
from numpy.testing import assert_equal

from roto.methods.gaussianprocess import GPPeriodFinder
from roto.methods.periodfinder import PeriodResult


@mock.patch("roto.methods.gaussianprocess.np.nanmedian", return_value=3.0)
def test_init(mock_median, timeseries, flux, flux_errors):

    pf = GPPeriodFinder(timeseries, flux, flux_errors)

    assert_equal(pf.timeseries, timeseries)
    assert_equal(pf.flux, flux)
    assert_equal(pf.flux_errors, flux_errors)

    mock_median.assert_called_once_with(pf.flux)

    assert_equal(pf.flux_ppt, (pf.flux / 3.0 - 1) * 1.0e3)
    assert_equal(pf.flux_errors_ppt, (pf.flux_errors / 3.0) * 1.0e3)

    assert pf.trace is None
    assert pf.model is None
    assert pf.solution is None


def test_periodogram(timeseries, flux, flux_errors):
    pf = GPPeriodFinder(timeseries, flux, flux_errors)

    assert pf.calculate_periodogram() is None


@mock.patch.object(GPPeriodFinder, "calcuate_gp_period")
def test_call(mock_gp_return, timeseries, flux, flux_errors):

    kwargs = {"some": "random", "Kwargs": True}
    mock_gp_return.return_value = PeriodResult(420, 69, 3.14, "GPPeriodFinder")

    pf = GPPeriodFinder(timeseries, flux, flux_errors)

    period_result = pf(**kwargs)

    mock_gp_return.assert_called_once_with(**kwargs)
    assert period_result == mock_gp_return.return_value


def test_build_model():
    # TODO: add test for build model
    pass


@mock.patch.object(GPPeriodFinder, "build_model")
def test_calculate_gp_period_no_mcmc(mock_build_model, timeseries, flux, flux_errors):

    mock_model = mock.Mock()

    mock_map_soln = mock.MagicMock()
    solution_dict = {"period": 69.420}
    mock_map_soln.__getitem__.side_effect = solution_dict.__getitem__

    mock_build_model.return_value = mock_model, mock_map_soln

    pf = GPPeriodFinder(timeseries, flux, flux_errors)

    period_result = pf.calcuate_gp_period()

    mock_build_model.assert_called_once_with()
    assert period_result == PeriodResult(69.420, 0, 0, "GPPeriodFinder")


@mock.patch.object(GPPeriodFinder, "build_model")
def test_calculate_gp_period_no_mcmc_remove_outliers(
    mock_build_model, timeseries, flux, flux_errors
):

    mock_model = mock.Mock()

    mock_map_soln = mock.MagicMock()
    solution_dict = {"period": 69.420, "pred": 0.0}
    mock_map_soln.__getitem__.side_effect = solution_dict.__getitem__

    mock_build_model.return_value = mock_model, mock_map_soln

    pf = GPPeriodFinder(timeseries, flux, flux_errors)

    pf.flux_ppt = np.ones(10)

    period_result = pf.calcuate_gp_period(remove_outliers=True)

    assert mock_build_model.call_count == 2

    assert mock_build_model.call_args_list[0] == ()

    assert mock_build_model.call_args_list[1][1]["start"] == mock_map_soln
    assert_equal(pf.mask, np.ones(10))

    assert period_result == PeriodResult(69.420, 0, 0, "GPPeriodFinder")


@mock.patch.object(GPPeriodFinder, "build_model")
@mock.patch("roto.methods.gaussianprocess.pmx.sample")
@mock.patch("roto.methods.gaussianprocess.np.percentile", return_value=[0, 1, 2])
def test_calculate_gp_period_mcmc(
    mock_percentile, mock_sample, mock_build_model, timeseries, flux, flux_errors
):

    mock_model = mock.MagicMock()

    mock_map_soln = mock.MagicMock()
    solution_dict = {"period": 69.420}
    mock_map_soln.__getitem__.side_effect = solution_dict.__getitem__

    mock_trace = mock.Mock()
    mock_posterior = mock.MagicMock()
    mock_period_samples = {"period": [1, 2, 3, 4]}
    mock_posterior.__getitem__.side_effect = mock_period_samples.__getitem__
    mock_trace.posterior = mock_posterior

    mock_sample.return_value = mock_trace

    mock_build_model.return_value = mock_model, mock_map_soln

    pf = GPPeriodFinder(timeseries, flux, flux_errors)

    period_result = pf.calcuate_gp_period(do_mcmc=True)

    mock_build_model.assert_called_once_with()

    mock_sample.assert_called_once_with(
        tune=500,
        draws=500,
        start=mock_map_soln,
        cores=1,
        chains=2,
        target_accept=0.9,
        return_inferencedata=True,
        discard_tuned_samples=True
    )

    assert mock_percentile.call_count == 1
    assert_equal(mock_percentile.call_args_list[0][0][0], mock_period_samples["period"])
    assert_equal(mock_percentile.call_args_list[0][0][1], [15.87, 50.0, 84.14])

    period_result == PeriodResult(1, 1, 1, "GPPeriodFinder")

    assert pf.trace == mock_trace
