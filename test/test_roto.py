import sys

with open("/tmp/python-sys-path.txt", "w") as outfile:
    outfile.write(str(sys.path))

from unittest import mock

import pytest

from roto.methods.periodfinder import PeriodFinder, PeriodResult
from roto.roto import RoTo

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


@mock.patch("roto.roto.LombScarglePeriodFinder", autospec=True)
@mock.patch("roto.roto.FFTPeriodFinder", autospec=True)
@mock.patch("roto.roto.GACFPeriodFinder", autospec=True)
@mock.patch("roto.roto.GPPeriodFinder", autospec=True)
def test_call(mock_gp, mock_gacf, mock_fft, mock_ls, timeseries, flux, flux_errors):

    mock_gacf_object = mock.Mock(return_value=PeriodResult(1))
    mock_fft_object = mock.Mock(return_value=PeriodResult(420))
    mock_ls_object = mock.Mock(return_value=PeriodResult(69))
    mock_gp_object = mock.Mock(return_value=PeriodResult(1))
    mock_broken_function = mock.MagicMock(side_effect=RuntimeError("this is broken"))

    mock_gacf.return_value = mock_gacf_object
    mock_fft.return_value = mock_fft_object
    mock_ls.return_value = mock_ls_object
    mock_gp.return_value = mock_gp_object
    mock_broken = mock.MagicMock()
    mock_broken.return_value = mock_broken_function

    with mock.patch.dict(
        RoTo.METHODS,
        {
            "lombscargle": mock_ls,
            "fft": mock_fft,
            "gacf": mock_gacf,
            "gp": mock_gp,
            "broken": mock_broken,
        },
    ) as patched_dict:

        roto = RoTo(timeseries, flux, flux_errors)

        kwargs = {"some": "random", "keywords": True, "gp_seed_period": 7}

        roto(**kwargs)
        print(roto)

        mock_gacf_object.assert_called_once_with(**kwargs)
        mock_fft_object.assert_called_once_with(**kwargs)
        mock_ls_object.assert_called_once_with(**kwargs)
        mock_gp_object.assert_called_once_with(**kwargs)
        mock_broken_function.assert_called_once_with(**kwargs)

        # check no extra calls have been made.
        # note this may fail if we allow multiple periods per method.
        assert len(roto.periods) == len(roto.METHODS) - 1


@pytest.mark.parametrize(
    "method, period, error, outputted_method",
    [
        ("mean", 3.0, 0.6324555320336759, "CombinedPeriodResult"),
        ("median", 3.0, 0.6630388766882376, "CombinedPeriodResult"),
        ("lombscargle", 1.0, 0.0, "LombScarglePeriodFinder"),
    ],
)
def test_best_period(
    method, period, error, outputted_method, timeseries, flux, flux_errors
):

    roto = RoTo(timeseries, flux, flux_errors)

    roto.periods = {
        "lombscargle": PeriodResult(1.0, 0.0, 0.0, "LombScarglePeriodFinder"),
        "fft": PeriodResult(3.0, 0.0, 0.0, "FFTPeriodFinder"),
        "fft2": PeriodResult(2.0, 0.0, 0.0, "FFTPeriodFinder"),
        "gacf": PeriodResult(4.0, 0.0, 0.0, "GACFPeriodFinder"),
        "gp": PeriodResult(5.0, 0.0, 0.0, "GPPeriodFinder"),
    }

    best_period = roto.best_period(method)

    assert best_period == PeriodResult(period, error, error, outputted_method)


@mock.patch("roto.roto.np.mean", autospec=True, return_value=69)
@mock.patch("roto.roto.np.std", autospec=True, return_value=2)
@mock.patch("roto.roto.np.sqrt", autospec=True, return_value=1)
@pytest.mark.parametrize(
    "include, periods",
    [
        ([], [1.0, 3.0, 4.0, 5.0]),
        (["lombscargle"], [1.0]),
        (["gp", "gacf"], [4.0, 5.0]),
    ],
)
def test_best_period_include(
    mock_sqrt, mock_std, mock_mean, include, periods, timeseries, flux, flux_errors
):

    roto = RoTo(timeseries, flux, flux_errors)

    roto.periods = {
        "lombscargle": PeriodResult(1.0, 0.0, 0.0, "LombScarglePeriodFinder"),
        "fft": PeriodResult(3.0, 0.0, 0.0, "FFTPeriodFinder"),
        "gacf": PeriodResult(4.0, 0.0, 0.0, "GACFPeriodFinder"),
        "gp": PeriodResult(5.0, 0.0, 0.0, "GPPeriodFinder"),
    }

    best_period = roto.best_period("mean", include=include)

    mock_mean.assert_called_once_with(periods)
    mock_std.assert_called_once_with(periods)
    mock_sqrt.assert_called_once_with(len(periods))

    assert best_period == PeriodResult(69, 2, 2, "CombinedPeriodResult")


def test_best_period_include_wrong_type(timeseries, flux, flux_errors):

    roto = RoTo(timeseries, flux, flux_errors)

    roto.periods = {
        "lombscargle": PeriodResult(1.0, 0.0, 0.0, "LombScarglePeriodFinder"),
    }

    with pytest.raises(ValueError) as err:
        roto.best_period("mean", include=["non_existent_method"])


@mock.patch("roto.roto.np.mean", autospec=True, return_value=69)
@mock.patch("roto.roto.np.std", autospec=True, return_value=2)
@mock.patch("roto.roto.np.sqrt", autospec=True, return_value=1)
@pytest.mark.parametrize(
    "exclude, periods",
    [
        ([], [1.0, 3.0, 4.0, 5.0]),
        (["lombscargle"], [3.0, 4.0, 5.0]),
        (["gp", "gacf"], [1.0, 3.0]),
    ],
)
def test_best_period_exclude(
    mock_sqrt, mock_std, mock_mean, exclude, periods, timeseries, flux, flux_errors
):

    roto = RoTo(timeseries, flux, flux_errors)

    roto.periods = {
        "lombscargle": PeriodResult(1.0, 0.0, 0.0, "LombScarglePeriodFinder"),
        "fft": PeriodResult(3.0, 0.0, 0.0, "FFTPeriodFinder"),
        "gacf": PeriodResult(4.0, 0.0, 0.0, "GACFPeriodFinder"),
        "gp": PeriodResult(5.0, 0.0, 0.0, "GPPeriodFinder"),
    }

    best_period = roto.best_period("mean", exclude=exclude)

    mock_mean.assert_called_once_with(periods)
    mock_std.assert_called_once_with(periods)
    mock_sqrt.assert_called_once_with(len(periods))

    assert best_period == PeriodResult(69, 2, 2, "CombinedPeriodResult")


def test_best_period_exclude_wrong_type(timeseries, flux, flux_errors):

    roto = RoTo(timeseries, flux, flux_errors)

    roto.periods = {
        "lombscargle": PeriodResult(1.0, 0.0, 0.0, "LombScarglePeriodFinder"),
    }

    with pytest.raises(ValueError) as err:
        roto.best_period("mean", exclude=["non_existent_method"])


def test_best_period_include_exclude_incompatible(timeseries, flux, flux_errors):

    roto = RoTo(timeseries, flux, flux_errors)

    roto.periods = {
        "lombscargle": PeriodResult(1.0, 0.0, 0.0, "LombScarglePeriodFinder"),
    }

    with pytest.raises(ValueError) as err:
        roto.best_period("mean", exclude=["lombscargle"], include=["lombscargle"])
