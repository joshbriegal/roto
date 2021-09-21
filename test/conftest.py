from unittest import mock

import numpy as np
import pytest


@pytest.fixture
def timeseries():
    return np.linspace(10, 110, 1000)


@pytest.fixture
def period():
    return 20.0


@pytest.fixture
def flux(timeseries, period):
    return np.sin(np.pi * timeseries * 2 / period)


@pytest.fixture
def flux_errors(timeseries):
    return np.random.rand(*timeseries.shape)


mock_model_object = mock.MagicMock()
mock_map_soln_object = mock.MagicMock()


@pytest.fixture
def mock_model():
    return mock_model_object


@pytest.fixture
def mock_map_soln(period):
    solution_dict = {"period": period}
    mock_map_soln_object.__getitem__.side_effect = solution_dict.__getitem__
    return mock_map_soln_object


@pytest.fixture
def mock_build_model(mock_model, mock_map_soln):
    def _method(self, start=None):
        self.model = mock_model
        self.solution = mock_map_soln

        return mock_model, mock_map_soln

    return _method
