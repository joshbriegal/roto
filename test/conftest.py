import pytest
import numpy as np

@pytest.fixture
def timeseries():
    return np.linspace(0, 100, 1000)

@pytest.fixture
def period():
    return 20.0

@pytest.fixture
def flux(timeseries, period):
    return np.sin(np.pi * timeseries * 2 / period)


@pytest.fixture
def flux_errors(timeseries):
    return np.random.rand(*timeseries.shape)