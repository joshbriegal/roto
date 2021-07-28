import math
from unittest import mock

import numpy as np
import pytest
from numpy.testing import assert_equal

from src.methods.periodfinder import PeriodResult


class TestPeriodResult:
    @pytest.fixture
    def periodresult1(self):
        return PeriodResult(2.0, 0.1, 0.1, "somemethod")

    @pytest.fixture
    def periodresult2(self):
        return PeriodResult(10.0, 0.4, 0.6, "someothermethod")

    def test_periodresult_add(self, periodresult1, periodresult2):
        new_periodresult = periodresult1 + periodresult2

        assert new_periodresult.period == 12.0
        assert new_periodresult.neg_error == math.sqrt((0.1 * 0.1) + (0.4 * 0.4))
        assert new_periodresult.pos_error == math.sqrt((0.1 * 0.1) + (0.6 * 0.6))
        assert new_periodresult.method == "CombinedPeriodResult"

    def test_periodresult_sub(self, periodresult1, periodresult2):
        new_periodresult = periodresult1 - periodresult2

        assert new_periodresult.period == 8.0
        assert new_periodresult.neg_error == math.sqrt((0.1 * 0.1) + (0.4 * 0.4))
        assert new_periodresult.pos_error == math.sqrt((0.1 * 0.1) + (0.6 * 0.6))
        assert new_periodresult.method == "CombinedPeriodResult"

    def test_periodresult_div(self, periodresult2):
        new_periodresult = periodresult2 / 2.0

        assert new_periodresult.period == 5.0
        assert new_periodresult.neg_error == 0.2
        assert new_periodresult.pos_error == 0.3
        assert new_periodresult.method == "CombinedPeriodResult"
