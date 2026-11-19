import numpy as np
import pytest

from src.calibration import ece


def test_constant_probability_at_prevalence_has_zero_ece():
    y = np.array([0, 0, 0, 1])
    assert ece(y, np.full(y.shape, y.mean())) == pytest.approx(0.0)


def test_inverted_extreme_probabilities_have_maximum_ece():
    y = np.array([0, 0, 1, 1])
    assert ece(y, 1.0 - y) == pytest.approx(1.0)
