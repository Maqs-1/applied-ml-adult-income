import numpy as np
import pytest

from src.fairness import audit


def test_audit_reports_known_group_disparities():
    y = np.tile([1, 1, 0, 0], 2)
    pred = np.array([1, 1, 1, 0, 1, 0, 0, 0])
    groups = np.repeat(["A", "B"], 4)

    result = audit(y, pred, groups)

    assert result["by_group"]["A"]["tpr"] == 1.0
    assert result["by_group"]["A"]["fpr"] == 0.5
    assert result["by_group"]["B"]["tpr"] == 0.5
    assert result["by_group"]["B"]["fpr"] == 0.0
    assert result["demographic_parity_diff"] == 0.5
    assert result["demographic_parity_ratio"] == pytest.approx(1 / 3, abs=0.00005)
    assert result["equalized_odds_diff"] == 0.5
