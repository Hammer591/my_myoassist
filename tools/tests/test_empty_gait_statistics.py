"""Missing full gait cycles must be reported without empty NumPy reductions."""

import warnings

import matplotlib.pyplot as plt
import pytest

from rl_train.analyzer.gait_analyze import GaitAnalyzer
from rl_train.analyzer.gait_data import GaitData
from rl_train.analyzer.train_analyzer import TrainAnalyzer


def test_empty_cycles_record_skips_and_preserve_raw_plots(tmp_path, monkeypatch):
    data = GaitData()
    monkeypatch.setattr(GaitAnalyzer, "get_gait_segment_index", lambda self, **kw: [])
    raw_plots = []
    monkeypatch.setattr(GaitAnalyzer, "plot_entire_result", lambda self, **kw: raw_plots.append(kw))
    monkeypatch.setattr(GaitAnalyzer, "plot_contact_data", lambda self, **kw: raw_plots.append(kw))
    before = plt.get_fignums()
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        report = TrainAnalyzer().analyze(data, {}, str(tmp_path), False)
    assert len(raw_plots) == 3
    assert len(report) == 5
    assert all("No complete gait cycles" in item["exception"] for item in report)
    assert plt.get_fignums() == before


@pytest.mark.parametrize("right", [True, False])
def test_toe_off_missing_and_valid_cycles(right, monkeypatch):
    analyzer = GaitAnalyzer(GaitData(), {}, False)
    monkeypatch.setattr(analyzer, "get_gait_segment_index", lambda **kw: [])
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with pytest.raises(ValueError, match="No complete gait cycles"):
            analyzer.get_toe_off_average(is_right_foot_based=right)
    monkeypatch.setattr(analyzer, "get_gait_segment_index", lambda **kw: [(0, 6, 10), (10, 18, 20)])
    assert analyzer.get_toe_off_average(is_right_foot_based=right) == pytest.approx(0.7)
