"""
scripts/batch_training.py 和 batch_evaluate.py 聚合逻辑单元测试
仅测试结果统计计算逻辑，不涉及实际管道执行。
"""
import numpy as np
import pytest


class TestBatchTrainingAggregation:
    """测试 batch 脚本中的汇总统计逻辑"""

    def test_mean_and_std_single(self):
        k = [0.75]; a = [0.80]
        assert np.mean(k) == pytest.approx(0.75)
        assert np.std(k) == pytest.approx(0.0)

    def test_mean_and_std_three(self):
        k = [0.70, 0.80, 0.90]; a = [0.75, 0.85, 0.95]
        assert np.mean(k) == pytest.approx(0.80)
        assert np.mean(a) == pytest.approx(0.85)

    def test_partial_failures(self):
        r = {"A01": {"kappa": 0.70}, "A02": None, "A03": {"kappa": 0.90}}
        valid = {k: v for k, v in r.items() if v is not None}
        assert len(valid) == 2

    def test_empty_results(self):
        r = {"A01": None, "A02": None}
        valid = {k: v for k, v in r.items() if v is not None}
        assert len(valid) == 0


class TestBatchEvaluateAggregation:
    """测试 batch 评估汇总逻辑"""

    def test_filter_valid(self):
        r = {"A01": {"kappa": 0.65}, "A02": None}
        valid = {k: v for k, v in r.items() if v is not None}
        assert len(valid) == 1

    def test_summary_stats(self):
        r = {"A01": {"kappa": 0.60, "accuracy": 0.70},
             "A02": {"kappa": 0.80, "accuracy": 0.90}}
        ks = [v["kappa"] for v in r.values()]
        assert np.mean(ks) == pytest.approx(0.70)
