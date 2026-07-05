"""
BCIEvaluator 单元测试
使用合成数据和 sklearn 内置分类器，覆盖交叉验证评估流程。
"""
import numpy as np
import pytest
from sklearn.dummy import DummyClassifier

from src.evaluation.evaluator import BCIEvaluator


@pytest.fixture
def synth_data():
    """生成合成特征和标签（4 分类）"""
    rng = np.random.RandomState(42)
    X = rng.randn(120, 12)
    y = np.array([0] * 30 + [1] * 30 + [2] * 30 + [3] * 30)
    return X, y


class TestEvaluate:

    def test_returns_required_keys(self, synth_data):
        X, y = synth_data
        evaluator = BCIEvaluator(cv_folds=4, random_state=17)
        result = evaluator.evaluate(X, y, DummyClassifier(strategy="stratified"))
        assert "accuracy_mean" in result
        assert "accuracy_std" in result
        assert "kappa_mean" in result
        assert "kappa_std" in result
        assert "cv_scores_accuracy" in result
        assert "cv_scores_kappa" in result

    def test_kappa_range(self, synth_data):
        """Kappa 应在 [-1, 1] 范围内"""
        X, y = synth_data
        evaluator = BCIEvaluator(cv_folds=4, random_state=17)
        result = evaluator.evaluate(X, y, DummyClassifier(strategy="uniform"))
        assert -1 <= result["kappa_mean"] <= 1

    def test_accuracy_range(self, synth_data):
        """准确率应在 [0, 1] 范围内"""
        X, y = synth_data
        evaluator = BCIEvaluator(cv_folds=3, random_state=42)
        result = evaluator.evaluate(X, y, DummyClassifier(strategy="stratified"))
        assert 0 <= result["accuracy_mean"] <= 1

    def test_cv_scores_match_folds(self, synth_data):
        """返回的数组长度等于 folds 数"""
        X, y = synth_data
        evaluator = BCIEvaluator(cv_folds=5, random_state=17)
        result = evaluator.evaluate(X, y, DummyClassifier(strategy="uniform"))
        assert len(result["cv_scores_accuracy"]) == 5
        assert len(result["cv_scores_kappa"]) == 5

    def test_default_folds(self):
        """未传值时使用默认 folds"""
        evaluator = BCIEvaluator(cv_folds=None, random_state=None)
        assert evaluator.cv_folds == 5

    def test_different_folds(self):
        """可指定不同 folds 数"""
        evaluator = BCIEvaluator(cv_folds=10, random_state=17)
        assert evaluator.cv_folds == 10
