"""
BayesianClassifier 单元测试
使用合成数据，覆盖 fit / predict / predict_proba / save / load
"""
import os
import tempfile

import numpy as np
import pytest
from sklearn.model_selection import cross_val_score

from src.classification.bayesian_classifier import BayesianClassifier


@pytest.fixture
def simple_data():
    """生成简单的二分类数据"""
    rng = np.random.RandomState(42)
    X = np.vstack([
        rng.randn(50, 10) + 1.0,   # class 0
        rng.randn(50, 10) - 1.0,   # class 1
    ])
    y = np.array([0] * 50 + [1] * 50)
    return X, y


@pytest.fixture
def multi_data():
    """生成简单的四分类数据"""
    rng = np.random.RandomState(42)
    X = np.vstack([
        rng.randn(30, 6) + np.array([2, 0, 0, 0, 0, 0]),
        rng.randn(30, 6) + np.array([0, 2, 0, 0, 0, 0]),
        rng.randn(30, 6) + np.array([0, 0, 2, 0, 0, 0]),
        rng.randn(30, 6) + np.array([0, 0, 0, 2, 0, 0]),
    ])
    y = np.array([0] * 30 + [1] * 30 + [2] * 30 + [3] * 30)
    return X, y


class TestFit:
    """训练相关测试"""

    def test_fit_sets_attributes(self, simple_data):
        X, y = simple_data
        clf = BayesianClassifier()
        clf.fit(X, y)
        assert clf.classes_ is not None
        assert clf.priors_ is not None
        assert clf.means_ is not None
        assert clf.shared_cov_ is not None
        assert clf.inv_cov_ is not None
        assert len(clf.classes_) == 2

    def test_fit_returns_self(self, simple_data):
        X, y = simple_data
        clf = BayesianClassifier()
        result = clf.fit(X, y)
        assert result is clf

    def test_fit_multiclass(self, multi_data):
        X, y = multi_data
        clf = BayesianClassifier()
        clf.fit(X, y)
        assert len(clf.classes_) == 4
        assert clf.means_.shape == (4, 6)

    def test_fit_prior_sums_to_one(self, simple_data):
        X, y = simple_data
        clf = BayesianClassifier()
        clf.fit(X, y)
        assert np.isclose(np.sum(clf.priors_), 1.0)


class TestPredict:
    """预测相关测试"""

    def test_predict_shape(self, simple_data):
        X, y = simple_data
        clf = BayesianClassifier().fit(X, y)
        pred = clf.predict(X)
        assert pred.shape == y.shape

    def test_predict_accuracy_reasonable(self, simple_data):
        """合成数据线性可分，准确率应很高"""
        X, y = simple_data
        clf = BayesianClassifier().fit(X, y)
        pred = clf.predict(X)
        acc = np.mean(pred == y)
        assert acc > 0.9

    def test_predict_multiclass(self, multi_data):
        X, y = multi_data
        clf = BayesianClassifier().fit(X, y)
        pred = clf.predict(X)
        assert pred.shape == y.shape
        assert set(pred) <= {0, 1, 2, 3}

    def test_predict_proba_sums_to_one(self, simple_data):
        X, y = simple_data
        clf = BayesianClassifier().fit(X, y)
        proba = clf.predict_proba(X)
        assert proba.shape == (len(X), 2)
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_predict_proba_range(self, simple_data):
        X, y = simple_data
        clf = BayesianClassifier().fit(X, y)
        proba = clf.predict_proba(X)
        assert np.all(proba >= 0) and np.all(proba <= 1)


class TestSklearnCompat:
    """sklearn 兼容性测试"""

    def test_cross_val_score(self, multi_data):
        X, y = multi_data
        clf = BayesianClassifier()
        scores = cross_val_score(clf, X, y, cv=3)
        assert len(scores) == 3
        assert np.mean(scores) > 0.25  # 4 分类随机概率基线


class TestPersistence:
    """持久化测试"""

    def test_save_and_load(self, multi_data):
        X, y = multi_data
        clf = BayesianClassifier().fit(X, y)
        pred_orig = clf.predict(X)

        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
            tmp = f.name
        try:
            clf.save(tmp)
            loaded = BayesianClassifier.load(tmp)
            assert loaded.classes_ is not None
            pred_loaded = loaded.predict(X)
            assert np.array_equal(pred_orig, pred_loaded)
        finally:
            os.unlink(tmp)

    def test_load_preserves_attributes(self, simple_data):
        X, y = simple_data
        clf = BayesianClassifier().fit(X, y)
        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
            tmp = f.name
        try:
            clf.save(tmp)
            loaded = BayesianClassifier.load(tmp)
            assert np.allclose(loaded.priors_, clf.priors_)
            assert np.allclose(loaded.means_, clf.means_)
            assert np.allclose(loaded.inv_cov_, clf.inv_cov_)
        finally:
            os.unlink(tmp)
