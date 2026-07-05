"""
scripts/search_best_params.py 集成测试（缩减参数组）
每个参数 1-2 个值，单个被试，验证网格搜索流程。
"""
import itertools
import numpy as np
import mne
import pytest
from config import get_epoch_path
from src.feature_extraction.ovocsp_feature_extractor import OVOCspFeatureExtractor
from src.classification.bayesian_classifier import BayesianClassifier
from src.evaluation.evaluator import BCIEvaluator


@pytest.mark.slow
class TestSearchBestParams:

    @pytest.fixture(scope="class")
    def data(self):
        ep = mne.read_epochs(get_epoch_path("BCICIV_2a", "A01", "T"),
                             preload=True, verbose=False)
        return ep.get_data(), ep.events[:, 2]

    def test_grid_count(self):
        g = {"csp_n_components": [2, 6], "csp_reg": ["ledoit_wolf"],
             "normalize_features": [True], "lda_n_components": [3],
             "log_transform": [False], "classifier": ["Bayesian"]}
        assert len(list(itertools.product(*g.values()))) == 2

    def test_evaluate_one(self, data):
        X, y = data
        ext = OVOCspFeatureExtractor(csp_n_components=6, csp_reg="ledoit_wolf",
                                     log_transform=False, normalize_features=True,
                                     lda_n_components=3)
        features = ext.fit_transform(X, y, verbose=False)
        assert features.ndim == 2

        result = BCIEvaluator(cv_folds=4, random_state=17).evaluate(
            features, y, BayesianClassifier())
        assert "kappa_mean" in result
        assert -1 <= result["kappa_mean"] <= 1
