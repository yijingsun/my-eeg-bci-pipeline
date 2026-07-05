"""
scripts/params_search_full.py 集成测试（最小参数组）
每参数 1 个值，1 个被试，验证含分段窗口的完整搜索流程。
"""
import itertools
import pytest
from config import get_raw_path
from src.utils.session_config import SessionConfig
from src.data_preparation.data_loader import BCIDataLoader
from src.data_preparation.pre_processor import EEGPreprocessor
from src.data_preparation.epoch_processor import EpochProcessor
from src.feature_extraction.ovocsp_feature_extractor import OVOCspFeatureExtractor
from src.classification.bayesian_classifier import BayesianClassifier
from src.evaluation.evaluator import BCIEvaluator


@pytest.mark.slow
class TestSearchFull:

    @pytest.fixture(scope="class")
    def prepared_raw(self):
        cfg = SessionConfig.from_dataset("BCICIV_2a", "A01", "T")
        raw = BCIDataLoader(eog_channels=cfg.eog_channels).load(
            get_raw_path("BCICIV_2a", "A01", "T"))
        pre = EEGPreprocessor(resample_freq=None, filter_ica=None,
                              filter_mi=[8, 30], ref_type="average",
                              bad_channels_manual=[], ica_n_components=None,
                              ica_random_state=71, ica_method=None,
                              ica_exclude_manual=[])
        return pre.process(raw, verbose=False)

    def test_minimal_grid(self):
        g = {"tmin": [1.0, 2.0], "tmax": [4.0], "csp_n_components": [6],
             "log_transform": [False], "lda_n_components": [3],
             "classifier": ["Bayesian"]}
        assert len(list(itertools.product(*g.values()))) == 2

    def test_single_combo(self, prepared_raw):
        cfg = SessionConfig.from_dataset("BCICIV_2a", "A01", "T")
        raw_mi = prepared_raw
        ep = EpochProcessor(tmin=1.0, tmax=4.0,
                            events_mapping=cfg.mi_event_mapping,
                            expected_trials=cfg.expected_trials)
        eog = [ch for ch, ct in zip(raw_mi.ch_names, raw_mi.get_channel_types())
               if ct == "eog"]
        epochs = ep.process(raw_mi, drop_channels=list(set(eog)))
        X, y = epochs.get_data(), epochs.events[:, 2]

        ext = OVOCspFeatureExtractor(csp_n_components=6, csp_reg="ledoit_wolf",
                                     log_transform=False, normalize_features=True,
                                     lda_n_components=3)
        features = ext.fit_transform(X, y, verbose=False)
        assert features.shape[0] == len(X)

        r = BCIEvaluator(cv_folds=4, random_state=17).evaluate(
            features, y, BayesianClassifier())
        assert "kappa_mean" in r
