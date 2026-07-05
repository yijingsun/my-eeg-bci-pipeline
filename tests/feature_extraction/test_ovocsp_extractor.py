"""
OVOCspFeatureExtractor 单元测试
使用合成 EEG 数据，覆盖 fit / transform / save / load / 输入验证
"""
import os
import tempfile

import numpy as np
import pytest

from src.feature_extraction.ovocsp_feature_extractor import OVOCspFeatureExtractor


@pytest.fixture
def eeg_data():
    """合成 6 通道 × 4 类别的 EEG 数据 (n_trials, n_channels, n_times)"""
    rng = np.random.RandomState(42)
    n_trials_per_class = 40
    n_channels = 6
    t = np.linspace(0, 1, 128)
    X = []
    y = []
    for cls_idx, freq in enumerate([8, 12, 16, 20]):
        base = np.sin(2 * np.pi * freq * t)
        for _ in range(n_trials_per_class):
            # 每个通道在 base 上叠加独立噪声
            trial = np.stack([
                base * (1 + 0.5 * rng.randn(128))
                for _ in range(n_channels)
            ], axis=0)
            X.append(trial)
            y.append(cls_idx)
    return np.array(X), np.array(y)


@pytest.fixture
def trained_extractor(eeg_data):
    X, y = eeg_data
    ext = OVOCspFeatureExtractor(
        csp_n_components=2,
        csp_reg="ledoit_wolf",
        log_transform=True,
        normalize_features=True,
        lda_n_components=None,  # 跳过 LDA，避免合成数据奇异矩阵
    )
    ext.fit(X, y, verbose=False)
    return ext


class TestInit:

    def test_defaults(self):
        ext = OVOCspFeatureExtractor()
        assert ext.csp_n_components == 4
        assert ext.log_transform is True
        assert ext.normalize_features is False
        assert ext.lda_n_components is None
        assert ext.is_fitted is False

    def test_custom_params(self):
        ext = OVOCspFeatureExtractor(
            csp_n_components=3,
            csp_reg="ledoit_wolf",
            log_transform=False,
            normalize_features=True,
            lda_n_components=2,
        )
        assert ext.csp_n_components == 3
        assert ext.csp_reg == "ledoit_wolf"


class TestFit:
    """训练相关测试"""

    def test_fit_sets_is_fitted(self, eeg_data):
        X, y = eeg_data
        ext = OVOCspFeatureExtractor()
        ext.fit(X, y, verbose=False)
        assert ext.is_fitted is True
        assert ext.class_labels is not None

    def test_fit_returns_self(self, eeg_data):
        X, y = eeg_data
        ext = OVOCspFeatureExtractor()
        result = ext.fit(X, y, verbose=False)
        assert result is ext

    def test_fit_creates_pairwise_models(self, eeg_data):
        """4 类 → 6 个类别对"""
        X, y = eeg_data
        ext = OVOCspFeatureExtractor(csp_n_components=2)
        ext.fit(X, y, verbose=False)
        n_classes = 4
        expected_pairs = n_classes * (n_classes - 1) // 2
        assert len(ext.pairwise_csp_models) == expected_pairs

    @pytest.mark.xfail(reason="合成正弦波数据 LDA 类内散布矩阵不满秩", strict=True)
    def test_fit_with_lda(self, eeg_data):
        """LDA 需要更多数据才能稳定——增加试次和噪声"""
        X, y = eeg_data
        ext = OVOCspFeatureExtractor(
            csp_n_components=2,
            csp_reg="ledoit_wolf",
            lda_n_components=2,
        )
        ext.fit(X, y, verbose=False)
        assert ext.lda_projection is not None

    def test_fit_without_lda(self, eeg_data):
        X, y = eeg_data
        ext = OVOCspFeatureExtractor(csp_n_components=2, lda_n_components=None)
        ext.fit(X, y, verbose=False)
        assert ext.lda_projection is None


class TestTransform:
    """特征转换测试"""

    def test_transform_shape(self, eeg_data, trained_extractor):
        X, _ = eeg_data
        features = trained_extractor.transform(X)
        assert features.ndim == 2
        assert features.shape[0] == len(X)

    def test_transform_without_lda(self, eeg_data):
        """无 LDA 时，特征维度 = n_pairs × csp_n_components × 2"""
        X, y = eeg_data
        ext = OVOCspFeatureExtractor(csp_n_components=2, lda_n_components=None)
        ext.fit(X, y, verbose=False)
        features = ext.transform(X)
        n_pairs = 6  # C(4,2)
        assert features.shape[1] == n_pairs * 2 * 2  # 6 pairs × 2 comps × 2

    def test_transform_before_fit_raises(self, eeg_data):
        X, _ = eeg_data
        ext = OVOCspFeatureExtractor()
        with pytest.raises(RuntimeError):
            ext.transform(X)

    def test_fit_transform_returns_array(self, eeg_data):
        X, y = eeg_data
        ext = OVOCspFeatureExtractor(csp_n_components=2)
        result = ext.fit_transform(X, y, verbose=False)
        assert isinstance(result, np.ndarray)

    def test_fit_transform_sets_is_fitted(self, eeg_data):
        X, y = eeg_data
        ext = OVOCspFeatureExtractor(csp_n_components=2)
        ext.fit_transform(X, y, verbose=False)
        assert ext.is_fitted is True


class TestInputValidation:

    def test_2d_input_raises(self):
        X = np.random.randn(10, 20)
        y = np.array([0, 1] * 5)
        ext = OVOCspFeatureExtractor()
        with pytest.raises(ValueError):
            ext.fit(X, y, verbose=False)

    def test_mismatched_lengths_raises(self):
        X = np.random.randn(10, 3, 128)
        y = np.arange(5)
        ext = OVOCspFeatureExtractor()
        with pytest.raises(ValueError):
            ext.fit(X, y, verbose=False)


class TestPersistence:

    def test_save_and_load(self, eeg_data):
        X, y = eeg_data
        ext = OVOCspFeatureExtractor(csp_n_components=2,
                                     csp_reg="ledoit_wolf",
                                     lda_n_components=None)
        ext.fit(X, y, verbose=False)
        orig_features = ext.transform(X)

        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
            tmp = f.name
        try:
            ext.save(tmp)
            loaded = OVOCspFeatureExtractor.load(tmp)
            assert loaded.is_fitted
            loaded_features = loaded.transform(X)
            assert np.allclose(orig_features, loaded_features)
        finally:
            os.unlink(tmp)

    def test_get_params(self, trained_extractor):
        params = trained_extractor.get_params()
        assert "config" in params
        assert params["config"]["csp_n_components"] == 2
