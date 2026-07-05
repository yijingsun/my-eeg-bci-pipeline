"""
Pipeline 组装逻辑测试（纯函数版）
"""
import os
import json
import tempfile
import shutil

import numpy as np
import mne
import pytest

from src.data_preparation.data_loader import BCIDataLoader
from src.utils.session_config import SessionConfig
from scripts.train import run_preprocessing, run_feature_extraction, run_classification


def _make_synth_raw(n_channels=22, n_times=80000, sfreq=250, n_trials=40):
    rng = np.random.RandomState(42)
    ch_names = ["Fz","C3","Cz","C4","Pz","FC5","FC1","FC2","FC6",
                "C5","C1","C2","C6","CP5","CP1","CP2","CP6",
                "P5","P1","P2","P6","POz","Oz"][:n_channels]
    data = rng.randn(n_channels, n_times) * 1e-6
    info = mne.create_info(ch_names, sfreq, ch_types="eeg")
    info.set_montage(mne.channels.make_standard_montage("standard_1020"), on_missing="warn")
    raw = mne.io.RawArray(data, info)
    for trial in range(n_trials):
        onset = trial * 7.0 + 2.0
        raw.annotations.append(onset=onset, duration=0,
                               description=str([769,770,771,772][trial % 4]))
    return raw


def _make_cfg():
    tmp = tempfile.mkdtemp(prefix="test_cfg_")
    dp = os.path.join(tmp, "config.json")
    cfg = {
        "default": {
            "eog_channels":[],"set_montage":True,"montage_name":"standard_1020",
            "resample_freq":None,"ref_type":"average","bad_channels_manual":[],
            "filter_ica":None,"ica_method":None,"ica_n_components":None,
            "ica_random_state":71,"ica_exclude_manual":[],
            "filter_mi":[8,30],"tmin":1.0,"tmax":4.0,
            "mi_event_mapping":{769:1,770:2,771:3,772:4},"expected_trials":40,
            "csp_n_components":2,"csp_reg":"ledoit_wolf","log_transform":False,
            "normalize_features":True,"lda_n_components":None,
            "classify_cv_folds":3,"classify_random_state":42,"classify_do_cv":True,
        }, "S01":{"T":{}}
    }
    json.dump(cfg, open(dp, "w"))
    return dp


@pytest.fixture
def cfg():
    p = _make_cfg()
    yield SessionConfig.from_json_file(p, "S01", "T")
    shutil.rmtree(os.path.dirname(p), ignore_errors=True)


@pytest.fixture
def patch_loader(monkeypatch):
    def patched(self, filepath, verbose=False):
        raw = _make_synth_raw()
        for ch in self.eog_channels:
            if ch in raw.ch_names:
                raw.set_channel_types({ch:"eog"}, verbose=False)
        self._raw = raw
        return raw
    monkeypatch.setattr(BCIDataLoader, "load", patched)


@pytest.mark.slow
class TestPreprocessing:
    def test_save(self, cfg, patch_loader):
        d = tempfile.mkdtemp()
        try:
            epochs = run_preprocessing("/fake/S01T.gdf", d, d, cfg)
            assert isinstance(epochs, mne.Epochs)
            assert os.path.exists(f"{d}/S01T_epo.fif")
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_no_save(self, cfg, patch_loader):
        d = tempfile.mkdtemp()
        try:
            epochs = run_preprocessing("/fake/S01T.gdf", d, d, cfg, save=False)
            assert isinstance(epochs, mne.Epochs)
            assert not os.path.exists(f"{d}/S01T_epo.fif")
        finally:
            shutil.rmtree(d, ignore_errors=True)


@pytest.mark.slow
class TestFeature:
    def test_save(self, cfg, patch_loader):
        pre_d = tempfile.mkdtemp()
        feat_d = tempfile.mkdtemp()
        try:
            run_preprocessing("/fake/S01T.gdf", pre_d, pre_d, cfg, save_label=True)
            features, ext = run_feature_extraction(
                f"{pre_d}/S01T_epo.fif", f"{pre_d}/S01T_labels.npy", feat_d, cfg)
            assert features.ndim == 2
            assert os.path.exists(f"{feat_d}/S01T_ovocsp_features.npy")
        finally:
            shutil.rmtree(pre_d, ignore_errors=True)
            shutil.rmtree(feat_d, ignore_errors=True)


@pytest.mark.slow
class TestClassification:
    def test_save(self, cfg, patch_loader):
        pre_d, feat_d, clf_d, res_d = (tempfile.mkdtemp() for _ in range(4))
        try:
            run_preprocessing("/fake/S01T.gdf", pre_d, pre_d, cfg, save_label=True)
            run_feature_extraction(f"{pre_d}/S01T_epo.fif",
                                   f"{pre_d}/S01T_labels.npy", feat_d, cfg)
            result, clf = run_classification(
                f"{feat_d}/S01T_ovocsp_features.npy",
                f"{pre_d}/S01T_labels.npy", clf_d, res_d, cfg)
            assert "cv_summary" in result
            assert os.path.exists(f"{clf_d}/S01T_bayesian_clf.joblib")
        finally:
            for d in [pre_d, feat_d, clf_d, res_d]:
                shutil.rmtree(d, ignore_errors=True)

    def test_reproducible(self, cfg, patch_loader):
        pre_d, feat_d, clf_d, res_d = (tempfile.mkdtemp() for _ in range(4))
        try:
            run_preprocessing("/fake/S01T.gdf", pre_d, pre_d, cfg, save_label=True)
            run_feature_extraction(f"{pre_d}/S01T_epo.fif",
                                   f"{pre_d}/S01T_labels.npy", feat_d, cfg)
            results = []
            for _ in range(2):
                r, _ = run_classification(
                    f"{feat_d}/S01T_ovocsp_features.npy",
                    f"{pre_d}/S01T_labels.npy", clf_d, res_d, cfg, save=False)
                results.append(r)
            assert results[0]["overall_accuracy"] == results[1]["overall_accuracy"]
        finally:
            for d in [pre_d, feat_d, clf_d, res_d]:
                shutil.rmtree(d, ignore_errors=True)
