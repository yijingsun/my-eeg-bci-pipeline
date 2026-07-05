"""
data_preparation 单元测试
使用合成 MNE Raw 对象，覆盖 BCIDataLoader / EEGPreprocessor / EpochProcessor
"""
import numpy as np
import mne
import pytest

from src.data_preparation.data_loader import BCIDataLoader
from src.data_preparation.pre_processor import EEGPreprocessor
from src.data_preparation.epoch_processor import EpochProcessor


# ============================================================
# 辅助：生成合成 MNE Raw
# ============================================================

def _make_synth_raw(n_channels=22, n_times=2000, sfreq=250, ch_types="eeg"):
    """生成一个包含注释事件和 dig 信息的合成 Raw 对象"""
    rng = np.random.RandomState(42)
    data = rng.randn(n_channels, n_times) * 1e-6
    # 使用标准 10-20 通道名以匹配 montage
    ch_names = ["Fz", "C3", "Cz", "C4", "Pz",
                "FC5", "FC1", "FC2", "FC6",
                "C5", "C1", "C2", "C6",
                "CP5", "CP1", "CP2", "CP6",
                "P5", "P1", "P2", "P6",
                "POz", "Oz"][:n_channels]
    info = mne.create_info(ch_names, sfreq, ch_types=ch_types)
    montage = mne.channels.make_standard_montage("standard_1020")
    info.set_montage(montage, on_missing="warn")
    raw = mne.io.RawArray(data, info)
    for onset, desc in [(1.0, "769"), (2.0, "770"), (3.0, "771"), (4.0, "772")]:
        raw.annotations.append(onset=onset, duration=0, description=desc)
    return raw


def _make_eog_raw():
    """带 EOG 通道的合成 Raw"""
    rng = np.random.RandomState(42)
    n_eeg, n_eog = 22, 3
    data = rng.randn(n_eeg + n_eog, 2000) * 1e-6
    ch_names = [f"EEG-{i:03d}" for i in range(n_eeg)] + ["EOG-left", "EOG-central", "EOG-right"]
    ch_types = ["eeg"] * n_eeg + ["eog"] * n_eog
    info = mne.create_info(ch_names, 250, ch_types=ch_types)
    return mne.io.RawArray(data, info)


# ============================================================
# BCIDataLoader
# ============================================================

class TestBCIDataLoader:

    def test_default_construction(self):
        loader = BCIDataLoader()
        assert loader.eog_channels == []
        assert loader.set_montage is False

    def test_custom_eog_channels(self):
        loader = BCIDataLoader(eog_channels=["EOG-left", "EOG-right"])
        assert "EOG-left" in loader.eog_channels
        assert "EOG-right" in loader.eog_channels

    def test_channel_renaming_default(self):
        loader = BCIDataLoader()
        assert "EEG-Fz" in loader.channel_renaming
        assert loader.channel_renaming["EEG-Fz"] == "Fz"

    def test_custom_channel_renaming(self):
        custom = {"EEG-A": "A1", "EEG-B": "B1"}
        loader = BCIDataLoader(channel_renaming=custom)
        assert loader.channel_renaming == custom

    def test_get_params(self):
        loader = BCIDataLoader(eog_channels=["EOG-left"], set_montage=True)
        params = loader.get_params()
        assert params["eog_channels"] == ["EOG-left"]
        assert params["set_montage"] is True
        assert "channel_renaming" in params


# ============================================================
# EEGPreprocessor
# ============================================================

class TestEEGPreprocessor:

    @pytest.fixture
    def raw(self):
        return _make_synth_raw()

    def test_default_construction(self):
        pre = EEGPreprocessor()
        assert pre.resample_freq is None
        assert pre.ica_n_components == 20

    def test_resample(self, raw):
        pre = EEGPreprocessor(resample_freq=100)
        result = pre.resample(raw)
        assert result.info["sfreq"] == 100

    def test_resample_skips_when_none(self, raw):
        pre = EEGPreprocessor(resample_freq=None)
        result = pre.resample(raw)
        assert result.info["sfreq"] == raw.info["sfreq"]

    def test_fix_bad_channels(self, raw):
        raw.info["bads"] = ["C3"]
        pre = EEGPreprocessor(bad_channels_manual=["C4"])
        result = pre.fix_bad_channels(raw)
        assert "C3" not in result.info["bads"]
        assert "C4" not in result.info["bads"]

    def test_apply_reference(self, raw):
        pre = EEGPreprocessor(ref_type="average")
        result = pre.apply_reference(raw)
        assert result is not None

    def test_apply_reference_skips_when_none(self, raw):
        pre = EEGPreprocessor(ref_type=None)
        result = pre.apply_reference(raw)
        assert result is not None  # 不报错即通过

    def test_mi_filter(self, raw):
        pre = EEGPreprocessor(filter_mi=[8, 30])
        result = pre.apply_mi_filter(raw)
        assert result is not None

    def test_process_runs_full_pipeline(self, raw):
        pre = EEGPreprocessor(
            resample_freq=100,
            ref_type="average",
            filter_mi=[8, 30],
        )
        result = pre.process(raw, verbose=False)
        assert result is not None
        assert result.info["sfreq"] == 100

    def test_get_params(self):
        pre = EEGPreprocessor(
            resample_freq=128,
            ref_type="average",
            filter_ica=[1, 40],
            filter_mi=[8, 30],
        )
        params = pre.get_params()
        assert params["resample_freq"] == 128
        assert params["ref_type"] == "average"
        assert params["filter_ica"] == [1, 40]


# ============================================================
# EpochProcessor
# ============================================================

class TestEpochProcessor:

    @pytest.fixture
    def raw(self):
        return _make_synth_raw()

    @pytest.fixture
    def processor(self):
        # MNE 对 "769"-"772" 自动分配 event_id 为 1,2,3,4
        return EpochProcessor(
            tmin=0.0,
            tmax=1.0,
            events_mapping={769: 1, 770: 2, 771: 3, 772: 4},
            expected_trials=4,
        )

    def test_default_construction(self):
        ep = EpochProcessor()
        assert ep.tmin == 1.0
        assert ep.tmax == 4.0

    def test_extract_events(self, raw, processor):
        events = processor.extract_events(raw)
        assert events.ndim == 2
        assert events.shape[1] == 3  # [sample, prev, id]

    def test_pick_events(self, raw, processor):
        events = processor.extract_events(raw)
        mi_events = processor.pick_events(events)
        assert mi_events.shape[0] == 4  # 4 个 MI 事件
        # 重映射后标签应为 1-4
        assert set(mi_events[:, 2]) == {1, 2, 3, 4}

    def test_pick_events_wrong_count_raises(self, raw):
        processor = EpochProcessor(
            tmin=0.0, tmax=1.0,
            events_mapping={"769": 7, "770": 8},
            expected_trials=999,  # 错误预期
        )
        events = processor.extract_events(raw)
        with pytest.raises(AssertionError):
            processor.pick_events(events)

    def test_create_epochs(self, raw, processor):
        events = processor.extract_events(raw)
        mi_events = processor.pick_events(events)
        epochs = processor.create_epochs(raw, mi_events)
        assert isinstance(epochs, mne.Epochs)

    def test_create_epochs_drop_channels(self, raw, processor):
        events = processor.extract_events(raw)
        mi_events = processor.pick_events(events)
        epochs = processor.create_epochs(raw, mi_events, drop_channels=["C3"])
        assert "C3" not in epochs.ch_names

    def test_process_one_stop(self, raw, processor):
        epochs = processor.process(raw, verbose=False)
        assert isinstance(epochs, mne.Epochs)
        assert len(epochs) == 4

    def test_get_params(self):
        ep = EpochProcessor(tmin=0.0, tmax=2.0, expected_trials=100)
        params = ep.get_params()
        assert params["tmin"] == 0.0
        assert params["tmax"] == 2.0
