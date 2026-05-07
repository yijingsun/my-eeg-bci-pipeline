"""
基础预处理器
降采样 → 坏通道修复 → 滤波 → 重参考
"""
import mne
from typing import Optional, List, Tuple
from src.utils.session_config import SessionConfig


class EEGPreprocessor:
    """EEG 信号预处理器"""

    def __init__(self, config: SessionConfig):
        self.config = config
        self.bad_channels_manual = config.bad_channels_manual or []

    def resample(self, raw: mne.io.Raw) -> mne.io.Raw:
        """降采样"""
        if self.config.resample_freq is None:
            return raw
        raw = raw.copy()
        raw.resample(self.config.resample_freq, npad='auto')
        return raw

    def fix_bad_channels(self, raw: mne.io.Raw) -> mne.io.Raw:
        """标记坏通道并插值修复"""
        raw = raw.copy()
        bads = list(set(raw.info['bads'] + self.config.bad_channels_manual))
        raw.info['bads'] = bads
        if bads:
            raw.interpolate_bads(reset_bads=True)
        return raw

    def apply_ica_filter_and_ref(self, raw: mne.io.Raw) -> mne.io.Raw:
        """粗滤波 + 重参考（为 ICA 做准备）"""
        raw = raw.copy()
        raw.filter(
            l_freq=self.config.filter_ica[0],
            h_freq=self.config.filter_ica[1],
            fir_design=self.config.filter_design_ica,
            verbose=False
        )
        mne.set_eeg_reference(raw, self.config.ref_type)
        return raw

    def apply_mi_filter(self, raw: mne.io.Raw) -> mne.io.Raw:
        """精滤波（保留运动想象相关频段）"""
        raw = raw.copy()
        raw.filter(
            l_freq=self.config.filter_mi[0],
            h_freq=self.config.filter_mi[1],
            fir_design=self.config.filter_design_mi,
            verbose=False
        )
        return raw