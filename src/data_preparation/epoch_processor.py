"""
Epoch 处理器
事件提取 → 去重 → 筛选 MI → 映射 ID → 分段
"""
import mne
import numpy as np


class EpochProcessor:
    """事件提取与分段处理器"""

    def __init__(self, tmin: float = None, tmax: float = None, 
                 events_mapping: dict = None, expected_trials: int = None):
        self.tmin = tmin if tmin is not None else 1.0
        self.tmax = tmax if tmax is not None else 4.0
        self.events_mapping = events_mapping if events_mapping is not None else {}
        self.expected_trials = expected_trials if expected_trials is not None else 0

    def extract_events(self, raw: mne.io.Raw) -> np.ndarray:
        """从 raw 中提取事件并去重"""
        events, _ = mne.events_from_annotations(raw, verbose=False)
        events = np.unique(events, axis=0)
        return events

    def pick_events(self, events: np.ndarray, verbose: bool = False) -> np.ndarray:
        """
        筛选 MI 事件并重编号为 1-4

        原始映射: 769→7(左手) 770→8(右手) 771→9(双脚) 772→10(舌头)
        重映射后: 1(左手) 2(右手) 3(双脚) 4(舌头)
        """
        # 筛选
        event_ids = list(self.events_mapping.values())
        events_mi = events[np.isin(events[:, 2], event_ids)]

        # 重映射到连续 ID
        mapping_dict = {old: new + 1 for new, old in enumerate(event_ids)}
        events_mi[:, 2] = np.array([mapping_dict[e] for e in events_mi[:, 2]])


        # 校验
        assert len(events_mi) == self.expected_trials, (
            f"MI 试次数异常: 期望 {self.expected_trials}，实际 {len(events_mi)}"
        )
        return events_mi

    def create_epochs(
        self,
        raw: mne.io.Raw,
        events: np.ndarray,
        drop_channels: list = None
    ) -> mne.Epochs:
        """创建 Epochs 并丢弃指定通道"""
        epochs = mne.Epochs(
            raw,
            events,
            tmin=self.tmin,
            tmax=self.tmax,
            baseline=None,
            preload=True,
            verbose=False
        )
        if drop_channels is not None and len(drop_channels) > 0:
            existing = [ch for ch in drop_channels if ch in epochs.ch_names]
            if existing:
                epochs.drop_channels(existing)
        return epochs

    def process(
        self,
        raw: mne.io.Raw,
        drop_channels: list = None,
        verbose: bool = False
    ) -> mne.Epochs:
        """一站式：事件提取 → 筛选 → 分段"""
        events = self.extract_events(raw)
        events_mi = self.pick_events(events, verbose=verbose)
        epochs = self.create_epochs(raw, events_mi, drop_channels)
        return epochs
    
    def get_params(self) -> dict:
        return {
            'tmin': self.tmin,
            'tmax': self.tmax,
            'events_mapping': self.events_mapping,
            'expected_trials': self.expected_trials,
        }