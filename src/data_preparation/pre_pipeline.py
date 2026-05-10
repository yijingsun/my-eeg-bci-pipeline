# src/data_preparation/pipeline.py
"""
完整预处理管道：加载 → 预处理 → 分段 → 保存
"""
import os
import numpy as np
import mne
from config import get_raw_path, get_epoch_dir, get_label_dir, ensure_dir
from src.utils.session_config import SessionConfig
from .data_loader import BCIDataLoader
from .preprocessor import EEGPreprocessor
from .epoch_processor import EpochProcessor


class DataPipeline:
    """数据预处理管道"""

    def __init__(self, dataset_name: str, subject_id: str, session: str):
        if dataset_name is None or subject_id is None or session is None:
            raise ValueError("dataset_name, subject_id, session 不能为空")
        self.dataset_name = dataset_name
        self.subject_id = subject_id
        self.session = session
        # ---- 加载配置 ----
        self.cfg = SessionConfig.from_dataset(dataset_name, subject_id, session)

    def run(self, save: bool = True, save_label: bool = True, verbose: bool = True) -> mne.Epochs:
        """执行完整预处理管道"""
        config = self.cfg

        # ---- 初始化处理器 ----
        loader = BCIDataLoader(eog_channels=config.get('eog_channels'), 
                               set_montage=config.get('set_montage'), 
                               channel_renaming=config.get('channel_renaming'), 
                               montage_name=config.get('montage_name'))

        preprocessor = EEGPreprocessor(
            resample_freq=config.get('resample_freq'),
            filter_ica=config.get('filter_ica'),
            filter_mi=config.get('filter_mi'),
            ref_type=config.get('ref_type'),
            bad_channels_manual=config.get('bad_channels_manual'),
            ica_n_components=config.get('ica_n_components'),
            ica_random_state=config.get('ica_random_state'),
            ica_method=config.get('ica_method'),
            ica_exclude_manual=config.get('ica_exclude_manual'),
        )

        epoch_processor = EpochProcessor(
            tmin=config.get('tmin'),
            tmax=config.get('tmax'),
            events_mapping=config.get('mi_event_mapping'),
            expected_trials=config.get('expected_trials'),
        )

        if verbose:
            print("=" * 60)
            print(f"BCI 预处理管道")
            print(f"  数据集: {self.dataset_name}")
            print(f"  被试:   {self.subject_id}{self.session}")
            print("=" * 60)

        # ---- [1] 加载原始数据 ----
        if verbose:
            print("\n[1/3] 加载原始数据")
        filepath = get_raw_path(self.dataset_name, self.subject_id, self.session, config.get('file_type'))
        raw = loader.load(filepath)
        if verbose:
            print(f"\n  ✓ 原始数据加载已完成: {filepath}")
            print(f"    通道数: {len(raw.ch_names)}")
            print(f"    采样点: {len(raw.times)}")
            print(f"    采样率: {raw.info['sfreq']}Hz")
            print(f"    事件数: {len(raw.annotations)}")
            print(f"    Bad 通道: {raw.info['bads']}")

        # ---- [2] 预处理（所有步骤合并为一次 process 调用）----
        if verbose:
            print("\n[2/3] 开始预处理")
        raw = preprocessor.process(raw, verbose=verbose)
        if verbose:
            print(f"\n✓ 预处理已完成")

        # ---- [3] 分段 ----
        if verbose:
            print(f"\n[3/3] 事件提取与分段 (t=[{config.get('tmin')}, {config.get('tmax')}]s)")

        # 动态获取 EOG 通道（已在加载时标记为 'eog' 类型）
        eog_chs = [ch for ch, ch_type in zip(raw.ch_names, raw.get_channel_types()) if ch_type == 'eog']
        chs_to_drop = list(set(eog_chs + raw.info['bads']))

        epochs = epoch_processor.process(raw, drop_channels=chs_to_drop, verbose=verbose)

        if verbose:
            print(f"\n✓ 事件提取与分段已完成")
            print(f"    试次数: {len(epochs)}")
            print(f"    通道数: {len(epochs.ch_names)}")
            print(f"    采样点: {epochs.get_data().shape[1]}")
            print(f"    事件标签: {np.unique(epochs.events[:, 2])}")
            print(f"    事件分布: {np.bincount(epochs.events[:, 2])[1:]}")

        # ---- 保存epochs ----
        if save:
            if verbose:
                print("\n   >>> 保存 epochs...")
            epoch_dir = get_epoch_dir(self.dataset_name)
            ensure_dir(epoch_dir)
            epoch_file = os.path.join(epoch_dir, f'{self.subject_id}{self.session}_epo.fif')
            epochs.save(epoch_file, overwrite=True)
            if verbose:
                print(f"  → {epoch_file}")
        
        # ---- 保存标签 T保存 E用官方标记的标签无需保存 ----
        if save_label:
            if verbose:
                print("     >>> 保存标签...")
            label_dir = get_label_dir(self.dataset_name)
            ensure_dir(label_dir)
            label_file = os.path.join(label_dir, f'{self.subject_id}{self.session}_labels.npy')
            np.save(label_file, epochs.events[:, 2])
            if verbose:
                print(f"  → {label_file}")
        
        if verbose:
            print(f"✓ 预处理完成！")

        return epochs