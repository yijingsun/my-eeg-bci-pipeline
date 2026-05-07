"""
完整预处理管道
串联所有步骤：加载 → 预处理 → ICA → 分段 → 保存
"""
import os
import numpy as np
import mne
from config import get_raw_path, get_epoch_path, ensure_dir
from src.utils import load_session_config
from .data_loader import BCIDataLoader
from .preprocessor import EEGPreprocessor
from .artifact_remover import ArtifactRemover
from .epoch_processor import EpochProcessor


class DataPipeline:
    """
    数据预处理管道

    使用示例:
        pipeline = DataPipeline(dataset_name='BCICIV_2a')
        epochs = pipeline.run(subject_id='A01', session='T')
    """

    def __init__(self, dataset_name: str = 'BCICIV_2a'):
        self.dataset_name = dataset_name

    def run(
        self,
        subject_id: str = 'A01',
        session: str = 'T',
        file_type: str = 'gdf',
        save: bool = True,
        verbose: bool = True
    ) -> mne.Epochs:
        """
        执行完整预处理管道
        """
        
        # ---从json读配置 ---
        cfg = load_session_config(self.dataset_name, subject_id, session)

        # --- 初始化处理器 ---
        self.loader = BCIDataLoader(eog_channels=cfg.get('eog_channels'))

        self.preprocessor = EEGPreprocessor(
            resample_freq=cfg.get('resample_freq'),
            filter_ica=cfg.get('filter_ica'),
            ref_type=cfg.get('ref_type'),
            bad_channels_manual=cfg.get('bad_channels_manual')
        )

        self.artifact_remover = ArtifactRemover(
            ica_n_components=cfg.get('ica_n_components'),
            ica_random_state=cfg.get('ica_random_state'),
            ica_method=cfg.get('ica_method'),
            ica_exclude_manual=cfg.get('ica_exclude_manual')
        )

        self.epoch_processor = EpochProcessor(
            tmin=cfg.get('tmin'),
            tmax=cfg.get('tmax'),
            events_mapping=cfg.get('events_mapping'),
            expected_trials=cfg.get('expected_trials'),
            bad_trials_manual=cfg.get('bad_trials_manual')
        )

        if verbose:
            print("=" * 60)
            print(f"BCI 预处理管道已初始化")
            print(f"  数据集: {self.dataset_name}")
            print(f"  被试:   {subject_id}{session}")
            print("=" * 60)

        # --- [1] 加载 ---
        filepath = get_raw_path(self.dataset_name, subject_id, session, file_type)
        if verbose:
            print(f"\n[1/8] 加载原始数据")
            print(f"  文件: {filepath}")
        raw = self.loader.load(filepath, verbose=False)
        if verbose:
            print(f"  通道数: {len(raw.ch_names)}")
            print(f"  采样率: {raw.info['sfreq']} Hz")
            print(f"  时长:   {raw.times[-1]:.0f}s")

        # --- [2] 降采样 ---
        if verbose: 
            if self.preprocessor.resample_freq is not None:
                raw = self.preprocessor.resample(raw)
                print(f"\n[2/8] 降采样 → {self.preprocessor.resample_freq} Hz")
            else:
                print("\n[2/8] 无需降采样")

        # --- [3] 坏通道修复 ---
        raw = self.preprocessor.fix_bad_channels(raw)
        if verbose:
            print("\n[3/8] 坏通道检测与插值")
            auto_bads = raw.info['bads']
            manual_bads = self.preprocessor.bad_channels_manual
            print(f"  自动坏通道: {auto_bads if len(auto_bads) > 0 else '无'}")
            print(f"  手动坏通道: {manual_bads if len(manual_bads) > 0 else '无'}")

        # --- [4] ICA 准备 ---
        if verbose:
            print(f"\n[4/8] ICA 准备（滤波 {self.preprocessor.filter_ica}Hz + {self.preprocessor.ref_type} 参考）")
        raw_ica = self.preprocessor.apply_ica_filter_and_ref(raw)

        # --- [5] ICA 伪迹去除 ---
        if verbose:
            print("\n[5/8] ICA 拟合与眼电去除")
        self.artifact_remover.fit(raw_ica)
        auto_exclude = self.artifact_remover.find_auto_artifacts(raw_ica)
        manual_exclude = self.artifact_remover.ica_exclude_manual
        
        all_artifacts = self.artifact_remover.get_all_artifacts()
        raw_clean = self.artifact_remover.apply(raw_ica, all_artifacts)
        if verbose:    
            print(f"  自动识别的眼电成分: {auto_exclude}")
            print(f"  手工排除的成分: {manual_exclude}")
            print(f"  最终去除的成分: {all_artifacts}")

        # --- [6] 精滤波 ---
        if verbose:
            print(f"\n[6/8] 精滤波（频段 {self.preprocessor.filter_mi}Hz）")
        raw_mi = self.preprocessor.apply_mi_filter(raw_clean)

        # --- [7] 分段 ---
        if verbose:
            print(f"\n[7/8] 事件提取与分段（t=[{self.epoch_processor.tmin}, {self.epoch_processor.tmax}]s）")

        # 直接从 raw 的动态属性中获取 EOG 通道
        eog_chs = [ch for ch, ch_type in zip(raw_mi.ch_names, raw_mi.get_channel_types()) 
           if ch_type == 'eog']
        chs_to_drop = np.unique(eog_chs + raw_mi.info['bads']) # 分段时去掉EOG通道和坏通道
        epochs = self.epoch_processor.process(raw_mi, drop_channels=chs_to_drop)
        epochs_clean = self.epoch_processor.drop_bad_trials(epochs)

        if verbose:
            print(f"  已去除坏通道: {raw_mi.info['bads']} 已去除EOG通道: {eog_chs}")
            dropped_idx = [i for i, r in enumerate(epochs_clean.drop_log) if r]
            print(f"  总共剔除 {len(dropped_idx)} 个试次: {dropped_idx}")
            print(f"  已完成分段，统计信息:")
            print(f"  试次数: {len(epochs_clean)}")
            print(f"  通道数: {len(epochs_clean.ch_names)}")
            print(f"  事件映射: {np.unique(epochs_clean.events[:, 2])}")
            print(f"  事件分布: {np.bincount(epochs_clean.events[:, 2])[1:]}") # np.bincount [0, 72, 72, 72, 72] 0是索引, 再取[1:]就得到[72, 72, 72, 72]

        

        # --- [8] 保存 ---
        if save:
            if verbose:
                print("\n[8/8] 保存 epochs")
            epoch_path = get_epoch_path(self.dataset_name, subject_id, session)
            ensure_dir(os.path.dirname(epoch_path))
            epochs_clean.save(epoch_path, overwrite=True)
            if verbose:
                print(f"  → {epoch_path}")

        if verbose:
            print("\n" + "=" * 60)
            print(f"✅ 完成! epochs shape: {epochs_clean.get_data().shape}")
            print("=" * 60 + "\n")

        return epochs_clean