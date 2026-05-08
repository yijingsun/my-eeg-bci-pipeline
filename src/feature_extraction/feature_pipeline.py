"""
特征提取管道：加载 epochs → 训练 CSP → 保存特征和提取器
"""
import os
import numpy as np
import mne
from config import get_epoch_path, get_dataset_dir, ensure_dir
from src.utils.config_loader import load_session_config
from src.feature_extraction.ovocsp_feature_extractor import OVOCspFeatureExtractor


class FeatureExtractionPipeline:
    """OVO-CSP 特征提取流水线"""

    def __init__(self, dataset_name: str = 'BCICIV_2a'):
        self.dataset_name = dataset_name
        self.save_features = True
        self.save_extractor = True
        self.verbose = True

    def run(self, subject_id: str = 'A01', session: str = 'T'):
        """
        执行特征提取：读取 epochs → 训练 CSP → 保存 `.npz` 和 `.joblib`
        """
        # 1. 加载配置
        cfg = load_session_config(self.dataset_name, subject_id, session)

        # 2. 读取预处理好的 epochs
        epoch_path = get_epoch_path(self.dataset_name, subject_id, session)
        if self.verbose:
            print(f"加载 epochs: {epoch_path}")
        epochs = mne.read_epochs(epoch_path, preload=True)
        X = epochs.get_data()
        y = epochs.events[:, 2]

        if self.verbose:
            print(f"数据形状: {X.shape}, 类别: {np.unique(y)}")

        # 3. 创建特征提取器（参数从配置读取，保留 kwargs 覆盖能力）
        extractor = OVOCspFeatureExtractor(
            csp_n_components    = cfg.get('csp_n_components', 4),
            csp_reg             = cfg.get('csp_reg', 'ledoit_wolf'),
            log_transform       = cfg.get('log_transform', True),
            normalize_features  = cfg.get('normalize_features', True),
            lda_n_components    = cfg.get('lda_n_components', 3),
            random_state        = cfg.get('random_state', 42),
        )

        # 4. 训练并提取特征
        if self.verbose:
            print("训练 CSP 提取器并提取特征...")
        features = extractor.fit_transform(X, y, verbose=False)

        if self.verbose:
            print(f"✓ 特征提取完成，形状: {features.shape}")

        # 5. 保存特征矩阵
        if self.save_features:
            out_dir = os.path.join(get_dataset_dir(self.dataset_name), 'model', 'feature')
            ensure_dir(out_dir)
            feat_file = os.path.join(out_dir, f'{subject_id}{session}_ovocsp_features.npz')
            np.savez(feat_file, features=features, labels=y)
            if self.verbose:
                print(f"✓ 特征+标签已保存至: {feat_file}")

        # 6. 保存特征提取器
        if self.save_extractor:
            out_dir = os.path.join(get_dataset_dir(self.dataset_name), 'model', 'feature')
            ensure_dir(out_dir)
            ext_file = os.path.join(out_dir, f'{subject_id}{session}_ovocsp_extractor.joblib')
            extractor.save(ext_file)
            if self.verbose:
                print(f"✓ 提取器已保存至: {ext_file}")

        if self.verbose:
            print("✓ 特征提取流程结束！")

        return features, extractor