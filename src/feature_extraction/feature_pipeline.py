"""
特征提取管道：加载 epochs → 训练 CSP → 保存特征和提取器
"""
import os
import numpy as np
import mne
from config import get_epoch_path, get_label_path, ensure_dir, get_feature_dir
from src.utils.session_config import SessionConfig
from src.feature_extraction.ovocsp_feature_extractor import OVOCspFeatureExtractor


class FeatureExtractionPipeline:
    """OVO-CSP 特征提取流水线"""

    def __init__(self, dataset_name: str, subject_id: str, session: str):
        if dataset_name is None or subject_id is None or session is None:
            raise ValueError("dataset_name, subject_id, session 不能为空")
        self.dataset_name = dataset_name
        self.subject_id = subject_id
        self.session = session
        # 读取配置
        self.cfg = SessionConfig.from_dataset(dataset_name, subject_id, session)

    def run(self, save_features: bool = True, save_extractor: bool = True, verbose: bool = True):
        """
        执行特征提取：读取 epochs → 训练 CSP → 保存 feature `.npy`, label `.npy` 和 `.joblib`
        """
        # 1. 读取预处理好的 epochs 和 labels
        epoch_path = get_epoch_path(self.dataset_name, self.subject_id, self.session)
        if verbose:
            print(f"加载 epochs: {epoch_path}")
        epochs = mne.read_epochs(epoch_path, preload=True, verbose=False)
        X = epochs.get_data()
        label_path = get_label_path(self.dataset_name, self.subject_id, self.session)
        y = np.load(label_path)

        if verbose:
            print(f"数据形状: {X.shape}, 类别: {np.unique(y)}")

        # 3. 创建特征提取器（参数从配置读取）
        extractor = OVOCspFeatureExtractor(
            csp_n_components    = self.cfg.get('csp_n_components'),
            csp_reg             = self.cfg.get('csp_reg'),
            log_transform       = self.cfg.get('log_transform'),
            normalize_features  = self.cfg.get('normalize_features'),
            lda_n_components    = self.cfg.get('lda_n_components')
        )

        # 4. 训练并提取特征
        if verbose:
            print("训练 CSP 提取器并提取特征...")
        features = extractor.fit_transform(X, y, verbose=False)

        if verbose:
            print(f"✓ 特征提取完成，形状: {features.shape}")

        # 5. 保存特征矩阵
        if save_features:
            out_dir = get_feature_dir(self.dataset_name)
            ensure_dir(out_dir)
            feat_file = os.path.join(out_dir, f'{self.subject_id}{self.session}_ovocsp_features.npy')
            np.save(feat_file, features)
            if verbose:
                print(f"✓ 特征已保存至: {feat_file}")

        # 6. 保存特征提取器
        if save_extractor:
            out_dir = get_feature_dir(self.dataset_name)
            ensure_dir(out_dir)
            ext_file = os.path.join(out_dir, f'{self.subject_id}{self.session}_ovocsp_extractor.joblib')
            extractor.save(ext_file)
            if verbose:
                print(f"✓ 提取器已保存至: {ext_file}")

        if verbose:
            print("✓ 特征提取流程结束！")

        return features, extractor