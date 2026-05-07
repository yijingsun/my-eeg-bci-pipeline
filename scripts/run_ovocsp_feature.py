#!/usr/bin/env python3
"""
运行 OVO-CSP 特征提取脚本
从预处理好的 epochs 中提取特征，保存特征矩阵（.npy）和特征提取器（.joblib）
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mne
import numpy as np
import joblib
from config import get_epoch_path, get_dataset_dir, ensure_dir
from src.utils.config_loader import load_session_config
from src.feature_extraction.ovocsp_feature_extractor import OVOCspFeatureExtractor


# ======================= 配置 =======================
DATASET   = 'BCICIV_2a'   # 数据集名称
SUBJECT   = 'A01'         # 被试编号
SESSION   = 'T'           # 会话标识 (T=训练, E=测试)

SAVE_FEATURES   = True    # 是否保存特征矩阵
SAVE_EXTRACTOR  = True    # 是否保存提取器模型

# 特征输出目录 ( data/<数据集>/model/feature/ )
def get_feature_dir(dataset_name):
    return os.path.join(get_dataset_dir(dataset_name), 'model', 'feature')


# ======================= 主流程 =======================
def execute():
    # 1. 加载被试/会话的合并配置 (default + 手动覆盖)
    cfg = load_session_config(DATASET, SUBJECT, SESSION)

    # 2. 读取预处理好的 epochs
    epoch_path = get_epoch_path(DATASET, SUBJECT, SESSION)
    print(f"加载 epochs: {epoch_path}")
    epochs = mne.read_epochs(epoch_path, preload=True)
    X = epochs.get_data()               # (n_trials, n_channels, n_times)
    y = epochs.events[:, 2]             # 标签已映射为 1～4
    print(f"数据形状: {X.shape}, 类别: {np.unique(y)}")

    # 3. 创建特征提取器 (参数从配置读取，亦可临时覆盖)
    extractor = OVOCspFeatureExtractor(
        csp_n_components    = cfg.get('csp_n_components', 4),
        csp_reg             = cfg.get('csp_reg', 'ledoit_wolf'),
        log_transform       = cfg.get('log_transform', True),
        normalize_features  = cfg.get('normalize_features', True),
        lda_n_components    = cfg.get('lda_n_components', 3),
        random_state        = cfg.get('random_state', 42),
    )

    # 4. 训练并提取特征
    print("开始训练 CSP 提取器并提取特征...")
    features = extractor.fit_transform(X, y, verbose=True)
    print(f"✓ 特征提取完成，形状: {features.shape}")
    print(f"      前五个trials的特征: {features[:5]}")

    # 5. 保存特征矩阵
    if SAVE_FEATURES:
        out_dir = get_feature_dir(DATASET)
        ensure_dir(out_dir)
        feat_file = os.path.join(out_dir, f'{SUBJECT}{SESSION}_ovocsp_features.npy')
        np.save(feat_file, features)
        print(f"✓ 特征已保存至: {feat_file}")

    # 6. 保存特征提取器模型
    if SAVE_EXTRACTOR:
        out_dir = get_feature_dir(DATASET)
        ensure_dir(out_dir)
        ext_file = os.path.join(out_dir, f'{SUBJECT}{SESSION}_ovocsp_extractor.joblib')
        extractor.save(ext_file)   # OVOCspFeatureExtractor 已实现 save 方法
        print(f"✓ 提取器已保存至: {ext_file}")

    print("✓ 特征提取流程结束！")
    # print(features)
    # print(extractor)


if __name__ == '__main__':
    execute()