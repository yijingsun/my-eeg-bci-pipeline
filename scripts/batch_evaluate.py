#!/usr/bin/env python3
"""
批量测试集评估脚本
- 对每个被试的 E 数据进行预处理（若 epochs 文件已存在则跳过）
- 加载 T 数据训练好的特征提取器与分类器
- 预测并计算与官方真实标签的 Kappa 和 Accuracy
- 汇总所有被试的结果
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import mne
from sklearn.metrics import cohen_kappa_score, accuracy_score
from config import (
    get_classifier_path, get_epoch_path, get_extractor_path, get_label_path
)
from src.pipeline.preprocess_pipeline import DataPipeline
from src.feature_extraction.ovocsp_feature_extractor import OVOCspFeatureExtractor
from src.classification.bayesian_classifier import BayesianClassifier

DATASET = 'BCICIV_2a'
SUBJECT_IDS = ['A01', 'A02', 'A03', 'A04', 'A05', 'A06', 'A07', 'A08', 'A09']
SESSION = 'E'

# 是否强制重新预处理（若已存在 epochs 文件且想复用，设为 False）
FORCE_PREPROCESS = False

# 存储每位被试的指标
results = {}

print("=" * 60)
print("测试集批量评估")
print("=" * 60)

for subject in SUBJECT_IDS:
    print(f"\n>>> 处理被试: {subject}E")
    try:
        # 1. 预处理测试数据（若 epochs 不存在则运行）
        epoch_path = get_epoch_path(DATASET, subject, SESSION)
        if not os.path.exists(epoch_path) or FORCE_PREPROCESS:
            print("  运行预处理管道...")
            pipeline = DataPipeline(dataset_name=DATASET, subject_id=subject, session=SESSION)
            pipeline.cfg['mi_event_mapping'] = {"768": 6}
            # pipeline.cfg['tmin'] = 3.0 # 768事件比cue事件早2s
            # pipeline.cfg['tmax'] = 6.0
            pipeline.run(verbose=True, save_label=False)
        else:
            print("  复用已存在的 epochs 文件")

        # 2. 加载测试 epochs
        epochs = mne.read_epochs(epoch_path, preload=True)
        X_test = epochs.get_data()
        print(f"  数据形状: {X_test.shape}")

        # 3. 加载训练好的特征提取器（从 T 训练）
        extractor_path = get_extractor_path(DATASET, subject, 'T', 'ovocsp')
        if not os.path.exists(extractor_path):
            raise FileNotFoundError(f"特征提取器不存在: {extractor_path}")
        extractor = OVOCspFeatureExtractor.load(extractor_path)

        # 4. 加载训练好的分类器（从 T 训练）
        clf_path = get_classifier_path(DATASET, subject, 'T', 'bayesian')
        if not os.path.exists(clf_path):
            raise FileNotFoundError(f"分类器不存在: {clf_path}")
        clf = BayesianClassifier.load(clf_path)

        # 5. 特征提取与预测
        features_test = extractor.transform(X_test)
        y_pred = clf.predict(features_test).flatten().astype(int)

        # 6. 加载官方真实标签
        true_labels = np.load(get_label_path(DATASET, subject, SESSION))

        # 7. 计算指标
        kappa = cohen_kappa_score(true_labels, y_pred)
        acc = accuracy_score(true_labels, y_pred)

        results[subject] = {'kappa': kappa, 'accuracy': acc}
        print(f"  ✓ Kappa: {kappa:.4f}, Accuracy: {acc:.4f}")

    except Exception as e:
        print(f"  ✗ 处理失败: {e}")
        results[subject] = None

# 汇总
valid_results = {k: v for k, v in results.items() if v is not None}
if valid_results:
    kappas = [v['kappa'] for v in valid_results.values()]
    accs = [v['accuracy'] for v in valid_results.values()]
    print("\n" + "=" * 60)
    print("测试集评估汇总")
    print("=" * 60)
    print(f"成功被试数: {len(valid_results)}")
    print(f"平均 Kappa  : {np.mean(kappas):.4f} ± {np.std(kappas):.4f}")
    print(f"平均 Accuracy: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print("\n各被试详情:")
    for subj, val in results.items():
        if val:
            print(f"  {subj}: Kappa={val['kappa']:.4f}, Acc={val['accuracy']:.4f}")
        else:
            print(f"  {subj}: 失败")
else:
    print("\n没有成功评估的被试。")

print("\n评估完成。")

# ============================================================
# 测试集评估汇总
# ============================================================
# 成功被试数: 9
# 平均 Kappa  : 0.3591 ± 0.2800
# 平均 Accuracy: 0.5193 ± 0.2100

# # 各被试详情:
#   A01: Kappa=0.6574, Acc=0.7431
#   A02: Kappa=0.4444, Acc=0.5833
#   A03: Kappa=0.6296, Acc=0.7222
#   A04: Kappa=0.4769, Acc=0.6076
#   A05: Kappa=-0.0602, Acc=0.2049
#   A06: Kappa=0.1620, Acc=0.3715
#   A07: Kappa=-0.1343, Acc=0.1493
#   A08: Kappa=0.4676, Acc=0.6007
#   A09: Kappa=0.5880, Acc=0.6910

# 评估完成。

# 0.52    0.69 	0.34 	0.71 	0.44 	0.16 	0.21 	0.66 	0.73 	0.69

# ============================================================
# 测试集评估汇总
# ============================================================
# 成功被试数: 9
# 平均 Kappa  : 0.4583 ± 0.1884
# 平均 Accuracy: 0.5938 ± 0.1413

# 各被试详情:
#   A01: Kappa=0.6574, Acc=0.7431
#   A02: Kappa=0.4954, Acc=0.6215
#   A03: Kappa=0.6343, Acc=0.7257
#   A04: Kappa=0.3287, Acc=0.4965
#   A05: Kappa=0.1528, Acc=0.3646 < 0.25
#   A06: Kappa=0.1620, Acc=0.3715 < 0.25
#   A07: Kappa=0.6389, Acc=0.7292
#   A08: Kappa=0.4676, Acc=0.6007
#   A09: Kappa=0.5880, Acc=0.6910

# 评估完成。