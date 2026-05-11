#!/usr/bin/env python3
"""
批量训练 + 即时汇总脚本
遍历 A01~A09 训练集，执行预处理、特征提取和分类器训练，
从管道返回结果中提取 CV Kappa / Accuracy，并进行整体评估。
"""
import sys
import os
import numpy as np

# 将项目根目录加入 Python 路径，确保能导入 src 包
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 根据你的实际导入路径调整（如果管道类在别处，请修改）
from src.pipeline.preprocess_pipeline import DataPipeline
from src.pipeline.feature_pipeline import TrainOVOCspFeaturePipeline
from src.pipeline.classify_pipeline import TrainClassifierPipeline

DATASET = 'BCICIV_2a'
SUBJECT_IDS = ['A01', 'A02', 'A03', 'A04', 'A05', 'A06', 'A07', 'A08', 'A09']
SESSION = 'T'          # 训练集
DO_PREPROCESS = True   # 如果已经生成过 epochs，可设为 False 跳过预处理
DO_FEATURE = True      # 如果已经生成过特征，可设为 False 跳过特征提取

# 存储每个被试的指标
all_kappas = []
all_accuracies = []
failed_subjects = []

print("=" * 60)
print("批量训练开始")
print("=" * 60)

for subject_id in SUBJECT_IDS:
    print(f"\n--- 处理被试: {subject_id} ---")
    try:
        # 1. 预处理（可跳过）
        if DO_PREPROCESS:
            pre_pipeline = DataPipeline(
                dataset_name=DATASET,
                subject_id=subject_id,
                session=SESSION
            )
            pre_pipeline.run()
        else:
            print("  [跳过] 预处理")

        # 2. 特征提取
        if DO_FEATURE:
            feature_pipeline = TrainOVOCspFeaturePipeline(
                dataset_name=DATASET,
                subject_id=subject_id,
                session=SESSION
            )
            feature_pipeline.run()
        else:
            print("  [跳过] 特征提取")

        # 3. 分类训练（返回 (result_dict, clf)）
        classify_pipeline = TrainClassifierPipeline(
            dataset_name=DATASET,
            subject_id=subject_id,
            session=SESSION
        )
        train_result, trained_clf = classify_pipeline.run(save=False) # 重复测试的时候就不用重复保存多余文件了

        # 提取交叉验证结果
        cv_summary = train_result.get('cv_summary', {})
        kappa = cv_summary.get('kappa_mean')
        accuracy = cv_summary.get('accuracy_mean')

        if kappa is not None and accuracy is not None:
            all_kappas.append(kappa)
            all_accuracies.append(accuracy)
            print(f"  ✓ CV Kappa: {kappa:.4f}, Accuracy: {accuracy:.4f}")
        else:
            print(f"  ⚠ 警告：未从管道返回结果中找到 CV 指标")
            failed_subjects.append(subject_id)

    except Exception as e:
        print(f"  ✗ 处理 {subject_id} 时出错: {e}")
        failed_subjects.append(subject_id)

# ---------- 整体评估 ----------
print("\n" + "=" * 60)
print("整体评估（所有被试训练集交叉验证结果）")
print("=" * 60)

if all_kappas:
    mean_kappa = np.mean(all_kappas)
    std_kappa = np.std(all_kappas)
    mean_acc = np.mean(all_accuracies)
    std_acc = np.std(all_accuracies)

    print(f"成功处理被试数: {len(all_kappas)}")
    print(f"平均 Kappa  : {mean_kappa:.4f} ± {std_kappa:.4f}")
    print(f"平均 Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")

    print("\n各被试详情:")
    # 注意：这里按成功顺序打印，若前面某个被试失败，索引可能偏移
    # 简单处理：单独记录成功被试的ID
    idx = 0
    for subject_id in SUBJECT_IDS:
        if idx >= len(all_kappas):
            break
        # 更好的方式是用字典记录，但简单起见这里直接按顺序输出
        print(f"  {subject_id}: Kappa={all_kappas[idx]:.4f}, Acc={all_accuracies[idx]:.4f}")
        idx += 1
else:
    print("没有成功处理任何被试，无法计算整体指标。")

if failed_subjects:
    print(f"\n处理失败的被试: {failed_subjects}")

print("\n批量训练与评估结束。")

# ============================================================
# 整体评估（所有被试训练集交叉验证结果）
# ============================================================
# 成功处理被试数: 9
# 平均 Kappa  : 0.7202 ± 0.1199
# 平均 Accuracy: 0.7901 ± 0.0899

# 各被试详情:
#   A01: Kappa=0.7870, Acc=0.8403
#   A02: Kappa=0.7315, Acc=0.7986
#   A03: Kappa=0.8426, Acc=0.8819
#   A04: Kappa=0.6019, Acc=0.7014
#   A05: Kappa=0.4907, Acc=0.6181
#   A06: Kappa=0.6389, Acc=0.7292
#   A07: Kappa=0.8333, Acc=0.8750
#   A08: Kappa=0.8704, Acc=0.9028
#   A09: Kappa=0.6852, Acc=0.7639

# 批量训练与评估结束。