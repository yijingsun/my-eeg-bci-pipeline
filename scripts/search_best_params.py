#!/usr/bin/env python3
"""
特征提取 + 分类器参数搜索脚本（固定预处理）
遍历 CSP 成分数、正则化、标准化、降维、分类器类型等，找出最高 Kappa 组合。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import itertools
import numpy as np
import mne
from config import get_epoch_path
from src.feature_extraction.ovocsp_feature_extractor import OVOCspFeatureExtractor
from src.classification.bayesian_classifier import BayesianClassifier
from src.evaluation.evaluator import BCIEvaluator

# ================= 固定设置 =================
DATASET = 'BCICIV_2a'
SUBJECT = 'A04'
SESSION = 'T'
CV_FOLDS = 4
RANDOM_STATE = 17

# ================= 1. 加载预处理好的 epochs =================
epoch_path = get_epoch_path(DATASET, SUBJECT, SESSION)
epochs = mne.read_epochs(epoch_path, preload=True)
X = epochs.get_data()          # (288, 22, 751)
y = epochs.events[:, 2]        # 标签 1~4

# ================= 2. 定义搜索网格（仅特征提取 + 分类器） =================
param_grid = {
    'csp_n_components': [2, 4, 6],
    'csp_reg': [None, 'ledoit_wolf', 'shrinkage'],
    'normalize_features': [True, False],
    'lda_n_components': [None, 3],
    'log_transform': [True, False],
    'classifier': ['Bayesian', 'SVM'],
}

# ================= 3. 辅助函数 =================
def create_clf(cls_name, random_state=RANDOM_STATE):
    if cls_name == 'Bayesian':
        return BayesianClassifier()
    elif cls_name == 'SVM':
        from sklearn.svm import SVC
        return SVC(kernel='rbf', random_state=random_state)
    else:
        raise ValueError(f'Unknown classifier: {cls_name}')

def evaluate_one(params, X, y):
    extractor = OVOCspFeatureExtractor(
        csp_n_components    = params['csp_n_components'],
        csp_reg             = params['csp_reg'],
        log_transform       = params['log_transform'],
        normalize_features  = params['normalize_features'],
        lda_n_components    = params['lda_n_components'],
    )
    features = extractor.fit_transform(X, y, verbose=False)

    clf = create_clf(params['classifier'], random_state=RANDOM_STATE)
    evaluator = BCIEvaluator(cv_folds=CV_FOLDS, random_state=RANDOM_STATE)
    result = evaluator.evaluate(features, y, clf)
    return result['kappa_mean'], result['accuracy_mean'], result['kappa_std']

# ================= 4. 网格搜索 =================
keys = list(param_grid.keys())
combinations = list(itertools.product(*param_grid.values()))
total = len(combinations)
print(f"开始特征提取参数搜索，共 {total} 个组合...\n")

best_kappa = -1.0
best_params = None
best_accuracy = 0.0

for idx, combo in enumerate(combinations, 1):
    params = dict(zip(keys, combo))
    try:
        kappa, acc, kappa_std = evaluate_one(params, X, y)
        print(f"[{idx:3d}/{total}] {params} -> Kappa={kappa:.4f} (±{kappa_std:.4f}), Acc={acc:.4f}")
        if kappa > best_kappa:
            best_kappa = kappa
            best_params = params
            best_accuracy = acc
    except Exception as e:
        print(f"[{idx:3d}/{total}] {params} 出错: {e}")

print("\n" + "="*60)
print(f"{SUBJECT}{SESSION}")
print("最优参数组合 (基于交叉验证 Kappa):")
print(f"  Kappa     = {best_kappa:.4f}")
print(f"  Accuracy  = {best_accuracy:.4f}")
print(f"  参数      = {best_params}")
print("="*60)

# [119/144] {'csp_n_components': 6, 'csp_reg': 'ledoit_wolf', 'normalize_features': True, 'lda_n_components': 3, 'log_transform': False, 'classifier': 'Bayesian'} -> Kappa=0.7870 (±0.0278), Acc=0.8403
# [120/144] {'csp_n_components': 6, 'csp_reg': 'ledoit_wolf', 'normalize_features': True, 'lda_n_components': 3, 'log_transform': False, 'classifier': 'SVM'} -> Kappa=0.7917 (±0.0304), Acc=0.8438

# 搜索全部完成！最优参数组合是 第 120 组 (和第 119 组贝叶斯结果几乎一样):

# 该组合在贝叶斯分类器下也达到了 0.7870 (组合 119)，两者非常接近。

# 统计 144 组实验的关键规律

# 1. log_transform 是决定性因素
#    · log_transform = True 的所有组合 Kappa 几乎都低于 0.5，最高仅 0.5093。
#    · log_transform = False 的组合普遍在 0.57 以上，最优区在 0.75~0.79。
#      结论： 在当前数据与预处理下，不取对数明显更好。
# 2. CSP 成分数越多越好
#    · 成分数 2 → 最高 Kappa 0.71
#    · 成分数 4 → 最高 0.7454
#    · 成分数 6 → 最高 0.7917
#      增加成分提取了更多判别信息，且未引起明显过拟合。
# 3. 正则化 ledoit_wolf 与 None 相近，shrinkage 较差
#    · ledoit_wolf 和 None 在最优配置下差异很小（例如 0.7917 vs 0.7685）。
#    · shrinkage 几乎在所有组合中表现最差（最高仅 0.75），不建议使用。
# 4. LDA 降维到 3 优于不降维
#    · 无论成分数是 4 还是 6，lda_n_components=3 的 Kappa 平均高于 None（48 维）。
#    · 降维到 3 维有助于去除噪声，且配合贝叶斯/SVM 效果更好。
# 5. 标准化对结果影响不大，但开启更稳定
#    · normalize_features=True 与 False 在相同条件下 Kappa 差异多在 ±0.01 以内。
#    · 打开标准化略微提高稳定性，可以保持。
# 6. 分类器：SVM 略胜贝叶斯，但差距很小
#    · 在最优参数下，SVM 比贝叶斯高约 0.004；在多数其他组合中，两者交替领先。

# 如果项目要求可解释性更强，可以选择贝叶斯；
# 若追求指标极限，用 SVM。两者在你的数据上都能达到 Kappa ≈ 0.79 的优秀水平。
# 接下来你可以将此配置推广到其他被试，或在测试集上评估。

# 开始特征提取参数搜索，共 144 个组合...

# [  1/144] {'csp_n_components': 2, 'csp_reg': None, 'normalize_features': True, 'lda_n_components': None, 'log_transform': True, 'classifier': 'Bayesian'} -> Kappa=0.4861 (±0.0154), Acc=0.6146
# [  2/144] {'csp_n_components': 2, 'csp_reg': None, 'normalize_features': True, 'lda_n_components': None, 'log_transform': True, 'classifier': 'SVM'} -> Kappa=0.4769 (±0.0274), Acc=0.6076
# [  3/144] {'csp_n_components': 2, 'csp_reg': None, 'normalize_features': True, 'lda_n_components': None, 'log_transform': False, 'classifier': 'Bayesian'} -> Kappa=0.5880 (±0.0304), Acc=0.6910
# [  4/144] {'csp_n_components': 2, 'csp_reg': None, 'normalize_features': True, 'lda_n_components': None, 'log_transform': False, 'classifier': 'SVM'} -> Kappa=0.5648 (±0.0404), Acc=0.6736
# [  5/144] {'csp_n_components': 2, 'csp_reg': None, 'normalize_features': True, 'lda_n_components': 3, 'log_transform': True, 'classifier': 'Bayesian'} -> Kappa=0.4722 (±0.0424), Acc=0.6042
# [  6/144] {'csp_n_components': 2, 'csp_reg': None, 'normalize_features': True, 'lda_n_components': 3, 'log_transform': True, 'classifier': 'SVM'} -> Kappa=0.4676 (±0.0241), Acc=0.6007
# [  7/144] {'csp_n_components': 2, 'csp_reg': None, 'normalize_features': True, 'lda_n_components': 3, 'log_transform': False, 'classifier': 'Bayesian'} -> Kappa=0.5880 (±0.0605), Acc=0.6910
# [  8/144] {'csp_n_components': 2, 'csp_reg': None, 'normalize_features': True, 'lda_n_components': 3, 'log_transform': False, 'classifier': 'SVM'} -> Kappa=0.5880 (±0.0745), Acc=0.6910
# [  9/144] {'csp_n_components': 2, 'csp_reg': None, 'normalize_features': False, 'lda_n_components': None, 'log_transform': True, 'classifier': 'Bayesian'} -> Kappa=0.4861 (±0.0154), Acc=0.6146
# [ 10/144] {'csp_n_components': 2, 'csp_reg': None, 'normalize_features': False, 'lda_n_components': None, 'log_transform': True, 'classifier': 'SVM'} -> Kappa=0.4537 (±0.0160), Acc=0.5903
# [ 11/144] {'csp_n_components': 2, 'csp_reg': None, 'normalize_features': False, 'lda_n_components': None, 'log_transform': False, 'classifier': 'Bayesian'} -> Kappa=0.5880 (±0.0304), Acc=0.6910
# [ 12/144] {'csp_n_components': 2, 'csp_reg': None, 'normalize_features': False, 'lda_n_components': None, 'log_transform': False, 'classifier': 'SVM'} -> Kappa=0.5463 (±0.0382), Acc=0.6597
# [ 13/144] {'csp_n_components': 2, 'csp_reg': None, 'normalize_features': False, 'lda_n_components': 3, 'log_transform': True, 'classifier': 'Bayesian'} -> Kappa=0.4259 (±0.0262), Acc=0.5694
# [ 14/144] {'csp_n_components': 2, 'csp_reg': None, 'normalize_features': False, 'lda_n_components': 3, 'log_transform': True, 'classifier': 'SVM'} -> Kappa=0.4444 (±0.0131), Acc=0.5833
# [ 15/144] {'csp_n_components': 2, 'csp_reg': None, 'normalize_features': False, 'lda_n_components': 3, 'log_transform': False, 'classifier': 'Bayesian'} -> Kappa=0.5880 (±0.0605), Acc=0.6910
# [ 16/144] {'csp_n_components': 2, 'csp_reg': None, 'normalize_features': False, 'lda_n_components': 3, 'log_transform': False, 'classifier': 'SVM'} -> Kappa=0.5602 (±0.0633), Acc=0.6701
# [ 17/144] {'csp_n_components': 2, 'csp_reg': 'ledoit_wolf', 'normalize_features': True, 'lda_n_components': None, 'log_transform': True, 'classifier': 'Bayesian'} -> Kappa=0.4722 (±0.0207), Acc=0.6042
# [ 18/144] {'csp_n_components': 2, 'csp_reg': 'ledoit_wolf', 'normalize_features': True, 'lda_n_components': None, 'log_transform': True, 'classifier': 'SVM'} -> Kappa=0.5046 (±0.0331), Acc=0.6285
# [ 19/144] {'csp_n_components': 2, 'csp_reg': 'ledoit_wolf', 'normalize_features': True, 'lda_n_components': None, 'log_transform': False, 'classifier': 'Bayesian'} -> Kappa=0.6296 (±0.0454), Acc=0.7222
# [ 20/144] {'csp_n_components': 2, 'csp_reg': 'ledoit_wolf', 'normalize_features': True, 'lda_n_components': None, 'log_transform': False, 'classifier': 'SVM'} -> Kappa=0.6435 (±0.0530), Acc=0.7326
# [ 21/144] {'csp_n_components': 2, 'csp_reg': 'ledoit_wolf', 'normalize_features': True, 'lda_n_components': 3, 'log_transform': True, 'classifier': 'Bayesian'} -> Kappa=0.4352 (±0.0207), Acc=0.5764
# [ 22/144] {'csp_n_components': 2, 'csp_reg': 'ledoit_wolf', 'normalize_features': True, 'lda_n_components': 3, 'log_transform': True, 'classifier': 'SVM'} -> Kappa=0.4954 (±0.0241), Acc=0.6215
# [ 23/144] {'csp_n_components': 2, 'csp_reg': 'ledoit_wolf', 'normalize_features': True, 'lda_n_components': 3, 'log_transform': False, 'classifier': 'Bayesian'} -> Kappa=0.6667 (±0.0131), Acc=0.7500
# [ 24/144] {'csp_n_components': 2, 'csp_reg': 'ledoit_wolf', 'normalize_features': True, 'lda_n_components': 3, 'log_transform': False, 'classifier': 'SVM'} -> Kappa=0.7083 (±0.0356), Acc=0.7812
# [ 25/144] {'csp_n_components': 2, 'csp_reg': 'ledoit_wolf', 'normalize_features': False, 'lda_n_components': None, 'log_transform': True, 'classifier': 'Bayesian'} -> Kappa=0.4722 (±0.0207), Acc=0.6042
# [ 26/144] {'csp_n_components': 2, 'csp_reg': 'ledoit_wolf', 'normalize_features': False, 'lda_n_components': None, 'log_transform': True, 'classifier': 'SVM'} -> Kappa=0.4815 (±0.0346), Acc=0.6111
# [ 27/144] {'csp_n_components': 2, 'csp_reg': 'ledoit_wolf', 'normalize_features': False, 'lda_n_components': None, 'log_transform': False, 'classifier': 'Bayesian'} -> Kappa=0.6296 (±0.0454), Acc=0.7222
# [ 28/144] {'csp_n_components': 2, 'csp_reg': 'ledoit_wolf', 'normalize_features': False, 'lda_n_components': None, 'log_transform': False, 'classifier': 'SVM'} -> Kappa=0.5833 (±0.0404), Acc=0.6875
# [ 29/144] {'csp_n_components': 2, 'csp_reg': 'ledoit_wolf', 'normalize_features': False, 'lda_n_components': 3, 'log_transform': True, 'classifier': 'Bayesian'} -> Kappa=0.4352 (±0.0382), Acc=0.5764
# [ 30/144] {'csp_n_components': 2, 'csp_reg': 'ledoit_wolf', 'normalize_features': False, 'lda_n_components': 3, 'log_transform': True, 'classifier': 'SVM'} -> Kappa=0.4444 (±0.0185), Acc=0.5833
# [ 31/144] {'csp_n_components': 2, 'csp_reg': 'ledoit_wolf', 'normalize_features': False, 'lda_n_components': 3, 'log_transform': False, 'classifier': 'Bayesian'} -> Kappa=0.6667 (±0.0131), Acc=0.7500
# [ 32/144] {'csp_n_components': 2, 'csp_reg': 'ledoit_wolf', 'normalize_features': False, 'lda_n_components': 3, 'log_transform': False, 'classifier': 'SVM'} -> Kappa=0.6157 (±0.0202), Acc=0.7118
# [ 33/144] {'csp_n_components': 2, 'csp_reg': 'shrinkage', 'normalize_features': True, 'lda_n_components': None, 'log_transform': True, 'classifier': 'Bayesian'} -> Kappa=0.3333 (±0.0414), Acc=0.5000
# [ 34/144] {'csp_n_components': 2, 'csp_reg': 'shrinkage', 'normalize_features': True, 'lda_n_components': None, 'log_transform': True, 'classifier': 'SVM'} -> Kappa=0.3333 (±0.0293), Acc=0.5000
# [ 35/144] {'csp_n_components': 2, 'csp_reg': 'shrinkage', 'normalize_features': True, 'lda_n_components': None, 'log_transform': False, 'classifier': 'Bayesian'} -> Kappa=0.6250 (±0.0241), Acc=0.7188
# [ 36/144] {'csp_n_components': 2, 'csp_reg': 'shrinkage', 'normalize_features': True, 'lda_n_components': None, 'log_transform': False, 'classifier': 'SVM'} -> Kappa=0.6111 (±0.0434), Acc=0.7083
# [ 37/144] {'csp_n_components': 2, 'csp_reg': 'shrinkage', 'normalize_features': True, 'lda_n_components': 3, 'log_transform': True, 'classifier': 'Bayesian'} -> Kappa=0.3472 (±0.0274), Acc=0.5104
# [ 38/144] {'csp_n_components': 2, 'csp_reg': 'shrinkage', 'normalize_features': True, 'lda_n_components': 3, 'log_transform': True, 'classifier': 'SVM'} -> Kappa=0.3009 (±0.0080), Acc=0.4757
# [ 39/144] {'csp_n_components': 2, 'csp_reg': 'shrinkage', 'normalize_features': True, 'lda_n_components': 3, 'log_transform': False, 'classifier': 'Bayesian'} -> Kappa=0.6204 (±0.0499), Acc=0.7153
# [ 40/144] {'csp_n_components': 2, 'csp_reg': 'shrinkage', 'normalize_features': True, 'lda_n_components': 3, 'log_transform': False, 'classifier': 'SVM'} -> Kappa=0.6991 (±0.0496), Acc=0.7743
# [ 41/144] {'csp_n_components': 2, 'csp_reg': 'shrinkage', 'normalize_features': False, 'lda_n_components': None, 'log_transform': True, 'classifier': 'Bayesian'} -> Kappa=0.3333 (±0.0414), Acc=0.5000
# [ 42/144] {'csp_n_components': 2, 'csp_reg': 'shrinkage', 'normalize_features': False, 'lda_n_components': None, 'log_transform': True, 'classifier': 'SVM'} -> Kappa=0.3241 (±0.0334), Acc=0.4931
# [ 43/144] {'csp_n_components': 2, 'csp_reg': 'shrinkage', 'normalize_features': False, 'lda_n_components': None, 'log_transform': False, 'classifier': 'Bayesian'} -> Kappa=0.6250 (±0.0241), Acc=0.7188
# [ 44/144] {'csp_n_components': 2, 'csp_reg': 'shrinkage', 'normalize_features': False, 'lda_n_components': None, 'log_transform': False, 'classifier': 'SVM'} -> Kappa=0.6157 (±0.0356), Acc=0.7118
# [ 45/144] {'csp_n_components': 2, 'csp_reg': 'shrinkage', 'normalize_features': False, 'lda_n_components': 3, 'log_transform': True, 'classifier': 'Bayesian'} -> Kappa=0.3380 (±0.0154), Acc=0.5035
# [ 46/144] {'csp_n_components': 2, 'csp_reg': 'shrinkage', 'normalize_features': False, 'lda_n_components': 3, 'log_transform': True, 'classifier': 'SVM'} -> Kappa=0.3148 (±0.0185), Acc=0.4861
# [ 47/144] {'csp_n_components': 2, 'csp_reg': 'shrinkage', 'normalize_features': False, 'lda_n_components': 3, 'log_transform': False, 'classifier': 'Bayesian'} -> Kappa=0.6204 (±0.0499), Acc=0.7153
# [ 48/144] {'csp_n_components': 2, 'csp_reg': 'shrinkage', 'normalize_features': False, 'lda_n_components': 3, 'log_transform': False, 'classifier': 'SVM'} -> Kappa=0.6296 (±0.0293), Acc=0.7222
# [ 49/144] {'csp_n_components': 4, 'csp_reg': None, 'normalize_features': True, 'lda_n_components': None, 'log_transform': True, 'classifier': 'Bayesian'} -> Kappa=0.5093 (±0.0382), Acc=0.6319
# [ 50/144] {'csp_n_components': 4, 'csp_reg': None, 'normalize_features': True, 'lda_n_components': None, 'log_transform': True, 'classifier': 'SVM'} -> Kappa=0.4769 (±0.0331), Acc=0.6076
# [ 51/144] {'csp_n_components': 4, 'csp_reg': None, 'normalize_features': True, 'lda_n_components': None, 'log_transform': False, 'classifier': 'Bayesian'} -> Kappa=0.6759 (±0.0424), Acc=0.7569
# [ 52/144] {'csp_n_components': 4, 'csp_reg': None, 'normalize_features': True, 'lda_n_components': None, 'log_transform': False, 'classifier': 'SVM'} -> Kappa=0.5926 (±0.0472), Acc=0.6944
# [ 53/144] {'csp_n_components': 4, 'csp_reg': None, 'normalize_features': True, 'lda_n_components': 3, 'log_transform': True, 'classifier': 'Bayesian'} -> Kappa=0.3935 (±0.0513), Acc=0.5451
# [ 54/144] {'csp_n_components': 4, 'csp_reg': None, 'normalize_features': True, 'lda_n_components': 3, 'log_transform': True, 'classifier': 'SVM'} -> Kappa=0.3750 (±0.0274), Acc=0.5312
# [ 55/144] {'csp_n_components': 4, 'csp_reg': None, 'normalize_features': True, 'lda_n_components': 3, 'log_transform': False, 'classifier': 'Bayesian'} -> Kappa=0.6620 (±0.0154), Acc=0.7465
# [ 56/144] {'csp_n_components': 4, 'csp_reg': None, 'normalize_features': True, 'lda_n_components': 3, 'log_transform': False, 'classifier': 'SVM'} -> Kappa=0.6806 (±0.0801), Acc=0.7604
# [ 57/144] {'csp_n_components': 4, 'csp_reg': None, 'normalize_features': False, 'lda_n_components': None, 'log_transform': True, 'classifier': 'Bayesian'} -> Kappa=0.5093 (±0.0382), Acc=0.6319
# [ 58/144] {'csp_n_components': 4, 'csp_reg': None, 'normalize_features': False, 'lda_n_components': None, 'log_transform': True, 'classifier': 'SVM'} -> Kappa=0.4722 (±0.0548), Acc=0.6042
# [ 59/144] {'csp_n_components': 4, 'csp_reg': None, 'normalize_features': False, 'lda_n_components': None, 'log_transform': False, 'classifier': 'Bayesian'} -> Kappa=0.6759 (±0.0424), Acc=0.7569
# [ 60/144] {'csp_n_components': 4, 'csp_reg': None, 'normalize_features': False, 'lda_n_components': None, 'log_transform': False, 'classifier': 'SVM'} -> Kappa=0.5741 (±0.0393), Acc=0.6806
# [ 61/144] {'csp_n_components': 4, 'csp_reg': None, 'normalize_features': False, 'lda_n_components': 3, 'log_transform': True, 'classifier': 'Bayesian'} -> Kappa=0.3889 (±0.0556), Acc=0.5417
# [ 62/144] {'csp_n_components': 4, 'csp_reg': None, 'normalize_features': False, 'lda_n_components': 3, 'log_transform': True, 'classifier': 'SVM'} -> Kappa=0.3750 (±0.0241), Acc=0.5312
# [ 63/144] {'csp_n_components': 4, 'csp_reg': None, 'normalize_features': False, 'lda_n_components': 3, 'log_transform': False, 'classifier': 'Bayesian'} -> Kappa=0.6620 (±0.0154), Acc=0.7465
# [ 64/144] {'csp_n_components': 4, 'csp_reg': None, 'normalize_features': False, 'lda_n_components': 3, 'log_transform': False, 'classifier': 'SVM'} -> Kappa=0.6667 (±0.0414), Acc=0.7500
# [ 65/144] {'csp_n_components': 4, 'csp_reg': 'ledoit_wolf', 'normalize_features': True, 'lda_n_components': None, 'log_transform': True, 'classifier': 'Bayesian'} -> Kappa=0.4815 (±0.0628), Acc=0.6111
# [ 66/144] {'csp_n_components': 4, 'csp_reg': 'ledoit_wolf', 'normalize_features': True, 'lda_n_components': None, 'log_transform': True, 'classifier': 'SVM'} -> Kappa=0.4722 (±0.0334), Acc=0.6042
# [ 67/144] {'csp_n_components': 4, 'csp_reg': 'ledoit_wolf', 'normalize_features': True, 'lda_n_components': None, 'log_transform': False, 'classifier': 'Bayesian'} -> Kappa=0.7176 (±0.0461), Acc=0.7882
# [ 68/144] {'csp_n_components': 4, 'csp_reg': 'ledoit_wolf', 'normalize_features': True, 'lda_n_components': None, 'log_transform': False, 'classifier': 'SVM'} -> Kappa=0.6852 (±0.0807), Acc=0.7639
# [ 69/144] {'csp_n_components': 4, 'csp_reg': 'ledoit_wolf', 'normalize_features': True, 'lda_n_components': 3, 'log_transform': True, 'classifier': 'Bayesian'} -> Kappa=0.4861 (±0.0331), Acc=0.6146
# [ 70/144] {'csp_n_components': 4, 'csp_reg': 'ledoit_wolf', 'normalize_features': True, 'lda_n_components': 3, 'log_transform': True, 'classifier': 'SVM'} -> Kappa=0.4815 (±0.0346), Acc=0.6111
# [ 71/144] {'csp_n_components': 4, 'csp_reg': 'ledoit_wolf', 'normalize_features': True, 'lda_n_components': 3, 'log_transform': False, 'classifier': 'Bayesian'} -> Kappa=0.7037 (±0.0293), Acc=0.7778
# [ 72/144] {'csp_n_components': 4, 'csp_reg': 'ledoit_wolf', 'normalize_features': True, 'lda_n_components': 3, 'log_transform': False, 'classifier': 'SVM'} -> Kappa=0.6991 (±0.0756), Acc=0.7743
# [ 73/144] {'csp_n_components': 4, 'csp_reg': 'ledoit_wolf', 'normalize_features': False, 'lda_n_components': None, 'log_transform': True, 'classifier': 'Bayesian'} -> Kappa=0.4815 (±0.0628), Acc=0.6111
# [ 74/144] {'csp_n_components': 4, 'csp_reg': 'ledoit_wolf', 'normalize_features': False, 'lda_n_components': None, 'log_transform': True, 'classifier': 'SVM'} -> Kappa=0.4815 (±0.0507), Acc=0.6111
# [ 75/144] {'csp_n_components': 4, 'csp_reg': 'ledoit_wolf', 'normalize_features': False, 'lda_n_components': None, 'log_transform': False, 'classifier': 'Bayesian'} -> Kappa=0.7176 (±0.0461), Acc=0.7882
# [ 76/144] {'csp_n_components': 4, 'csp_reg': 'ledoit_wolf', 'normalize_features': False, 'lda_n_components': None, 'log_transform': False, 'classifier': 'SVM'} -> Kappa=0.6574 (±0.0593), Acc=0.7431
# [ 77/144] {'csp_n_components': 4, 'csp_reg': 'ledoit_wolf', 'normalize_features': False, 'lda_n_components': 3, 'log_transform': True, 'classifier': 'Bayesian'} -> Kappa=0.3750 (±0.0401), Acc=0.5312
# [ 78/144] {'csp_n_components': 4, 'csp_reg': 'ledoit_wolf', 'normalize_features': False, 'lda_n_components': 3, 'log_transform': True, 'classifier': 'SVM'} -> Kappa=0.4074 (±0.0131), Acc=0.5556
# [ 79/144] {'csp_n_components': 4, 'csp_reg': 'ledoit_wolf', 'normalize_features': False, 'lda_n_components': 3, 'log_transform': False, 'classifier': 'Bayesian'} -> Kappa=0.7037 (±0.0293), Acc=0.7778
# [ 80/144] {'csp_n_components': 4, 'csp_reg': 'ledoit_wolf', 'normalize_features': False, 'lda_n_components': 3, 'log_transform': False, 'classifier': 'SVM'} -> Kappa=0.6944 (±0.0481), Acc=0.7708
# [ 81/144] {'csp_n_components': 4, 'csp_reg': 'shrinkage', 'normalize_features': True, 'lda_n_components': None, 'log_transform': True, 'classifier': 'Bayesian'} -> Kappa=0.3519 (±0.0227), Acc=0.5139
# [ 82/144] {'csp_n_components': 4, 'csp_reg': 'shrinkage', 'normalize_features': True, 'lda_n_components': None, 'log_transform': True, 'classifier': 'SVM'} -> Kappa=0.3102 (±0.0202), Acc=0.4826
# [ 83/144] {'csp_n_components': 4, 'csp_reg': 'shrinkage', 'normalize_features': True, 'lda_n_components': None, 'log_transform': False, 'classifier': 'Bayesian'} -> Kappa=0.6713 (±0.0422), Acc=0.7535
# [ 84/144] {'csp_n_components': 4, 'csp_reg': 'shrinkage', 'normalize_features': True, 'lda_n_components': None, 'log_transform': False, 'classifier': 'SVM'} -> Kappa=0.6528 (±0.0605), Acc=0.7396
# [ 85/144] {'csp_n_components': 4, 'csp_reg': 'shrinkage', 'normalize_features': True, 'lda_n_components': 3, 'log_transform': True, 'classifier': 'Bayesian'} -> Kappa=0.3056 (±0.0160), Acc=0.4792
# [ 86/144] {'csp_n_components': 4, 'csp_reg': 'shrinkage', 'normalize_features': True, 'lda_n_components': 3, 'log_transform': True, 'classifier': 'SVM'} -> Kappa=0.3380 (±0.0080), Acc=0.5035
# [ 87/144] {'csp_n_components': 4, 'csp_reg': 'shrinkage', 'normalize_features': True, 'lda_n_components': 3, 'log_transform': False, 'classifier': 'Bayesian'} -> Kappa=0.6806 (±0.0605), Acc=0.7604
# [ 88/144] {'csp_n_components': 4, 'csp_reg': 'shrinkage', 'normalize_features': True, 'lda_n_components': 3, 'log_transform': False, 'classifier': 'SVM'} -> Kappa=0.7454 (±0.0202), Acc=0.8090
# [ 89/144] {'csp_n_components': 4, 'csp_reg': 'shrinkage', 'normalize_features': False, 'lda_n_components': None, 'log_transform': True, 'classifier': 'Bayesian'} -> Kappa=0.3519 (±0.0227), Acc=0.5139
# [ 90/144] {'csp_n_components': 4, 'csp_reg': 'shrinkage', 'normalize_features': False, 'lda_n_components': None, 'log_transform': True, 'classifier': 'SVM'} -> Kappa=0.3472 (±0.0479), Acc=0.5104
# [ 91/144] {'csp_n_components': 4, 'csp_reg': 'shrinkage', 'normalize_features': False, 'lda_n_components': None, 'log_transform': False, 'classifier': 'Bayesian'} -> Kappa=0.6713 (±0.0422), Acc=0.7535
# [ 92/144] {'csp_n_components': 4, 'csp_reg': 'shrinkage', 'normalize_features': False, 'lda_n_components': None, 'log_transform': False, 'classifier': 'SVM'} -> Kappa=0.6435 (±0.0530), Acc=0.7326
# [ 93/144] {'csp_n_components': 4, 'csp_reg': 'shrinkage', 'normalize_features': False, 'lda_n_components': 3, 'log_transform': True, 'classifier': 'Bayesian'} -> Kappa=0.3056 (±0.0160), Acc=0.4792
# [ 94/144] {'csp_n_components': 4, 'csp_reg': 'shrinkage', 'normalize_features': False, 'lda_n_components': 3, 'log_transform': True, 'classifier': 'SVM'} -> Kappa=0.3380 (±0.0080), Acc=0.5035
# [ 95/144] {'csp_n_components': 4, 'csp_reg': 'shrinkage', 'normalize_features': False, 'lda_n_components': 3, 'log_transform': False, 'classifier': 'Bayesian'} -> Kappa=0.6806 (±0.0605), Acc=0.7604
# [ 96/144] {'csp_n_components': 4, 'csp_reg': 'shrinkage', 'normalize_features': False, 'lda_n_components': 3, 'log_transform': False, 'classifier': 'SVM'} -> Kappa=0.7176 (±0.0241), Acc=0.7882
# [ 97/144] {'csp_n_components': 6, 'csp_reg': None, 'normalize_features': True, 'lda_n_components': None, 'log_transform': True, 'classifier': 'Bayesian'} -> Kappa=0.4676 (±0.0530), Acc=0.6007
# [ 98/144] {'csp_n_components': 6, 'csp_reg': None, 'normalize_features': True, 'lda_n_components': None, 'log_transform': True, 'classifier': 'SVM'} -> Kappa=0.4583 (±0.0605), Acc=0.5938
# [ 99/144] {'csp_n_components': 6, 'csp_reg': None, 'normalize_features': True, 'lda_n_components': None, 'log_transform': False, 'classifier': 'Bayesian'} -> Kappa=0.7593 (±0.0293), Acc=0.8194
# [100/144] {'csp_n_components': 6, 'csp_reg': None, 'normalize_features': True, 'lda_n_components': None, 'log_transform': False, 'classifier': 'SVM'} -> Kappa=0.6852 (±0.0752), Acc=0.7639
# [101/144] {'csp_n_components': 6, 'csp_reg': None, 'normalize_features': True, 'lda_n_components': 3, 'log_transform': True, 'classifier': 'Bayesian'} -> Kappa=0.4167 (±0.0548), Acc=0.5625
# [102/144] {'csp_n_components': 6, 'csp_reg': None, 'normalize_features': True, 'lda_n_components': 3, 'log_transform': True, 'classifier': 'SVM'} -> Kappa=0.3843 (±0.0331), Acc=0.5382
# [103/144] {'csp_n_components': 6, 'csp_reg': None, 'normalize_features': True, 'lda_n_components': 3, 'log_transform': False, 'classifier': 'Bayesian'} -> Kappa=0.7639 (±0.0304), Acc=0.8229
# [104/144] {'csp_n_components': 6, 'csp_reg': None, 'normalize_features': True, 'lda_n_components': 3, 'log_transform': False, 'classifier': 'SVM'} -> Kappa=0.7639 (±0.0442), Acc=0.8229
# [105/144] {'csp_n_components': 6, 'csp_reg': None, 'normalize_features': False, 'lda_n_components': None, 'log_transform': True, 'classifier': 'Bayesian'} -> Kappa=0.4676 (±0.0530), Acc=0.6007
# [106/144] {'csp_n_components': 6, 'csp_reg': None, 'normalize_features': False, 'lda_n_components': None, 'log_transform': True, 'classifier': 'SVM'} -> Kappa=0.4676 (±0.0479), Acc=0.6007
# [107/144] {'csp_n_components': 6, 'csp_reg': None, 'normalize_features': False, 'lda_n_components': None, 'log_transform': False, 'classifier': 'Bayesian'} -> Kappa=0.7593 (±0.0293), Acc=0.8194
# [108/144] {'csp_n_components': 6, 'csp_reg': None, 'normalize_features': False, 'lda_n_components': None, 'log_transform': False, 'classifier': 'SVM'} -> Kappa=0.6620 (±0.0530), Acc=0.7465
# [109/144] {'csp_n_components': 6, 'csp_reg': None, 'normalize_features': False, 'lda_n_components': 3, 'log_transform': True, 'classifier': 'Bayesian'} -> Kappa=0.4120 (±0.0660), Acc=0.5590
# [110/144] {'csp_n_components': 6, 'csp_reg': None, 'normalize_features': False, 'lda_n_components': 3, 'log_transform': True, 'classifier': 'SVM'} -> Kappa=0.3796 (±0.0463), Acc=0.5347
# [111/144] {'csp_n_components': 6, 'csp_reg': None, 'normalize_features': False, 'lda_n_components': 3, 'log_transform': False, 'classifier': 'Bayesian'} -> Kappa=0.7639 (±0.0304), Acc=0.8229
# [112/144] {'csp_n_components': 6, 'csp_reg': None, 'normalize_features': False, 'lda_n_components': 3, 'log_transform': False, 'classifier': 'SVM'} -> Kappa=0.7685 (±0.0424), Acc=0.8264
# [113/144] {'csp_n_components': 6, 'csp_reg': 'ledoit_wolf', 'normalize_features': True, 'lda_n_components': None, 'log_transform': True, 'classifier': 'Bayesian'} -> Kappa=0.4583 (±0.0496), Acc=0.5938
# [114/144] {'csp_n_components': 6, 'csp_reg': 'ledoit_wolf', 'normalize_features': True, 'lda_n_components': None, 'log_transform': True, 'classifier': 'SVM'} -> Kappa=0.4583 (±0.0619), Acc=0.5938
# [115/144] {'csp_n_components': 6, 'csp_reg': 'ledoit_wolf', 'normalize_features': True, 'lda_n_components': None, 'log_transform': False, 'classifier': 'Bayesian'} -> Kappa=0.7407 (±0.0131), Acc=0.8056
# [116/144] {'csp_n_components': 6, 'csp_reg': 'ledoit_wolf', 'normalize_features': True, 'lda_n_components': None, 'log_transform': False, 'classifier': 'SVM'} -> Kappa=0.7269 (±0.0442), Acc=0.7951
# [117/144] {'csp_n_components': 6, 'csp_reg': 'ledoit_wolf', 'normalize_features': True, 'lda_n_components': 3, 'log_transform': True, 'classifier': 'Bayesian'} -> Kappa=0.4861 (±0.0356), Acc=0.6146
# [118/144] {'csp_n_components': 6, 'csp_reg': 'ledoit_wolf', 'normalize_features': True, 'lda_n_components': 3, 'log_transform': True, 'classifier': 'SVM'} -> Kappa=0.4861 (±0.0331), Acc=0.6146
# [119/144] {'csp_n_components': 6, 'csp_reg': 'ledoit_wolf', 'normalize_features': True, 'lda_n_components': 3, 'log_transform': False, 'classifier': 'Bayesian'} -> Kappa=0.7870 (±0.0278), Acc=0.8403
# [120/144] {'csp_n_components': 6, 'csp_reg': 'ledoit_wolf', 'normalize_features': True, 'lda_n_components': 3, 'log_transform': False, 'classifier': 'SVM'} -> Kappa=0.7917 (±0.0304), Acc=0.8438
# [121/144] {'csp_n_components': 6, 'csp_reg': 'ledoit_wolf', 'normalize_features': False, 'lda_n_components': None, 'log_transform': True, 'classifier': 'Bayesian'} -> Kappa=0.4583 (±0.0496), Acc=0.5938
# [122/144] {'csp_n_components': 6, 'csp_reg': 'ledoit_wolf', 'normalize_features': False, 'lda_n_components': None, 'log_transform': True, 'classifier': 'SVM'} -> Kappa=0.4676 (±0.0479), Acc=0.6007
# [123/144] {'csp_n_components': 6, 'csp_reg': 'ledoit_wolf', 'normalize_features': False, 'lda_n_components': None, 'log_transform': False, 'classifier': 'Bayesian'} -> Kappa=0.7407 (±0.0131), Acc=0.8056
# [124/144] {'csp_n_components': 6, 'csp_reg': 'ledoit_wolf', 'normalize_features': False, 'lda_n_components': None, 'log_transform': False, 'classifier': 'SVM'} -> Kappa=0.7176 (±0.0513), Acc=0.7882
# [125/144] {'csp_n_components': 6, 'csp_reg': 'ledoit_wolf', 'normalize_features': False, 'lda_n_components': 3, 'log_transform': True, 'classifier': 'Bayesian'} -> Kappa=0.3981 (±0.0499), Acc=0.5486
# [126/144] {'csp_n_components': 6, 'csp_reg': 'ledoit_wolf', 'normalize_features': False, 'lda_n_components': 3, 'log_transform': True, 'classifier': 'SVM'} -> Kappa=0.4398 (±0.0241), Acc=0.5799
# [127/144] {'csp_n_components': 6, 'csp_reg': 'ledoit_wolf', 'normalize_features': False, 'lda_n_components': 3, 'log_transform': False, 'classifier': 'Bayesian'} -> Kappa=0.7870 (±0.0278), Acc=0.8403
# [128/144] {'csp_n_components': 6, 'csp_reg': 'ledoit_wolf', 'normalize_features': False, 'lda_n_components': 3, 'log_transform': False, 'classifier': 'SVM'} -> Kappa=0.7824 (±0.0331), Acc=0.8368
# [129/144] {'csp_n_components': 6, 'csp_reg': 'shrinkage', 'normalize_features': True, 'lda_n_components': None, 'log_transform': True, 'classifier': 'Bayesian'} -> Kappa=0.3657 (±0.0080), Acc=0.5243
# [130/144] {'csp_n_components': 6, 'csp_reg': 'shrinkage', 'normalize_features': True, 'lda_n_components': None, 'log_transform': True, 'classifier': 'SVM'} -> Kappa=0.3241 (±0.0207), Acc=0.4931
# [131/144] {'csp_n_components': 6, 'csp_reg': 'shrinkage', 'normalize_features': True, 'lda_n_components': None, 'log_transform': False, 'classifier': 'Bayesian'} -> Kappa=0.6852 (±0.0227), Acc=0.7639
# [132/144] {'csp_n_components': 6, 'csp_reg': 'shrinkage', 'normalize_features': True, 'lda_n_components': None, 'log_transform': False, 'classifier': 'SVM'} -> Kappa=0.6481 (±0.0346), Acc=0.7361
# [133/144] {'csp_n_components': 6, 'csp_reg': 'shrinkage', 'normalize_features': True, 'lda_n_components': 3, 'log_transform': True, 'classifier': 'Bayesian'} -> Kappa=0.3194 (±0.0080), Acc=0.4896
# [134/144] {'csp_n_components': 6, 'csp_reg': 'shrinkage', 'normalize_features': True, 'lda_n_components': 3, 'log_transform': True, 'classifier': 'SVM'} -> Kappa=0.3380 (±0.0154), Acc=0.5035
# [135/144] {'csp_n_components': 6, 'csp_reg': 'shrinkage', 'normalize_features': True, 'lda_n_components': 3, 'log_transform': False, 'classifier': 'Bayesian'} -> Kappa=0.7639 (±0.0274), Acc=0.8229
# [136/144] {'csp_n_components': 6, 'csp_reg': 'shrinkage', 'normalize_features': True, 'lda_n_components': 3, 'log_transform': False, 'classifier': 'SVM'} -> Kappa=0.7546 (±0.0202), Acc=0.8160
# [137/144] {'csp_n_components': 6, 'csp_reg': 'shrinkage', 'normalize_features': False, 'lda_n_components': None, 'log_transform': True, 'classifier': 'Bayesian'} -> Kappa=0.3657 (±0.0080), Acc=0.5243
# [138/144] {'csp_n_components': 6, 'csp_reg': 'shrinkage', 'normalize_features': False, 'lda_n_components': None, 'log_transform': True, 'classifier': 'SVM'} -> Kappa=0.3611 (±0.0207), Acc=0.5208
# [139/144] {'csp_n_components': 6, 'csp_reg': 'shrinkage', 'normalize_features': False, 'lda_n_components': None, 'log_transform': False, 'classifier': 'Bayesian'} -> Kappa=0.6852 (±0.0227), Acc=0.7639
# [140/144] {'csp_n_components': 6, 'csp_reg': 'shrinkage', 'normalize_features': False, 'lda_n_components': None, 'log_transform': False, 'classifier': 'SVM'} -> Kappa=0.6435 (±0.0605), Acc=0.7326
# [141/144] {'csp_n_components': 6, 'csp_reg': 'shrinkage', 'normalize_features': False, 'lda_n_components': 3, 'log_transform': True, 'classifier': 'Bayesian'} -> Kappa=0.3102 (±0.0154), Acc=0.4826
# [142/144] {'csp_n_components': 6, 'csp_reg': 'shrinkage', 'normalize_features': False, 'lda_n_components': 3, 'log_transform': True, 'classifier': 'SVM'} -> Kappa=0.3333 (±0.0131), Acc=0.5000
# [143/144] {'csp_n_components': 6, 'csp_reg': 'shrinkage', 'normalize_features': False, 'lda_n_components': 3, 'log_transform': False, 'classifier': 'Bayesian'} -> Kappa=0.7639 (±0.0274), Acc=0.8229
# [144/144] {'csp_n_components': 6, 'csp_reg': 'shrinkage', 'normalize_features': False, 'lda_n_components': 3, 'log_transform': False, 'classifier': 'SVM'} -> Kappa=0.7546 (±0.0154), Acc=0.8160

# ============================================================
# A01T
#   最优参数组合 (基于交叉验证 Kappa):
#   Kappa     = 0.7917
#   Accuracy  = 0.8438
#   参数      = {'csp_n_components': 6, 'csp_reg': 'ledoit_wolf', 'normalize_features': True, 'lda_n_components': 3, 'log_transform': False, 'classifier': 'SVM'}
# ============================================================
# ============================================================
# A04T
#   最优参数组合 (基于交叉验证 Kappa):
#   Kappa     = 0.6111
#   Accuracy  = 0.7083
#   参数      = {'csp_n_components': 6, 'csp_reg': None, 'normalize_features': True, 'lda_n_components': 3, 'log_transform': False, 'classifier': 'Bayesian'}
# ============================================================