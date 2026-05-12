#!/usr/bin/env python3
"""
动态分段参数搜索脚本
- 固定预处理（从 config.json 读取）
- 每次改变 tmin/tmax 都重新分段
- 自动将最佳参数保存到 config.json 中对应被试的 session T
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import itertools
import numpy as np
from config import get_raw_path, get_dataset_dir, ensure_dir
from src.utils.session_config import SessionConfig
from src.data_preparation.data_loader import BCIDataLoader
from src.data_preparation.pre_processor import EEGPreprocessor
from src.data_preparation.epoch_processor import EpochProcessor
from src.feature_extraction.ovocsp_feature_extractor import OVOCspFeatureExtractor
from src.classification.bayesian_classifier import BayesianClassifier
from src.evaluation.evaluator import BCIEvaluator


# ================= 配置区域 =================
DATASET = 'BCICIV_2a'
SUBJECT = 'A09'                 # 可改为其他被试，或循环处理多个
SESSION = 'T'                   # 只在训练集上搜索
CV_FOLDS = 4
RANDOM_STATE = 17

# ---------- 参数网格（按需修改）----------
SEARCH_PARAMS = {
    # 窗口（相对于训练集的线索事件，即 769-772） 最关键的参数
    'tmin': [1.0, 1.5, 2.0, 2.5],
    'tmax': [3.5, 4.0, 4.5, 5],
    # 特征提取
    'csp_n_components': [4, 6],
    'log_transform': [True, False],
    'lda_n_components': [3, None],          # None 表示不降维
    # 分类器
    'classifier': ['Bayesian'] # ['Bayesian', 'SVM'],
}

# 这些参数固定不变，直接从 config.json 的 default 中读取
FIXED_PARAMS = ['csp_reg', 'normalize_features', 'random_state',
                'resample_freq', 'filter_ica', 'filter_mi', 'ref_type',
                'ica_n_components', 'ica_random_state', 'ica_method',
                'bad_channels_manual', 'ica_exclude_manual']
# ==========================================


def create_clf(clf_type, random_state=RANDOM_STATE):
    if clf_type == 'Bayesian':
        return BayesianClassifier()
    elif clf_type == 'SVM':
        from sklearn.svm import SVC
        return SVC(kernel='rbf', random_state=random_state)
    else:
        raise ValueError(f"Unknown classifier: {clf_type}")


def main():
    # 1. 加载该被试的训练集配置（合并 default + 覆盖）
    config = SessionConfig.from_dataset(DATASET, SUBJECT, SESSION)
    print(f"加载 {SUBJECT}{SESSION} 配置完成")

    # 2. 固定预处理参数从 config 中提取（之后不再改变）
    pre_params = {k: config.get(k) for k in FIXED_PARAMS if config.get(k) is not None}
    # 补充默认值（若 config 中没有）
    pre_params.setdefault('resample_freq', None)
    pre_params.setdefault('filter_ica', None)
    pre_params.setdefault('filter_mi', [8, 30])
    pre_params.setdefault('ref_type', 'average')
    pre_params.setdefault('ica_n_components', None)
    pre_params.setdefault('ica_random_state', 71)
    pre_params.setdefault('ica_method', 'fastica')
    pre_params.setdefault('bad_channels', [])
    pre_params.setdefault('ica_exclude_manual', [])
    pre_params.setdefault('csp_reg', 'ledoit_wolf')
    pre_params.setdefault('normalize_features', True)
    pre_params.setdefault('random_state', RANDOM_STATE)

    # 3. 加载原始数据并执行一次性的预处理（滤波、重参考等）
    print("加载原始数据并运行预处理（不包含分段）...")
    loader = BCIDataLoader(eog_channels=config.eog_channels)
    raw = loader.load(get_raw_path(DATASET, SUBJECT, SESSION))
    preprocessor = EEGPreprocessor(
        resample_freq=pre_params['resample_freq'],
        filter_ica=pre_params['filter_ica'],
        filter_mi=pre_params['filter_mi'],
        ref_type=pre_params['ref_type'],
        bad_channels_manual=pre_params['bad_channels_manual'],
        ica_n_components=pre_params['ica_n_components'],
        ica_random_state=pre_params['ica_random_state'],
        ica_method=pre_params['ica_method'],
        ica_exclude_manual=pre_params['ica_exclude_manual']
    )
    raw_mi = preprocessor.process(raw, verbose=False)
    print("预处理完成，数据形状:", raw_mi.get_data().shape)

    # 4. 构建网格组合
    keys = list(SEARCH_PARAMS.keys())
    values = list(SEARCH_PARAMS.values())
    combos = list(itertools.product(*values))
    total = len(combos)
    print(f"共有 {total} 组参数待搜索...")

    # 5. 网格搜索
    best_kappa = -1.0
    best_params = None
    best_accuracy = 0.0

    for idx, combo in enumerate(combos):
        params = dict(zip(keys, combo))
        # 合并固定的预处理和特征提取参数
        full_params = pre_params.copy()
        full_params.update(params)

        # 窗口有效性检查
        if full_params['tmin'] >= full_params['tmax']:
            continue

        try:
            # ---- 动态分段：使用当前的 tmin/tmax 重新生成 epochs ----
            epoch_proc = EpochProcessor(
                tmin=full_params['tmin'],
                tmax=full_params['tmax'],
                events_mapping=config.mi_event_mapping,   # 训练集的线索事件映射
                expected_trials=config.expected_trials
            )
            eog_chs = [ch for ch, ch_type in zip(raw_mi.ch_names, raw_mi.get_channel_types()) if ch_type == 'eog']
            drop_chs = list(set(eog_chs + raw_mi.info['bads']))
            epochs = epoch_proc.process(raw_mi, drop_channels=drop_chs)
            X = epochs.get_data()
            y = epochs.events[:, 2]          # 已映射为 1-4

            # ---- 特征提取 ----
            extractor = OVOCspFeatureExtractor(
                csp_n_components=full_params['csp_n_components'],
                csp_reg=full_params['csp_reg'],
                log_transform=full_params['log_transform'],
                normalize_features=full_params['normalize_features'],
                lda_n_components=full_params['lda_n_components'],
            )
            features = extractor.fit_transform(X, y, verbose=False)

            # ---- 分类器与评估 ----
            clf = create_clf(full_params['classifier'], RANDOM_STATE)
            evaluator = BCIEvaluator(cv_folds=CV_FOLDS, random_state=RANDOM_STATE)
            result = evaluator.evaluate(features, y, clf)
            kappa = result['kappa_mean']
            acc = result['accuracy_mean']
            print(f"[{idx+1:3d}/{total}] {params} → Kappa={kappa:.4f} (±{result['kappa_std']:.4f}), Acc={acc:.4f}")

            if kappa > best_kappa:
                best_kappa = kappa
                best_params = full_params
                best_accuracy = acc

        except Exception as e:
            print(f"[{idx+1:3d}/{total}] {params} 出错: {e}")

    # 6. 输出并保存最优参数
    if best_params is None:
        print("没有找到成功的参数组合。")
        return

    print("\n" + "=" * 60)
    print(f"{DATASET}-{SUBJECT}-{SESSION} 最佳参数组合 (基于训练集 CV Kappa):")
    print(f"  Kappa     = {best_kappa:.4f}")
    print(f"  Accuracy  = {best_accuracy:.4f}")
    # 只显示搜索过的参数
    best_display = {k: best_params[k] for k in SEARCH_PARAMS.keys() if k in best_params}
    print(f"  参数      = {best_display}")

    # # 7. 将搜索得到的参数写回 config.json 中该被试的 session T
    # #    注意：只更新搜索过的参数，不覆盖其他配置
    # print("正在保存最优参数到 config.json ...")
    # for key in SEARCH_PARAMS.keys():
    #     if key in best_params:
    #         config[key] = best_params[key]
    # config.save()
    # print(f"✓ 最优参数已保存到 {SUBJECT} 的 {SESSION} 配置中。")


if __name__ == '__main__':
    main()


# ============================================================
# BCICIV_2a-A01-T 最佳参数组合 (基于训练集 CV Kappa):
#   Kappa     = 0.7963
#   Accuracy  = 0.8472
#   参数      = {'tmin': 1.0, 'tmax': 4.0, 'csp_n_components': 6, 'log_transform': False, 'lda_n_components': 3, 'classifier': 'Bayesian'}
# ============================================================
# BCICIV_2a-A02-T 最佳参数组合 (基于训练集 CV Kappa):
#   Kappa     = 0.7824
#   Accuracy  = 0.8368
#   参数      = {'tmin': 2.0, 'tmax': 5, 'csp_n_components': 6, 'log_transform': False, 'lda_n_components': 3, 'classifier': 'Bayesian'}
# ============================================================
# BCICIV_2a-A03-T 最佳参数组合 (基于训练集 CV Kappa):
#   Kappa     = 0.8750
#   Accuracy  = 0.9062
#   参数      = {'tmin': 1.0, 'tmax': 5, 'csp_n_components': 6, 'log_transform': False, 'lda_n_components': 3, 'classifier': 'Bayesian'}
# ============================================================
# BCICIV_2a-A04-T 最佳参数组合 (基于训练集 CV Kappa):
#   Kappa     = 0.6528
#   Accuracy  = 0.7396
#   参数      = {'tmin': 1.5, 'tmax': 3.5, 'csp_n_components': 6, 'log_transform': False, 'lda_n_components': 3, 'classifier': 'Bayesian'}
# ============================================================
# BCICIV_2a-A05-T 最佳参数组合 (基于训练集 CV Kappa):
#   Kappa     = 0.5787
#   Accuracy  = 0.6840
#   参数      = {'tmin': 1.0, 'tmax': 4.0, 'csp_n_components': 6, 'log_transform': False, 'lda_n_components': 3, 'classifier': 'Bayesian'}
# ============================================================
# BCICIV_2a-A06-T 最佳参数组合 (基于训练集 CV Kappa):
#   Kappa     = 0.6528
#   Accuracy  = 0.7396
#   参数      = {'tmin': 1.0, 'tmax': 4.0, 'csp_n_components': 6, 'log_transform': False, 'lda_n_components': 3, 'classifier': 'Bayesian'}
# ============================================================
# BCICIV_2a-A07-T 最佳参数组合 (基于训练集 CV Kappa):
#   Kappa     = 0.8287
#   Accuracy  = 0.8715
#   参数      = {'tmin': 1.0, 'tmax': 4.0, 'csp_n_components': 6, 'log_transform': False, 'lda_n_components': 3, 'classifier': 'Bayesian'}
# ============================================================
# BCICIV_2a-A08-T 最佳参数组合 (基于训练集 CV Kappa):
#   Kappa     = 0.8704
#   Accuracy  = 0.9028
#   参数      = {'tmin': 1.0, 'tmax': 4.0, 'csp_n_components': 6, 'log_transform': False, 'lda_n_components': 3, 'classifier': 'Bayesian'}
# ============================================================
# BCICIV_2a-A09-T 最佳参数组合 (基于训练集 CV Kappa):
#   Kappa     = 0.6806
#   Accuracy  = 0.7604
#   参数      = {'tmin': 1.0, 'tmax': 4.0, 'csp_n_components': 6, 'log_transform': False, 'lda_n_components': 3, 'classifier': 'Bayesian'}