#!/usr/bin/env python3
"""
训练脚本：预处理 → 特征提取 → 分类器训练

用法:
    python scripts/train.py                          # 单被试 A01，训练集 T
    python scripts/train.py --subject A03            # 指定被试
    python scripts/train.py --batch                  # 批量 A01/A02/A03...
    python scripts/train.py --step feature           # 仅运行特征提取
"""
import argparse
import os
import json
import time
import numpy as np
import mne
from sklearn.metrics import accuracy_score, cohen_kappa_score

from config import (
    get_raw_path, get_epoch_dir, get_label_dir,
    get_feature_dir, get_classifier_dir, get_result_dir,
)
from src.utils.session_config import SessionConfig
from src.data_preparation.data_loader import BCIDataLoader
from src.data_preparation.pre_processor import EEGPreprocessor
from src.data_preparation.epoch_processor import EpochProcessor
from src.feature_extraction.ovocsp_feature_extractor import OVOCspFeatureExtractor
from src.classification.bayesian_classifier import BayesianClassifier
from src.evaluation.evaluator import BCIEvaluator


# ============================================================
# Pipeline 函数
# ============================================================

def run_preprocessing(raw_path, epoch_dir, label_dir, cfg, *,
                      save=True, save_label=False, verbose=True):
    """加载 → 预处理 → 分段 → 保存。返回 epochs。"""
    loader = BCIDataLoader(
        eog_channels=cfg.get('eog_channels'),
        set_montage=cfg.get('set_montage'),
        channel_renaming=cfg.get('channel_renaming'),
        montage_name=cfg.get('montage_name'))

    preprocessor = EEGPreprocessor(
        resample_freq=cfg.get('resample_freq'),
        filter_ica=cfg.get('filter_ica'),
        filter_mi=cfg.get('filter_mi'),
        ref_type=cfg.get('ref_type', 'average'),
        bad_channels_manual=cfg.get('bad_channels_manual'),
        ica_n_components=cfg.get('ica_n_components'),
        ica_random_state=cfg.get('ica_random_state'),
        ica_method=cfg.get('ica_method'),
        ica_exclude_manual=cfg.get('ica_exclude_manual'))

    epoch_processor = EpochProcessor(
        tmin=cfg.get('tmin'), tmax=cfg.get('tmax'),
        events_mapping=cfg.get('mi_event_mapping'),
        expected_trials=cfg.get('expected_trials'))

    if verbose:
        print(f"\n{'='*40}\n  预处理: {raw_path}\n{'='*40}")
    raw = loader.load(raw_path, verbose=verbose)
    raw = preprocessor.process(raw, verbose=verbose)

    eog_chs = [ch for ch, ct in zip(raw.ch_names, raw.get_channel_types()) if ct == 'eog']
    epochs = epoch_processor.process(raw, drop_channels=list(set(eog_chs + raw.info['bads'])), verbose=verbose)
    if verbose:
        print(f"  试次数: {len(epochs)}, 通道数: {len(epochs.ch_names)}")

    if save:
        os.makedirs(epoch_dir, exist_ok=True)
        epochs.save(f"{epoch_dir}/{cfg._subject_id}{cfg._session}_epo.fif", overwrite=True)

    if save_label:
        os.makedirs(label_dir, exist_ok=True)
        np.save(f"{label_dir}/{cfg._subject_id}{cfg._session}_labels.npy", epochs.events[:, 2])

    return epochs


def run_feature_extraction(epoch_path, label_path, feature_dir, cfg, *,
                           save_features=True, save_extractor=True, verbose=True):
    """读 epochs → 训练 CSP → 保存特征。返回 (features, extractor)。"""
    if verbose:
        print(f"\n{'='*40}\n  特征提取: {epoch_path}\n{'='*40}")
    X = mne.read_epochs(epoch_path, preload=True, verbose=False).get_data()
    y = np.load(label_path)
    if verbose:
        print(f"  数据: {X.shape}, 类别: {np.unique(y)}")

    extractor = OVOCspFeatureExtractor(
        csp_n_components=cfg.get('csp_n_components'),
        csp_reg=cfg.get('csp_reg'),
        log_transform=cfg.get('log_transform'),
        normalize_features=cfg.get('normalize_features'),
        lda_n_components=cfg.get('lda_n_components'))
    features = extractor.fit_transform(X, y, verbose=verbose)

    if save_features:
        os.makedirs(feature_dir, exist_ok=True)
        np.save(f"{feature_dir}/{cfg._subject_id}{cfg._session}_ovocsp_features.npy", features)
    if save_extractor:
        os.makedirs(feature_dir, exist_ok=True)
        extractor.save(f"{feature_dir}/{cfg._subject_id}{cfg._session}_ovocsp_extractor.joblib")
    return features, extractor


def run_classification(feature_path, label_path, clf_dir, res_dir, cfg, *,
                       classifier_class=BayesianClassifier, evaluator_class=BCIEvaluator,
                       save=True, verbose=True):
    """加载特征 → 训练 → CV → 保存。返回 (result_dict, classifier)。"""
    _cv_folds = cfg.get('classify_cv_folds', 4)
    _random_state = cfg.get('classify_random_state', 17)
    _do_cv = cfg.get('classify_do_cv', True)

    if verbose:
        print(f"\n{'='*40}\n  分类训练: {classifier_class.__name__}\n{'='*40}")

    features = np.load(feature_path)
    labels = np.load(label_path)
    clf = classifier_class().fit(features, labels)
    train_pred = clf.predict(features)
    if np.issubdtype(train_pred.dtype, np.integer) or train_pred.ndim == 1:
        train_pred = train_pred.astype(int).flatten()
    overall_acc = accuracy_score(labels, train_pred)
    overall_kappa = cohen_kappa_score(labels, train_pred)

    cv_results = None
    if _do_cv:
        cv_clf = classifier_class()
        if hasattr(cv_clf, 'random_state'):
            cv_clf.random_state = _random_state
        cv_results = BCIEvaluator(cv_folds=_cv_folds, random_state=_random_state).evaluate(features, labels, cv_clf)

    if save:
        os.makedirs(clf_dir, exist_ok=True)
        clf.save(f"{clf_dir}/{cfg._subject_id}{cfg._session}_bayesian_clf.joblib")

    result = {
        'dataset': cfg._dataset_name, 'subject_id': cfg._subject_id, 'session': cfg._session,
        'classifier': clf.__class__.__name__, 'cv_folds': _cv_folds, 'random_state': _random_state,
        'overall_accuracy': round(overall_acc, 4), 'overall_kappa': round(overall_kappa, 4),
        'do_cv': _do_cv, 'cv_summary': None, 'config_snapshot': dict(cfg),
    }
    if cv_results is not None:
        result['cv_summary'] = {
            'accuracy_mean': round(cv_results['accuracy_mean'], 4),
            'accuracy_std': round(cv_results['accuracy_std'], 4),
            'kappa_mean': round(cv_results['kappa_mean'], 4),
            'kappa_std': round(cv_results['kappa_std'], 4)}
    if save:
        os.makedirs(res_dir, exist_ok=True)
        fp = f"{res_dir}/{cfg._subject_id}{cfg._session}_bayesian_train_results_{int(time.time())}.json"
        json.dump(result, open(fp, 'w'), indent=2, ensure_ascii=False)

    if verbose:
        print(f"  整体 Accuracy: {overall_acc:.4f}, Kappa: {overall_kappa:.4f}")
        if cv_results:
            print(f"  CV Accuracy: {cv_results['accuracy_mean']:.4f} ± {cv_results['accuracy_std']:.4f}")
            print(f"  CV Kappa:    {cv_results['kappa_mean']:.4f} ± {cv_results['kappa_std']:.4f}")
    return result, clf


# ============================================================
# 主入口
# ============================================================

def main():
    p = argparse.ArgumentParser(description="BCI 训练脚本")
    p.add_argument("--dataset", default="BCICIV_2a")
    p.add_argument("--subject", default="A01")
    p.add_argument("--session", default="T")
    p.add_argument("--batch", action="store_true", help="批量模式（A01/A02/A03...）")
    p.add_argument("--step", choices=["preprocess", "feature", "classify"],
                   help="仅运行指定步骤")
    args = p.parse_args()

    if args.batch:
        subjects = ["A01", "A02", "A03", "A04", "A05", "A06", "A07", "A08", "A09"]
    else:
        subjects = [args.subject]

    all_kappas, all_accuracies = [], []
    results = {}
    for sid in subjects:
        print(f"\n{'='*60}\n  被试: {sid}\n{'='*60}")
        try:
            cfg = SessionConfig.from_dataset(args.dataset, sid, args.session)

            if args.step in (None, "preprocess"):
                run_preprocessing(
                    get_raw_path(args.dataset, sid, args.session),
                    get_epoch_dir(args.dataset), get_label_dir(args.dataset), cfg,
                    save_label=True)

            if args.step in (None, "feature"):
                base = f"{sid}{args.session}"
                run_feature_extraction(
                    f"{get_epoch_dir(args.dataset)}/{base}_epo.fif",
                    f"{get_label_dir(args.dataset)}/{base}_labels.npy",
                    get_feature_dir(args.dataset), cfg)

            if args.step in (None, "classify"):
                base = f"{sid}{args.session}"
                result, _ = run_classification(
                    f"{get_feature_dir(args.dataset)}/{base}_ovocsp_features.npy",
                    f"{get_label_dir(args.dataset)}/{base}_labels.npy",
                    get_classifier_dir(args.dataset), get_result_dir(args.dataset), cfg)
                cv = result.get("cv_summary", {})
                k = cv.get("kappa_mean")
                if k is not None:
                    all_kappas.append(k)
                    all_accuracies.append(cv.get("accuracy_mean"))
                    results[sid] = {'kappa': k, 'accuracy': cv['accuracy_mean']}
                    print(f"  ✓ CV Kappa: {k:.4f}, Accuracy: {cv['accuracy_mean']:.4f}")
                else:
                    results[sid] = None
            else:
                results[sid] = None  # 仅运行预处理/特征时不记录分类结果

        except Exception as e:
            print(f"  ✗ 失败: {e}")
            results[sid] = None

    if len(subjects) > 1 and all_kappas:
        print(f"\n{'='*60}")
        print(f"训练集 CV 汇总")
        print(f"{'='*60}")
        print(f"成功被试数: {len(all_kappas)}")
        print(f"平均 Kappa  : {np.mean(all_kappas):.4f} ± {np.std(all_kappas):.4f}")
        print(f"平均 Accuracy: {np.mean(all_accuracies):.4f} ± {np.std(all_accuracies):.4f}")
        print(f"\n各被试详情:")
        for sid in subjects:
            r = results.get(sid)
            if r:
                print(f"  {sid}: Kappa={r['kappa']:.4f}, Acc={r['accuracy']:.4f}")
            else:
                print(f"  {sid}: 失败")


if __name__ == "__main__":
    main()


# ============================================================
# 历史结果（训练集 CV）
# ============================================================
# 平均 Kappa  : 0.7387 ± 0.1120
# 平均 Accuracy: 0.8040 ± 0.0840
# 各被试详情:
#   A01: Kappa=0.7870, Acc=0.8403
#   A02: Kappa=0.7546, Acc=0.8160
#   A03: Kappa=0.8611, Acc=0.8958
#   A04: Kappa=0.6343, Acc=0.7257
#   A05: Kappa=0.5463, Acc=0.6597
#   A06: Kappa=0.6389, Acc=0.7292
#   A07: Kappa=0.8704, Acc=0.9028
#   A08: Kappa=0.8704, Acc=0.9028
#   A09: Kappa=0.6852, Acc=0.7639
