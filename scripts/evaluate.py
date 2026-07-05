#!/usr/bin/env python3
"""
测试集评估脚本
用训练集(T)训练的模型，评估测试集(E)数据。

用法:
    python scripts/evaluate.py                          # 单被试 A01
    python scripts/evaluate.py --subject A03            # 指定被试
    python scripts/evaluate.py --batch                  # 批量 A01/A02/A03...
    python scripts/evaluate.py --force-preprocess       # 强制重新预处理
"""
import argparse
import json
import os
import time
import numpy as np
import mne
from sklearn.metrics import cohen_kappa_score, accuracy_score

from config import (
    get_raw_path, get_epoch_dir, get_label_dir, get_label_path,
    get_epoch_path, get_extractor_path, get_classifier_path, get_result_dir,
)
from src.utils.session_config import SessionConfig
from src.data_preparation.data_loader import BCIDataLoader
from src.data_preparation.pre_processor import EEGPreprocessor
from src.data_preparation.epoch_processor import EpochProcessor
from src.feature_extraction.ovocsp_feature_extractor import OVOCspFeatureExtractor
from src.classification.bayesian_classifier import BayesianClassifier


def run_preprocessing(raw_path, epoch_dir, label_dir, cfg, *,
                      save=True, save_label=False, verbose=True):
    """精简版预处理——仅用于评估脚本（复用 train.py 的逻辑）。"""
    loader = BCIDataLoader(
        eog_channels=cfg.get('eog_channels'),
        set_montage=cfg.get('set_montage'),
        channel_renaming=cfg.get('channel_renaming'),
        montage_name=cfg.get('montage_name'))
    preprocessor = EEGPreprocessor(
        resample_freq=cfg.get('resample_freq'),
        filter_ica=cfg.get('filter_ica'), filter_mi=cfg.get('filter_mi'),
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
        print(f"  预处理: {raw_path}")
    raw = loader.load(raw_path, verbose=verbose)
    raw = preprocessor.process(raw, verbose=verbose)
    eog_chs = [ch for ch, ct in zip(raw.ch_names, raw.get_channel_types()) if ct == 'eog']
    epochs = epoch_processor.process(raw, drop_channels=list(set(eog_chs + raw.info['bads'])))
    if verbose and epochs is not None:
        print(f"  试次数: {len(epochs)}")

    if save:
        os.makedirs(epoch_dir, exist_ok=True)
        epochs.save(f"{epoch_dir}/{cfg._subject_id}{cfg._session}_epo.fif", overwrite=True)
    if save_label:
        os.makedirs(label_dir, exist_ok=True)
        np.save(f"{label_dir}/{cfg._subject_id}{cfg._session}_labels.npy", epochs.events[:, 2])
    return epochs


def main():
    p = argparse.ArgumentParser(description="测试集评估")
    p.add_argument("--dataset", default="BCICIV_2a")
    p.add_argument("--subject", default="A01")
    p.add_argument("--train-session", default="T")
    p.add_argument("--eval-session", default="E")
    p.add_argument("--batch", action="store_true")
    p.add_argument("--force-preprocess", action="store_true")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    subjects = ["A01", "A02", "A03", "A04", "A05", "A06", "A07", "A08", "A09"] if args.batch else [args.subject]
    results = {}

    for sid in subjects:
        identifier = f"{sid}{args.eval_session}"
        if not args.quiet:
            print(f"\n{'='*60}\n  评估: {args.dataset} / {identifier}  (模型: {sid}{args.train_session})\n{'='*60}")
        try:
            epoch_path = get_epoch_path(args.dataset, sid, args.eval_session)
            if not os.path.exists(epoch_path) or args.force_preprocess:
                cfg = SessionConfig.from_dataset(args.dataset, sid, args.eval_session)
                cfg['mi_event_mapping'] = {"768": 6}
                run_preprocessing(
                    get_raw_path(args.dataset, sid, args.eval_session),
                    get_epoch_dir(args.dataset), get_label_dir(args.dataset), cfg,
                    save_label=False, verbose=not args.quiet)
            else:
                if not args.quiet:
                    print("  (复用已有 epochs)")

            epochs = mne.read_epochs(epoch_path, preload=True, verbose=False)
            extractor = OVOCspFeatureExtractor.load(
                get_extractor_path(args.dataset, sid, args.train_session, "ovocsp"))
            clf = BayesianClassifier.load(
                get_classifier_path(args.dataset, sid, args.train_session, "bayesian"))

            y_pred = clf.predict(extractor.transform(epochs.get_data())).flatten().astype(int)
            y_true = np.load(get_label_path(args.dataset, sid, args.eval_session))
            kappa, acc = cohen_kappa_score(y_true, y_pred), accuracy_score(y_true, y_pred)
            results[sid] = {'kappa': kappa, 'accuracy': acc}

            # 保存评估结果
            res_dir = get_result_dir(args.dataset)
            os.makedirs(res_dir, exist_ok=True)
            eval_result = {
                'dataset': args.dataset,
                'subject': sid,
                'train_session': args.train_session,
                'eval_session': args.eval_session,
                'model': {
                    'extractor': get_extractor_path(args.dataset, sid, args.train_session, "ovocsp"),
                    'classifier': get_classifier_path(args.dataset, sid, args.train_session, "bayesian"),
                },
                'kappa': round(kappa, 4),
                'accuracy': round(acc, 4),
                'n_trials': len(y_true),
                'config_snapshot': dict(SessionConfig.from_dataset(args.dataset, sid, args.eval_session)),
            }
            fp = f"{res_dir}/{sid}{args.eval_session}_bayesian_eval_results_{int(time.time())}.json"
            json.dump(eval_result, open(fp, 'w'), indent=2, ensure_ascii=False)
            if not args.quiet:
                print(f"  → 结果已保存: {fp}")
            if not args.quiet:
                print(f"  ✓ Kappa: {kappa:.4f}, Accuracy: {acc:.4f}")
        except Exception as e:
            print(f"  ✗ 失败: {e}")
            results[sid] = None

    if len(subjects) > 1:
        valid = {k: v for k, v in results.items() if v is not None}
        if valid:
            ks = [v['kappa'] for v in valid.values()]
            ac = [v['accuracy'] for v in valid.values()]
            print(f"\n{'='*60}")
            print(f"测试集评估汇总")
            print(f"{'='*60}")
            print(f"成功被试数: {len(valid)}")
            print(f"平均 Kappa  : {np.mean(ks):.4f} ± {np.std(ks):.4f}")
            print(f"平均 Accuracy: {np.mean(ac):.4f} ± {np.std(ac):.4f}")
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
# 历史结果（测试集评估）
# ============================================================
# 平均 Kappa  : 0.4583 ± 0.1884
# 平均 Accuracy: 0.5938 ± 0.1413
# 各被试详情:
#   A01: Kappa=0.6574, Acc=0.7431
#   A02: Kappa=0.4954, Acc=0.6215
#   A03: Kappa=0.6343, Acc=0.7257
#   A04: Kappa=0.3287, Acc=0.4965
#   A05: Kappa=0.1528, Acc=0.3646
#   A06: Kappa=0.1620, Acc=0.3715
#   A07: Kappa=0.6389, Acc=0.7292
#   A08: Kappa=0.4676, Acc=0.6007
#   A09: Kappa=0.5880, Acc=0.6910
