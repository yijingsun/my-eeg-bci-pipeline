"""
分类管道：加载特征 → 训练分类器 → 交叉验证 → 保存模型
支持灵活修改内部参数
"""
import os
import json
import time
import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score
from config import get_feature_path, get_classifier_dir, get_label_path, get_result_dir, ensure_dir
from src.classification.bayesian_classifier import BayesianClassifier
from src.evaluation.evaluator import BCIEvaluator
from src.utils import SessionConfig


class TrainClassifierPipeline:
    """轻量级分类器训练流水线，支持运行参数和手动修改内部属性"""

    def __init__(self, dataset_name: str, subject_id: str, session: str):
        if dataset_name is None or subject_id is None or session is None:
            raise ValueError("dataset_name, subject_id, session 不能为空")
        self.dataset_name = dataset_name
        self.subject_id = subject_id
        self.session = session
        self.cfg = SessionConfig.from_dataset(dataset_name, subject_id, session)

        # 可以替换的分类器类（默认贝叶斯）
        self.classifier_class = BayesianClassifier
        # 评估器实例，也可以自行替换
        self.evaluator_class = BCIEvaluator


    def run(self, save: bool = True, verbose: bool = True):
        """
        运行分类流程
        """
        # 运行变量：配置文件中读取
        _cv_folds = self.cfg.get('classify_cv_folds', 4)
        _random_state = self.cfg.get('classify_random_state', 17)
        _do_cv = self.cfg.get('classify_do_cv', True)

        if verbose:
            print('=' * 50)
            print(f"开始训练{self.classifier_class.__name__}分类器...")

        # 1. 加载特征+标签
        feature_path = get_feature_path(self.dataset_name, self.subject_id, self.session, 'ovocsp')
        features = np.load(feature_path)
        label_path = get_label_path(self.dataset_name, self.subject_id, self.session)
        labels = np.load(label_path)

        if verbose:
            print(f"✓ 特征加载完成")
            print(f"特征维度: {features.shape}, 类别分布: {np.bincount(labels)[1:]}")

        # 2. 全量数据训练分类器（使用当前分类器类）
        clf = self.classifier_class()
        clf.fit(features, labels)
        if verbose:
            print(f"\n✓ {clf.__class__.__name__}分类器已训练")
        
        # 3. 预测并计算整体指标
        train_pred = clf.predict(features)
        if np.issubdtype(train_pred.dtype, np.integer) or train_pred.ndim == 1:
             train_pred = train_pred.astype(int).flatten()
        overall_acc = accuracy_score(labels, train_pred)
        overall_kappa = cohen_kappa_score(labels, train_pred)
        if verbose:
            print(f"训练集整体 Accuracy: {overall_acc:.4f}, Kappa: {overall_kappa:.4f}")

        # 4. 交叉验证
        if _do_cv:
            evaluator = self.evaluator_class(cv_folds=_cv_folds, random_state=_random_state)
            cv_clf = self.classifier_class()
            if hasattr(cv_clf, 'random_state'):
                cv_clf.random_state = _random_state
            cv_results = evaluator.evaluate(features, labels, cv_clf)
            if verbose:
                print(f"\n交叉验证 ({_cv_folds}-fold):")
                print(f"  Accuracy: {cv_results['accuracy_mean']:.4f} ± {cv_results['accuracy_std']:.4f}")
                print(f"  Kappa:    {cv_results['kappa_mean']:.4f} ± {cv_results['kappa_std']:.4f}")

        # 5. 保存模型
        if save:
            clf_dir = get_classifier_dir(self.dataset_name)
            ensure_dir(clf_dir)
            clf_file = os.path.join(clf_dir, f'{self.subject_id}{self.session}_bayesian_clf.joblib')
            clf.save(clf_file)
            if verbose:
                print(f"\n✓ 分类器已保存至: {clf_file}")

        # 5. 保存交叉验证结果
        # 保存为 JSON 包括前期处理参数
        config_snapshot = dict(self.cfg)
        result = {
            'dataset': self.dataset_name,
            'subject_id': self.subject_id,
            'session': self.session,
            "classifier": clf.__class__.__name__,
            'cv_folds': _cv_folds,
            'random_state': _random_state,
            'overall_accuracy': round(overall_acc, 4),
            'overall_kappa': round(overall_kappa, 4),
            'do_cv': _do_cv,
            "cv_summary": None,
            "config_snapshot": config_snapshot
        }
        if cv_results is not None:
            result['cv_summary'] = {
                    'accuracy_mean': round(cv_results['accuracy_mean'], 4),
                    'accuracy_std': round(cv_results['accuracy_std'], 4),
                    'kappa_mean': round(cv_results['kappa_mean'], 4),
                    'kappa_std': round(cv_results['kappa_std'], 4),
                } 
        if save:
            cv_results_dir = get_result_dir(self.dataset_name)
            ensure_dir(cv_results_dir)
            cv_results_path = os.path.join(cv_results_dir, f'{self.subject_id}{self.session}_bayesian_train_results_{int(time.time())}.json')

            with open(cv_results_path, 'w') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
                if verbose:
                    print(f"✓ 模型评估结果已保存至: {cv_results_path}")

        return result, clf