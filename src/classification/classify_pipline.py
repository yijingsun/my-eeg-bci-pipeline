# src/classification/classify_pipeline.py
"""
分类管道：加载特征 → 训练分类器 → 交叉验证 → 保存模型
支持灵活修改内部参数
"""
import os
from lark import Tree
import numpy as np
from config import get_epoch_path, get_dataset_dir, ensure_dir
from src.classification.bayesian_classifier import BayesianClassifier
from src.evaluation.evaluator import BCIEvaluator
from src.utils import load_session_config


class ClassifyPipeline:
    """轻量级分类流水线，支持运行参数和手动修改内部属性"""

    def __init__(self, dataset_name: str = 'BCICIV_2a'):
        self.dataset_name = dataset_name
        # 可以替换的分类器类（默认贝叶斯）
        self.classifier_class = BayesianClassifier
        # 评估器实例，也可以自行替换
        self.evaluator_class = BCIEvaluator


    def run(self, subject_id: str = 'A01', session: str = 'T',
            cv_folds: int = None, random_state: int = None,
            do_cv: bool = None, save: bool = True,
            verbose: bool = True):
        """
        运行分类流程，可选参数会临时覆盖实例属性（不修改实例）

        cv_folds, random_state, do_cv : 交叉检验参数
        """
        # 局部变量：如果传入则用传入值，否读配置文件，配置文件不存在则用默认值
        cfg = load_session_config(self.dataset_name, subject_id, session)
        _cv_folds = cv_folds if cv_folds is not None else cfg.get('classify_cv_folds')
        _random_state = random_state if random_state is not None else cfg.get('classify_random_state')
        _do_cv = do_cv if do_cv is not None else cfg.get('classify_do_cv')
        _save = save
        _verbose = verbose

        # 1. 加载特征+标签
        feature_dir = os.path.join(get_dataset_dir(self.dataset_name), 'model', 'feature')
        feature_file = os.path.join(feature_dir, f'{subject_id}{session}_ovocsp_features.npz')
        data = np.load(feature_file)
        features = data['features']
        labels = data['labels']

        if _verbose:
            print(f"特征维度: {features.shape}, 类别分布: {np.bincount(labels)[1:]}")

        # 2. 训练分类器（使用当前分类器类）
        clf = self.classifier_class()
        clf.fit(features, labels)
        if _verbose:
            print(f"\n分类器已训练")

        # 3. 交叉验证
        if _do_cv:
            # 每次 run 根据最新参数创建评估器，保证 cv_folds 和 random_state 同步
            eval_cv = self.evaluator_class(cv_folds=_cv_folds, random_state=_random_state)
            results = eval_cv.evaluate(features, labels, clf)
            if _verbose:
                print(f"\n交叉验证 ({_cv_folds}-fold):")
                print(f"  Accuracy: {results['accuracy_mean']:.4f} ± {results['accuracy_std']:.4f}")
                print(f"  Kappa:    {results['kappa_mean']:.4f} ± {results['kappa_std']:.4f}")

        # 4. 保存模型
        if _save:
            clf_dir = os.path.join(get_dataset_dir(self.dataset_name), 'model', 'classification')
            ensure_dir(clf_dir)
            clf_file = os.path.join(clf_dir, f'{subject_id}{session}_bayesian_clf.joblib')
            clf.save(clf_file)
            if _verbose:
                print(f"\n分类器已保存至: {clf_file}")

        return clf