"""
分类管道：加载特征 → 训练分类器 → 交叉验证 → 保存模型
支持灵活修改内部参数
"""
import os
import json
import numpy as np
from config import get_feature_path, get_classifier_dir, get_label_path, get_cv_result_dir, ensure_dir
from src.classification.bayesian_classifier import BayesianClassifier
from src.evaluation.evaluator import BCIEvaluator
from src.utils import SessionConfig


class ClassifyPipeline:
    """轻量级分类流水线，支持运行参数和手动修改内部属性"""

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
        # 运行变量：读配置文件，配置文件不存在则用默认值
        cv_folds = self.cfg.get('classify_cv_folds')
        random_state = self.cfg.get('classify_random_state')
        do_cv = self.cfg.get('classify_do_cv')

        # 1. 加载特征+标签
        feature_path = get_feature_path(self.dataset_name, self.subject_id, self.session, 'ovocsp')
        features = np.load(feature_path)
        label_path = get_label_path(self.dataset_name, self.subject_id, self.session)
        labels = np.load(label_path)

        if verbose:
            print(f"✓ 特征加载完成")
            print(f"特征维度: {features.shape}, 类别分布: {np.bincount(labels)[1:]}")

        # 2. 训练分类器（使用当前分类器类）
        clf = self.classifier_class()
        clf.fit(features, labels)
        if verbose:
            print(f"\n✓ {clf.__class__.__name__}分类器已训练")

        # 3. 交叉验证
        if do_cv:
            # 每次 run 根据最新参数创建评估器，保证 cv_folds 和 random_state 同步
            eval_cv = self.evaluator_class(cv_folds=cv_folds, random_state=random_state)
            results = eval_cv.evaluate(features, labels, clf)
            if verbose:
                print(f"\n交叉验证 ({cv_folds}-fold):")
                print(f"  Accuracy: {results['accuracy_mean']:.4f} ± {results['accuracy_std']:.4f}")
                print(f"  Kappa:    {results['kappa_mean']:.4f} ± {results['kappa_std']:.4f}")

        # 4. 保存模型
        if save:
            clf_dir = get_classifier_dir(self.dataset_name)
            ensure_dir(clf_dir)
            clf_file = os.path.join(clf_dir, f'{self.subject_id}{self.session}_bayesian_clf.joblib')
            clf.save(clf_file)
            if verbose:
                print(f"\n✓ 分类器已保存至: {clf_file}")

        # 5. 保存交叉验证结果
        if save and do_cv:
            cv_results_dir = get_cv_result_dir(self.dataset_name)
            ensure_dir(cv_results_dir)
            cv_results_path = os.path.join(cv_results_dir, f'{self.subject_id}{self.session}_bayesian_cv_results.json')
            # 保存为 JSON
            save_data = {
                'dataset': self.dataset_name,
                'subject_id': self.subject_id,
                'session': self.session,
                'cv_folds': cv_folds,
                'random_state': random_state,
                'accuracy_mean': round(results['accuracy_mean'], 4),
                'accuracy_std': round(results['accuracy_std'], 4),
                'kappa_mean': round(results['kappa_mean'], 4),
                'kappa_std': round(results['kappa_std'], 4),
            }

            with open(cv_results_path, 'w') as f:
                json.dump(save_data, f, indent=2, ensure_ascii=False)
                if verbose:
                    print(f"✓ 模型评估结果已保存至: {cv_results_path}")

        return clf