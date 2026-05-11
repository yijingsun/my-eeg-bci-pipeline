import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import cohen_kappa_score, make_scorer

class BCIEvaluator:
    """独立评估器，对任意特征和任意分类器进行 K-Fold 评估"""
    def __init__(self, cv_folds: int, random_state: int):
        self.cv_folds = cv_folds if cv_folds else 5
        self.random_state = random_state if random_state else 17

    def evaluate(self, features, y, classifier):
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True,
                             random_state=self.random_state)
        # 准确率
        acc_scores = cross_val_score(classifier, features, y, cv=cv,
                                     scoring='accuracy')
        # Kappa
        kappa_scorer = make_scorer(cohen_kappa_score)
        kappa_scores = cross_val_score(classifier, features, y, cv=cv,
                                       scoring=kappa_scorer)
        return {
            'accuracy_mean': np.mean(acc_scores),
            'accuracy_std': np.std(acc_scores),
            'kappa_mean': np.mean(kappa_scores),
            'kappa_std': np.std(kappa_scores),
            'cv_scores_accuracy': acc_scores,
            'cv_scores_kappa': kappa_scores
        }