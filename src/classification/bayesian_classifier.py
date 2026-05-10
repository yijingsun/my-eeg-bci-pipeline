import numpy as np
from scipy.linalg import pinv
from sklearn.base import BaseEstimator, ClassifierMixin
import joblib

"""
继承BaseEstimator和ClassifierMixin类, 获得标准接口fit, predict, predict_proba, 方便使用scikit-learn的交叉验证工具cross_val_score
"""
class BayesianClassifier(BaseEstimator, ClassifierMixin):
    """
    贝叶斯分类器（共享协方差矩阵，与LDA分类器等价）
    假设 P(x|y) ~ N(mu_y, Sigma)
    """
    def __init__(self):
        self.classes_ = None
        self.priors_ = None
        self.means_ = None
        self.shared_cov_ = None
        self.inv_cov_ = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]

        # 先验概率
        self.priors_ = np.zeros(n_classes)
        self.means_ = np.zeros((n_classes, n_features))
        cov_sum = np.zeros((n_features, n_features))

        for idx, c in enumerate(self.classes_):
            X_c = X[y == c]
            self.priors_[idx] = len(X_c) / len(X)
            self.means_[idx] = np.mean(X_c, axis=0)
            cov_sum += np.cov(X_c, rowvar=False) * (len(X_c) - 1)

        self.shared_cov_ = cov_sum / (len(X) - n_classes)
        self.inv_cov_ = pinv(self.shared_cov_) # 取伪逆矩阵，保证协方差矩阵可逆
        return self

    def predict(self, X):
        scores = self._discriminant_scores(X)
        return self.classes_[np.argmax(scores, axis=1)]

    def predict_proba(self, X):
        scores = self._discriminant_scores(X)
        # softmax 转为概率
        scores -= np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(scores)
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    def _discriminant_scores(self, X):
        scores = np.zeros((X.shape[0], len(self.classes_)))
        for idx in range(len(self.classes_)):
            diff = X - self.means_[idx]
            scores[:, idx] = (
                np.log(self.priors_[idx])
                - 0.5 * np.sum(diff @ self.inv_cov_ * diff, axis=1)
            )
        return scores

    # ---------- 持久化方法 ----------
    def save(self, filepath: str):
        """保存分类器到文件"""
        state = {
            'classes_': self.classes_,
            'priors_': self.priors_,
            'means_': self.means_,
            'shared_cov_': self.shared_cov_,
            'inv_cov_': self.inv_cov_
        }
        joblib.dump(state, filepath)

    @staticmethod
    def load(filepath: str):
        """从文件加载分类器"""
        state = joblib.load(filepath)
        clf = BayesianClassifier()
        clf.classes_ = state['classes_']
        clf.priors_ = state['priors_']
        clf.means_ = state['means_']
        clf.shared_cov_ = state['shared_cov_']
        clf.inv_cov_ = state['inv_cov_']
        print(f"分类器已加载: {filepath}")
        return clf