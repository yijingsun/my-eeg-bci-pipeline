"""
ICA 伪迹去除器
自动识别并去除眼电等生理伪迹
"""
import mne
import numpy as np

class ArtifactRemover:
    """ICA 眼电伪迹去除"""

    def __init__(self, ica_n_components: int = 20, ica_random_state: int = 71, ica_method: str = 'fastica', ica_exclude_manual=None):
        self.ica_n_components = ica_n_components
        self.ica_random_state = ica_random_state
        self.ica_method = ica_method
        self.ica_exclude_manual = ica_exclude_manual if ica_exclude_manual is not None else []

        # after fit ica
        self.ica_ = None
        self.auto_exclude_ = [] 

    def fit(self, raw: mne.io.Raw) -> 'ArtifactRemover':
        """在粗滤波+重参考后的数据上拟合 ICA"""
        self.ica_ = mne.preprocessing.ICA(
            n_components=self.ica_n_components,
            random_state=self.ica_random_state,
            max_iter='auto',
            method=self.ica_method
        )
        self.ica_.fit(raw)
        return self

    def find_auto_artifacts(self, raw: mne.io.Raw) -> list:
        """自动检测眼电成分,保存并返回"""
        if self.ica_ is None:
            raise RuntimeError("ICA 尚未拟合，请先调用 fit()")
        eog_indices, _ = self.ica_.find_bads_eog(raw, verbose=False)
        self.auto_exclude_ = list(eog_indices)
        return self.auto_exclude_
    
    def get_all_artifacts(self) -> np.ndarray:
        """返回所有标记的伪迹成分（自动+手工）"""
        manual_exclude_ = self.ica_exclude_manual
        return np.unique(self.auto_exclude_ + manual_exclude_)

    def apply(self, raw: mne.io.Raw, exclude: np.ndarray) -> mne.io.Raw:
        """从 raw 中去除指定 ICA 成分"""
        if self.ica_ is None:
            raise RuntimeError("ICA 尚未拟合，请先调用 fit()")
        if exclude is None or exclude.size == 0:
            exclude = self.get_all_artifacts()
        return self.ica_.apply(raw.copy(), exclude=exclude)
    
    def get_params(self) -> str:
        return {
            'ica_n_components': self.ica_n_components,
            'ica_random_state': self.ica_random_state,
            'ica_method': self.ica_method,
            'auto_exclude': self.auto_exclude_,
            'manual_exclude': self.ica_exclude_manual,
            'all_exclude': self.get_all_artifacts()
        }
