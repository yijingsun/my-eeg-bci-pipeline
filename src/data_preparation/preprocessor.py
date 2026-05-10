"""
完整预处理器：降采样 → 坏道修复 → 重参考 → (可选ICA滤波) → (可选ICA去除) → MI滤波
整合了原 EEGPreprocessor 与 ArtifactRemover 的功能
"""
import mne
import numpy as np


class EEGPreprocessor:
    def __init__(
        self,
        resample_freq: int = None,
        filter_ica: list = None,
        filter_mi: list = None,
        ref_type: str = None,
        bad_channels_manual: list = None,
        # ICA 参数
        ica_n_components: int = None,
        ica_random_state: int = None,
        ica_method: str = None,
        ica_exclude_manual: list = None,
    ):
        # 基础参数
        self.resample_freq = resample_freq if resample_freq is not None else 250
        self.filter_ica = filter_ica if filter_ica is not None else [0.5, 50]
        self.filter_mi = filter_mi if filter_mi is not None else [8, 30]
        self.ref_type = ref_type if ref_type is not None else 'average'
        self.bad_channels_manual = bad_channels_manual if bad_channels_manual is not None else []

        # ICA 相关
        self.ica_n_components = ica_n_components if ica_n_components is not None else 20
        self.ica_random_state = ica_random_state if ica_random_state is not None else 71
        self.ica_method = ica_method
        self.ica_exclude_manual = ica_exclude_manual if ica_exclude_manual is not None else []
        
        # 运行时填充
        self.ica_ = None
        self.auto_exclude_ = []

    # ===================== 基础预处理步骤 =====================
    def resample(self, raw: mne.io.Raw, verbose: bool = False) -> mne.io.Raw:
        raw = raw.copy()
        if self.resample_freq is not None and self.resample_freq != raw.info['sfreq']:
            raw.resample(self.resample_freq, npad='auto', verbose=False)
            if verbose:
                print(f'    ✓ 已完成重采样, 采样率: {raw.info["sfreq"]} Hz')
        return raw

    def fix_bad_channels(self, raw: mne.io.Raw, verbose: bool = False) -> mne.io.Raw:
        raw = raw.copy()
        auto_bads = raw.info['bads']
        all_bads = list(set(auto_bads + self.bad_channels_manual))
        raw.info['bads'] = all_bads
        if all_bads:
            raw.interpolate_bads(reset_bads=True, verbose=False)
            if verbose:
                print(f'    ✓ 已修复坏道: {all_bads}')
        return raw

    def apply_reference(self, raw: mne.io.Raw, verbose: bool = False) -> mne.io.Raw:
        raw = raw.copy()
        if self.ref_type is not None:
            raw.set_eeg_reference(ref_channels=self.ref_type, verbose=False)
            if verbose:
                print(f'    ✓ 已完成重参考, 参考方法: {self.ref_type}')
        return raw

    def apply_ica_filter(self, raw: mne.io.Raw, verbose: bool = False) -> mne.io.Raw:
        """用于 ICA 的宽频带通滤波（粗滤波）"""
        raw = raw.copy()
        if self.filter_ica is not None:
            raw.filter(l_freq=self.filter_ica[0], h_freq=self.filter_ica[1],
                       fir_design='firwin', verbose=False)
            if verbose:
                print(f'    ✓ 带通滤波, 频段: {self.filter_ica[0]} - {self.filter_ica[1]} Hz')
        return raw

    def apply_mi_filter(self, raw: mne.io.Raw, verbose: bool = False) -> mne.io.Raw:
        """运动想象频段滤波（精滤波）"""
        raw = raw.copy()
        if self.filter_mi is not None:
            raw.filter(l_freq=self.filter_mi[0], h_freq=self.filter_mi[1],
                       fir_design='firwin', verbose=False)
            if verbose:
                print(f'    ✓ 带通滤波, 频段: {self.filter_mi[0]} - {self.filter_mi[1]} Hz')
        return raw

    # ===================== ICA 相关方法 =====================
    def fit_ica(self, raw: mne.io.Raw, verbose: bool = False) -> 'EEGPreprocessor':
        """拟合 ICA（需要事先用 apply_ica_filter 准备好数据）"""
        if self.ica_n_components is None or self.ica_n_components <= 0:
            return self
        self.ica_ = mne.preprocessing.ICA(
            n_components=self.ica_n_components,
            random_state=self.ica_random_state,
            max_iter='auto',
            method=self.ica_method,
            verbose=False
        )
        self.ica_.fit(raw, verbose=False)
        if verbose:
            print(f'    ✓ 已完成 ICA 拟合, 成分数: {self.ica_n_components}')
        return self

    def find_auto_artifacts(self, raw: mne.io.Raw, verbose: bool = False) -> list:
        """自动检测眼电成分"""
        if self.ica_ is None:
            return []
        eog_indices, _ = self.ica_.find_bads_eog(raw, verbose=False)
        self.auto_exclude_ = list(eog_indices)
        if verbose:
            print(f'    ✓ 已自动检测到眼电成分: {self.auto_exclude_}')
        return self.auto_exclude_

    def get_all_artifacts(self) -> np.ndarray:
        """自动 + 手动去重后的全部伪迹成分"""
        manual = self.ica_exclude_manual if self.ica_exclude_manual else []
        return np.unique(self.auto_exclude_ + manual)

    def apply_ica(self, raw: mne.io.Raw, exclude: np.ndarray = None, verbose: bool = False) -> mne.io.Raw:
        """从 raw 中去除指定 ICA 成分"""
        if self.ica_ is None:
            return raw.copy()
        if exclude is None:
            exclude = self.get_all_artifacts()
        if exclude is not None and len(exclude) > 0:
            raw = self.ica_.apply(raw.copy(), exclude=exclude, verbose=False)
            if verbose:
                print(f'    手动去除 ICA 成分: {self.ica_exclude_manual}')
                print(f'    总共去除 ICA 成分: {exclude}')
        return raw

    # ===================== 总控流程 =====================
    def process(self, raw: mne.io.Raw, verbose: bool = False) -> mne.io.Raw:
        """
        一键执行完整预处理：
        resample → fix bads → reference → (ICA filter) → (ICA fit & remove) → MI filter
        resample 步骤在 resample_freq 为 None 时自动跳过
        reference 步骤在 ref_type 为 None 时自动跳过
        ICA 步骤在 ica_method 为 None 时自动跳过
        """
        # 1. 降采样
        if self.resample_freq is not None:
            raw = self.resample(raw, verbose=verbose)

        # 2. 坏通道插值
        raw = self.fix_bad_channels(raw, verbose=verbose)

        # 3. 重参考
        if self.ref_type is not None:
            raw = self.apply_reference(raw, verbose=verbose)

        # 4. ICA 滤波（粗滤波）—— 即使不做 ICA，也可能想保留宽频信号？
        #    这里依然保留滤波，但如果你直接做 8-30 Hz 可把 filter_ica 设为 None
        if self.filter_ica is not None:
            raw = self.apply_ica_filter(raw, verbose=verbose)

        # 5. ICA 拟合与去除（可选）
        if self.ica_method is not None:
            self.fit_ica(raw, verbose=verbose)
            # 对 ICA 滤波后的数据检测眼电
            self.find_auto_artifacts(raw, verbose=verbose)
            raw = self.apply_ica(raw, verbose=verbose)

        # 6. MI 频段滤波
        if self.filter_mi is not None:
            raw = self.apply_mi_filter(raw, verbose=verbose)

        return raw

    # ===================== 状态查询 =====================
    def get_params(self) -> dict:
        return {
            'resample_freq': self.resample_freq,
            'filter_ica': self.filter_ica,
            'filter_mi': self.filter_mi,
            'ref_type': self.ref_type,
            'bad_channels_manual': self.bad_channels_manual,
            'ica_n_components': self.ica_n_components,
            'ica_random_state': self.ica_random_state,
            'ica_method': self.ica_method,
            'ica_exclude_manual': self.ica_exclude_manual,
            'auto_exclude': self.auto_exclude_,
            'all_exclude': self.get_all_artifacts().tolist()
        }