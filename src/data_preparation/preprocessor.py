import mne


class EEGPreprocessor:

    def __init__(self, resample_freq: int = None, 
                 filter_ica: list = None, 
                 filter_mi: list = None, 
                 ref_type: str = None, 
                 bad_channels_manual: list = None):
        self.resample_freq = resample_freq if resample_freq is not None else 250
        self.filter_ica = filter_ica if filter_ica is not None else [0.5, 50]
        self.filter_mi = filter_mi if filter_mi is not None else [8, 30]
        self.ref_type = ref_type if ref_type is not None else 'average'
        self.bad_channels_manual = bad_channels_manual if bad_channels_manual is not None else []


    def resample(self, raw: mne.io.Raw) -> mne.io.Raw:
        """降采样"""
        raw = raw.copy()
        if self.resample_freq is not None:
            raw.resample(self.resample_freq, npad='auto')
        return raw

    def fix_bad_channels(self, raw: mne.io.Raw) -> mne.io.Raw:
        raw = raw.copy()
        auto_bads = raw.info['bads']
        all_bads = list(set(auto_bads + self.bad_channels_manual))
        raw.info['bads'] = all_bads
        if all_bads:
            raw.interpolate_bads(reset_bads=True)
        return raw

    def apply_ica_filter_and_ref(self, raw: mne.io.Raw) -> mne.io.Raw:
        raw = raw.copy()
        raw.filter(l_freq=self.filter_ica[0], h_freq=self.filter_ica[1], fir_design='firwin', verbose=False)
        mne.set_eeg_reference(raw, self.ref_type) # re-reference
        return raw

    def apply_mi_filter(self, raw: mne.io.Raw) -> mne.io.Raw:
        raw = raw.copy()
        raw.filter(l_freq=self.filter_mi[0], h_freq=self.filter_mi[1], fir_design='firwin', verbose=False)
        return raw

    def get_params(self) -> dict:
        return {
            'resample_freq': self.resample_freq,
            'filter_ica': self.filter_ica,
            'filter_mi': self.filter_mi,
            'ref_type': self.ref_type,
            'bad_channels_manual': self.bad_channels_manual,
        }