import mne
from config import get_raw_path
from src.utils.constants import DEFAULT_CHANNEL_RENAMING, DEFAULT_MONTAGE

class BCIDataLoader:
    def __init__(self, 
                 eog_channels: list = None, 
                 set_montage = False, 
                 channel_renaming: dict = None, 
                 montage_name: str =None):
        self.eog_channels = eog_channels if eog_channels is not None else []
        self.set_montage = set_montage
        self.channel_renaming = channel_renaming if channel_renaming is not None else DEFAULT_CHANNEL_RENAMING
        self.montage_name = montage_name if montage_name is not None else DEFAULT_MONTAGE
        self._raw = None

    def load(self, filepath: str, verbose: bool = False) -> mne.io.Raw:
        raw = mne.io.read_raw_gdf(filepath, eog=self.eog_channels, preload=True, verbose=False)
        for ch in self.eog_channels:
            if ch in raw.ch_names:
                raw.set_channel_types({ch: 'eog'}, verbose=False)
        if self.set_montage:
            raw.rename_channels(self.channel_renaming, verbose=False)
            montage = mne.channels.make_standard_montage(self.montage_name)
            raw.set_montage(montage, on_missing='warn', verbose=False)
            if verbose:
                print(f"已设置 {self.montage_name} montage, 通道名称为: {raw.ch_names}")
        self._raw = raw
        return raw

    @property
    def raw(self):
        return self._raw

    def get_params(self) -> dict:
        return {
            'eog_channels': self.eog_channels,
            'set_montage': self.set_montage,
            'channel_renaming': self.channel_renaming,
            'montage_name': self.montage_name,
        }