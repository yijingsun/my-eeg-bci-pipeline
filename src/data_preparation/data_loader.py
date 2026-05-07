"""
数据加载器
读取 GDF 文件，设置通道类型和蒙太奇
"""
import mne
from src.utils.session_config import SessionConfig
from src.utils.helpers import set_montage_gdf


class BCIDataLoader:
    """BCI 竞赛 GDF 数据加载器"""

    def __init__(self, config: SessionConfig):
        self.config = config
        self._raw = None

    def load(self, filepath: str, set_montage: bool = True) -> mne.io.Raw:
        """
        加载 GDF 文件

        步骤:
            1. 读取 GDF 文件
            2. 标记 EOG 通道类型
            3. 重命名通道
            4. 设置标准 10-20 蒙太奇

        参数
        ----
        filepath : str
        set_montage : bool  是否自动设置蒙太奇

        返回
        ----
        raw : mne.io.Raw
        """
        raw = mne.io.read_raw_gdf(input_fname=filepath, eog=self.config.eog_channels, preload=True)
        for ch in self.config.eog_channels:
            if ch in raw.ch_names:
                raw.set_channel_types({ch: 'eog'})

        # 设置蒙太奇
        if set_montage:
            raw = set_montage_gdf(raw)
        
        self._raw = raw
        return raw
    
    @property
    def raw(self):
        return self._raw