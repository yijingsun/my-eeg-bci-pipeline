"""
常量单元测试
"""
from src.utils.constants import DEFAULT_CHANNEL_RENAMING, DEFAULT_MONTAGE


class TestChannelRenaming:

    def test_keys_are_gdf_format(self):
        """所有 key 应以 EEG- 开头"""
        for k in DEFAULT_CHANNEL_RENAMING:
            assert k.startswith("EEG-")

    def test_values_are_10_20_labels(self):
        """value 应是标准 10-20 标签"""
        valid_labels = {
            "Fz", "C3", "Cz", "C4", "Pz",
            "FC5", "FC1", "FC2", "FC6",
            "C5", "C1", "C2", "C6",
            "CP5", "CP1", "CP2", "CP6",
            "P5", "P1", "P2", "P6",
            "POz",
        }
        for v in DEFAULT_CHANNEL_RENAMING.values():
            assert v in valid_labels

    def test_mapping_count(self):
        """应有 22 个通道映射"""
        assert len(DEFAULT_CHANNEL_RENAMING) == 22


class TestDefaultMontage:

    def test_is_string(self):
        assert isinstance(DEFAULT_MONTAGE, str)

    def test_is_recognized_montage(self):
        assert DEFAULT_MONTAGE in ("standard_1020", "standard_1005",
                                   "standard_1010", "standard_1020")
