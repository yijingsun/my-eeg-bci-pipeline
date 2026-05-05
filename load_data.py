import mne
from mne.channels import make_standard_montage
from eeg_constants import STANDARD_1020_MAPPING


def load_raw_gdf(file_path, mapping=STANDARD_1020_MAPPING, eog_channels=None):
    """加载GDF文件并进行预处理
    Args:
        file_path (str): GDF文件路径
        mapping (dict): 导联重命名字典, 默认为STANDARD_1020_MAPPING
        eog_channels (list): EOG通道列表, 默认为None
    Returns:
        raw (mne.io.Raw): 预处理后的原始数据对象
    """
    raw = mne.io.read_raw_gdf(file_path,
                              eog=eog_channels,
                              preload=True,
                              verbose=False)
    raw.rename_channels(mapping)
    montage = make_standard_montage('standard_1020')
    raw.set_montage(montage, on_missing='warn', verbose=False)

    print(f"{file_path}加载成功!")
    print(f"原始数据信息：\n{raw.info}")

    return raw

if __name__ == "__main__":
    load_raw_gdf("data/external/bciciv_2a_gdf/train/A01T.gdf")