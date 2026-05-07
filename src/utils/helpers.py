"""工具函数"""
import sys
import os
import json
import mne
from src.utils.constants import DEFAULT_CHANNEL_RENAMING
from config import get_dataset_dir


# def setup_project_path():
#     this_file = os.path.abspath(__file__)
#     project_root = os.path.dirname(os.path.dirname(os.path.dirname(this_file)))
#     if project_root not in sys.path:
#         sys.path.insert(0, project_root)


def set_montage_gdf(raw: mne.io.Raw, montage_name: str = 'standard_1020', channel_rename: dict = DEFAULT_CHANNEL_RENAMING) -> mne.io.Raw:
    print(f"原始通道: {raw.ch_names}")

     # 重命名
    raw = raw.copy()
    renamer = {}
    for ch_name in raw.ch_names:
        if ch_name in channel_rename:
            renamer[ch_name] = channel_rename[ch_name]
    if renamer:
        raw.rename_channels(renamer)

    print(f"重命名后通道: {raw.ch_names}")
    montage = mne.channels.make_standard_montage(montage_name)
    raw.set_montage(montage, on_missing='warn', verbose=False)
    return raw