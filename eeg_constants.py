"""
10-20国际标准导联系统的通道映射字典
- F = 额, C = 中央, P = 顶, CP = 中央顶, FC = 额中央, PO = 顶后, z = 中线
- 数字偶数 = 右半球, 数字奇数 = 左半球
- 数字越大 = 离中线越远
"""
STANDARD_1020_MAPPING = {
    'EEG-Fz':'Fz', 'EEG-C3':'C3', 'EEG-Cz':'Cz', 'EEG-C4':'C4', 'EEG-Pz':'Pz',
    'EEG-0':'FC5', 'EEG-1':'FC1', 'EEG-2':'FC2', 'EEG-3':'FC6',
    'EEG-4':'C5', 'EEG-5':'C1', 'EEG-6':'C2', 'EEG-7':'C6',
    'EEG-8':'CP5', 'EEG-9':'CP1', 'EEG-10':'CP2', 'EEG-11':'CP6',
    'EEG-12':'P5', 'EEG-13':'P1', 'EEG-14':'P2', 'EEG-15':'P6',
    'EEG-16':'POz'
}
