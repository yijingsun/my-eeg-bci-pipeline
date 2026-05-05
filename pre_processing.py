from load_data import load_raw_gdf

import mne
import numpy as np
import os

file_dir = "data/external/bciciv_2a_gdf/train"
file_name = "A01T"
file_type = "gdf"
file_path = f"{file_dir}/{file_name}.{file_type}"
DEFAULT_EOG_CHANNELS = ["EOG-left", "EOG-central", "EOG-right"]
raw = load_raw_gdf(file_path=file_path, eog_channels=DEFAULT_EOG_CHANNELS)

# down-sample到128Hz,减少数据量,加快后续处理速度
resampled_raw = raw.copy()
resampled_raw.resample(128, npad= "auto", verbose=False)

# 标记坏通道
bad_channels_old = raw.info["bads"]
print(f"原始标记的坏通道: {bad_channels_old}")

bad_channels_manual = [] # 这里可以添加人工标记的坏通道, 例如: ['Fz', 'Cz']
print(f"人工标记的坏通道: {bad_channels_manual}")

bad_channels = bad_channels_old + bad_channels_manual
raw.info["bads"] = bad_channels
print(f"最终标记的坏通道: {bad_channels}")

# 插值修复坏通道
fixed_raw = resampled_raw.copy()

if bad_channels:
    fixed_raw.interpolate_bads(reset_bads=True, verbose=False) # 默认球形样条插值(spherical spline)
    print("坏通道已插值修复完成!")
else:
    print("没有标记坏通道,无需插值修复.")


# 进行ICA预处理:粗滤波和重参考
ica_raw = fixed_raw.copy()

low_freq_first, high_freq_first = 0.5, 50 # (低通需要小于采样率的一半,128Hz/2=64Hz)带通滤波,保留0.5-50Hz的频段,去除慢漂移和高频噪声,去直流/极高频、防混叠
ica_raw.filter(l_freq=low_freq_first, h_freq=high_freq_first, fir_design='firwin', verbose=False)
print(f"粗滤波完成: {low_freq_first}-{high_freq_first}Hz")

mne.set_eeg_reference(ica_raw, 'average', verbose=False)
print("重参考完成: 平均参考")

# 进行ICA分解
ica = mne.preprocessing.ICA(
    n_components=20, # 一般10-20个
    random_state=71, # 固定随机种子以保证结果可复现
    max_iter="auto",
    method="fastica", # 默认fastica算法, 其他选项还有infomax、picard等, 可以根据需要选择
)

ica.fit(ica_raw, verbose=False)

print(f"ICA 分解完成!\nICA 分解结果: \n{ica}")

# 标记伪迹成分
eog_indices, eog_scores = list(ica.find_bads_eog(ica_raw, verbose=False))
print(f"自动识别的眼电伪迹成分: {eog_indices}")

artifacts_indices_manual = []  # 人工标记的伪迹成分, 例如: [5, 10]
print(f"人工标记的伪迹成分: {artifacts_indices_manual}")

artifacts_indices = np.array(eog_indices + artifacts_indices_manual)
print(f"最终标记的伪迹成分: {artifacts_indices}")

# 从原始数据中去除伪迹成分
cleaned_raw = ica_raw.copy()
cleaned_raw = ica.apply(cleaned_raw , exclude=artifacts_indices)
print("已去除伪迹成分,得到清洁数据!")

# 提取事件
print("提取事件数据...")
events, event_ids = mne.events_from_annotations(cleaned_raw, verbose=False) # 提取event和event_id数据
print("原始 events:", len(events))

# 对events去重
events = np.unique(events, axis=0)
print("去重 events:", len(events))

event_to_id = dict({'769': 7, '770': 8, '771': 9, '772': 10}) # MI事件ID映射,只有第四个被试映射成5678
events_mi = mne.pick_events(events, include=list(event_to_id.values())) # pick MI events

# 映射4类MI任务事件ID到1-4
events_mi[:, 2] = np.array([list(event_to_id.values()).index(e) + 1 for e in events_mi[:, 2]]) # 将事件ID映射到1-4, Cue on 1-left hand, 2-right hand, 3-both feet, 4-tongue
print("MI events:", len(events_mi))
print(events_mi[:5]) # 打印前5个事件检查

assert len(events_mi) == 288, f"MI trial 数异常: {len(events_mi)}" # 288是每个被试的MI trial数量, 4类任务每类72个
print("事件提取完成!")

# 精滤波,减少与 MI 关系较弱的频段,提高信噪比,提高后序模型泛化与稳定性
# μ 节律(Mu rhythm): 大约 8–13 Hz, 主要分布在感觉运动皮层(C3/Cz/C4 附近), 在运动/运动想象时出现 事件相关去同步化(ERD)
# β 节律(Beta rhythm): 大约 13–30 Hz, 同样与感觉运动皮层相关, 常在运动想象或运动准备/执行阶段出现 ERD/ERS 变化
low_freq_second, high_freq_second = 8, 30
model_raw = cleaned_raw.copy().filter(low_freq_second, high_freq_second, fir_design='firwin', verbose=False)
print(f"精滤波完成: {low_freq_second}-{high_freq_second}Hz")

# 提取MI epochs (从事件中提取每个MI trial的EEG数据段)
# 这里选择从事件开始后1s到4s的时间窗口, 持续3s, 因为MI任务通常在Cue出现后1s左右开始, 这样可以更好地捕捉MI相关的脑电活动
# 在trial中的绝对时间3.25s-6.25s
print("提取MI epochs...")
epochs = mne.Epochs(
    model_raw,
    events_mi,
    tmin=1.0,
    tmax=4.0,
    baseline=None,
    preload=True,
    verbose=False
)

# drop eog channels from epochs, 因为我们已经通过ICA去除了眼电伪迹, 这里不需要保留EOG通道了
epochs.drop_channels(DEFAULT_EOG_CHANNELS)
print("eog channels dropped from epochs!")


""""
ICA 必须在“所有通道存在”的前提下工作, 需要识别该通道上的噪声源, 构造对应的空间投影矩阵, 才能有效去除伪迹, 
如果一开始就去除坏通道, 可能会导致 ICA 无法正确识别和分离伪迹成分, 从而影响后续的伪迹去除效果
"""
# drop bad channels from epochs, 因为我们已经通过插值修复了坏通道, 这里不需要保留坏通道了
epochs.drop_channels(model_raw.info['bads'])
print("bad channels dropped from epochs!")

# pick channels of interest (感觉运动皮层附近的通道)
# mi_channels = [
#     'C3','Cz','C4',
#     'FC3','FCz','FC4',
#     'CP3','CPz','CP4'
# ]

# epochs.pick(mi_channels)
print("提取MI epochs完成!")
print("epochs shape:", epochs.get_data().shape) # (n_trial, n_channel, n_sample), n_sample=385, 因为3s * 128Hz(sfreq)=384, 加上起始点0s的样本就是385
print("event distribution:", np.bincount(epochs.events[:, 2])) # [ 0 72 72 72 72], 因为我们有4类MI任务, 每类72个trial, 事件ID从1-4, 所以0类没有trial
print("预处理完成!")

# 保存预处理后的epochs数据, 供后续分析和建模使用
epoch_file_dir = file_dir + "/epochs"

if not os.path.exists(epoch_file_dir):
    os.makedirs(epoch_file_dir)

epoch_file_name = file_name + "_epo"
epoch_file_type = "fif"
epoch_file_path = f"{epoch_file_dir}/{epoch_file_name}.{epoch_file_type}"

epochs.save(
    fname=epoch_file_path,
    overwrite=True
)

print(f"epochs 数据已保存到: {epoch_file_path}")