#!/usr/bin/env python3
"""
将官方 .mat 格式的真实标签转换为 .npy 格式，方便后续快速加载。
处理训练集（T）和测试集（E）的标签。
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.io import loadmat
from config import get_label_dir, ensure_dir

DATASET = 'BCICIV_2a'
SUBJECT_IDS = ['A01', 'A02', 'A03', 'A04', 'A05', 'A06', 'A07', 'A08', 'A09']
SESSIONS = ['T', 'E']  # 训练集和测试集都转换

label_dir = get_label_dir(DATASET)

print(f"从 {label_dir} 读取 .mat 标签...")
print("-" * 50)

for subject in SUBJECT_IDS:
    for session in SESSIONS:
        mat_file = os.path.join(label_dir, f'{subject}{session}.mat')
        npy_file = os.path.join(label_dir, f'{subject}{session}_labels.npy')

        if not os.path.exists(mat_file):
            print(f"⚠ {subject}{session}.mat 不存在，跳过")
            continue

        # 加载 .mat 文件
        mat = loadmat(mat_file)
        # 变量名通常是 'classlabel'
        labels = mat['classlabel'].flatten().astype(int)

        # 保存为 .npy
        np.save(npy_file, labels)
        print(f"✓ {subject}{session}: {labels.shape}, 类别: {np.unique(labels)} → 已保存为 .npy")

print("-" * 50)
print("转换完成！现在可以直接用 np.load 读取标签了。")