#!/usr/bin/env python3
"""
运行数据预处理管道

用法:
    cd /path/to/my-eeg-bci-pipeline
    python scripts/run_preprocessing.py
"""
import sys
import os

# 将项目根目录加入 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_preparation.pre_pipeline import DataPipeline


if __name__ == '__main__':

    DATASET = 'BCICIV_2a'
    SUBJECT_ID = 'A01'
    SESSION = 'T'
    pipeline = DataPipeline(dataset_name=DATASET, subject_id=SUBJECT_ID, session=SESSION)

    # 单个被试
    epochs = pipeline.run()

    # 批量处理
    # for sid in ['A01', 'A02', 'A03', 'A04', 'A05', 'A06', 'A07', 'A08', 'A09']:
    #     pipeline.run(subject_id=sid, session='T')