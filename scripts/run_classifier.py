#!/usr/bin/env python3
"""
运行分类管道
用法:
    cd /path/to/my-eeg-bci-pipeline
    python scripts/run_classifier.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline.classify_pipeline import TrainClassifierPipeline

if __name__ == '__main__':

    DATASET = 'BCICIV_2a'
    SUBJECT_ID = 'A01'
    SESSION = 'T'
    pipeline = TrainClassifierPipeline(dataset_name=DATASET, subject_id=SUBJECT_ID, session=SESSION)

    # 单个被试
    pipeline.run(verbose=True)