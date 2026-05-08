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

from src.classification.classify_pipline import ClassifyPipeline

if __name__ == '__main__':

    DATASET = 'BCICIV_2a'
    pipeline = ClassifyPipeline(dataset_name=DATASET)

    # 单个被试
    pipeline.run('A01', 'T', verbose=True)