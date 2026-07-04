#!/usr/bin/env python3
"""运行 OVO-CSP 特征提取"""
from src.pipeline.feature_pipeline import TrainOVOCspFeaturePipeline

if __name__ == '__main__':
    DATASET = 'BCICIV_2a'
    SUBJECT_ID = 'A01'
    SESSION = 'T'
    pipeline = TrainOVOCspFeaturePipeline(dataset_name=DATASET, subject_id=SUBJECT_ID, session=SESSION)
    
    # 单个被试
    features, extractor = pipeline.run()