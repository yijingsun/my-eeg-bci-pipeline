#!/usr/bin/env python3
"""运行 OVO-CSP 特征提取"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.feature_extraction.feature_pipeline import FeatureExtractionPipeline

if __name__ == '__main__':
    DATASET = 'BCICIV_2a'
    SUBJECT_ID = 'A01'
    SESSION = 'T'
    pipeline = FeatureExtractionPipeline(dataset_name=DATASET, subject_id=SUBJECT_ID, session=SESSION)
    
    # 单个被试
    features, extractor = pipeline.run()