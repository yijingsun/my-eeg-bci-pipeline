#!/usr/bin/env python3
"""运行 OVO-CSP 特征提取"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.feature_extraction.feature_pipeline import FeatureExtractionPipeline

if __name__ == '__main__':
    pipeline = FeatureExtractionPipeline('BCICIV_2a')
    features, extractor = pipeline.run(subject_id='A01', session='T')