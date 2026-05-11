import sys
import os

# 将项目根目录加入 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import feature_pipeline
from src.pipeline.preprocess_pipeline import DataPipeline
from src.pipeline.feature_pipeline import TrainOVOCspFeaturePipeline
from src.pipeline.classify_pipeline import TrainClassifierPipeline



if __name__ == '__main__':

    DATASET = 'BCICIV_2a'
    SUBJECT_ID = 'A01'
    SESSION = 'T'
    pre_pipeline = DataPipeline(dataset_name=DATASET, subject_id=SUBJECT_ID, session=SESSION)
    feature_pipeline = TrainOVOCspFeaturePipeline(dataset_name=DATASET, subject_id=SUBJECT_ID, session=SESSION)
    classify_pipeline = TrainClassifierPipeline(dataset_name=DATASET, subject_id=SUBJECT_ID, session=SESSION)

    # 单个被试
    pre_pipeline.run()
    feature_pipeline.run()
    classify_pipeline.run()