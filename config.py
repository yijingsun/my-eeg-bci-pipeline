"""
项目路径配置
只管理目录结构，不包含任何实验参数
"""
import os

# ============================================================
# 项目根目录
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# 数据根目录
# ============================================================
DATA_ROOT = os.path.join(PROJECT_ROOT, 'data')


# ============================================================
# 路径工具函数
# ============================================================
def get_dataset_dir(dataset_name: str) -> str:
    """获取数据集根目录"""
    return os.path.join(DATA_ROOT, dataset_name)


def get_raw_dir(dataset_name: str) -> str:
    """原始数据目录"""
    return os.path.join(get_dataset_dir(dataset_name), 'raw')

def get_epoch_dir(dataset_name: str) -> str:
    """预处理后 epochs 目录"""
    return os.path.join(get_dataset_dir(dataset_name), 'epochs')

def get_feature_dir(dataset_name: str) -> str:
    """特征目录"""
    return os.path.join(get_dataset_dir(dataset_name), 'feature')

def get_classifier_dir(dataset_name: str) -> str:
    """分类器目录"""
    return os.path.join(get_dataset_dir(dataset_name), 'classifier')

def get_cv_result_dir(dataset_name: str) -> str:
    """交叉验证结果目录"""
    return os.path.join(get_dataset_dir(dataset_name), 'result')

def get_label_dir(dataset_name: str) -> str:
    """标签目录"""
    return os.path.join(get_dataset_dir(dataset_name), 'label')

def get_raw_path(dataset_name: str, subject_id: str, session: str,
                 file_type: str = 'gdf') -> str:
    """原始数据文件完整路径"""
    return os.path.join(get_raw_dir(dataset_name), f"{subject_id}{session}.{file_type}")


def get_epoch_path(dataset_name: str, subject_id: str, session: str) -> str:
    """epochs 文件完整路径"""
    return os.path.join(get_epoch_dir(dataset_name), f"{subject_id}{session}_epo.fif")

def get_feature_path(dataset_name: str, subject_id: str, session: str, extractor_name: str = 'ovocsp') -> str:
    """特征文件完整路径"""
    return os.path.join(get_feature_dir(dataset_name), f"{subject_id}{session}_{extractor_name}_features.npy")

def get_extractor_path(dataset_name: str, subject_id: str, session: str, extractor_name: str = 'ovocsp') -> str:
    """特征提取器文件完整路径"""
    return os.path.join(get_feature_dir(dataset_name), f"{subject_id}{session}_{extractor_name}_extractor.joblib")

def get_classifier_path(dataset_name: str, subject_id: str, session: str, classifier_name: str = 'bayesian') -> str:
    """分类器文件完整路径"""
    return os.path.join(get_classifier_dir(dataset_name), f"{subject_id}{session}_{classifier_name}_clf.joblib")

def get_cv_result_path(dataset_name: str, subject_id: str, session: str, classifier_name: str = 'bayesian') -> str:
    """交叉验证结果文件完整路径"""
    return os.path.join(get_cv_result_dir(dataset_name), f"{subject_id}{session}_{classifier_name}_cv_results.json")
    

def get_label_path(dataset_name: str, subject_id: str, session: str) -> str:
    """标签文件完整路径"""
    return os.path.join(get_label_dir(dataset_name), f"{subject_id}{session}_labels.npy")

def ensure_dir(path: str):
    """确保目录存在"""
    os.makedirs(path, exist_ok=True)