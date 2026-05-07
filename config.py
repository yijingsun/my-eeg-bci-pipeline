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
DATA_ROOT = os.path.join(PROJECT_ROOT, 'data', 'external')


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

def get_label_dir(dataset_name: str) -> str:
    """真实标签目录"""
    return os.path.join(get_dataset_dir(dataset_name), 'truelabel')

def get_model_dir(dataset_name: str) -> str:
    """模型根目录"""
    return os.path.join(get_dataset_dir(dataset_name), 'model')

def get_feature_dir(dataset_name: str) -> str:
    """特征文件目录"""
    return os.path.join(get_model_dir(dataset_name), 'feature')

def get_classifier_dir(dataset_name: str) -> str:
    """分类器模型目录"""
    return os.path.join(get_model_dir(dataset_name), 'classification')

def get_result_dir(dataset_name: str) -> str:
    """评估结果目录"""
    return os.path.join(get_dataset_dir(dataset_name), 'results')

# --- 具体文件路径 ---
def get_raw_path(dataset_name: str, subject_id: str, session: str,
                 file_type: str = 'gdf') -> str:
    """原始数据文件完整路径"""
    return os.path.join(get_raw_dir(dataset_name), f"{subject_id}{session}.{file_type}")

def get_epoch_path(dataset_name: str, subject_id: str, session: str) -> str:
    """epochs 文件完整路径"""
    return os.path.join(get_epoch_dir(dataset_name), f"{subject_id}{session}_epo.fif")

def get_feature_path(dataset_name: str, filename: str) -> str:
    """特征文件完整路径（自动创建目录）"""
    d = get_feature_dir(dataset_name)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, filename)

def get_classifier_path(dataset_name: str, filename: str) -> str:
    """分类器文件完整路径（自动创建目录）"""
    d = get_classifier_dir(dataset_name)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, filename)

def get_result_path(dataset_name: str, filename: str) -> str:
    """结果文件完整路径（自动创建目录）"""
    d = get_result_dir(dataset_name)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, filename)

def get_label_path(dataset_name: str, subject_id: str, session: str) -> str:
    """标签文件完整路径"""
    return os.path.join(get_label_dir(dataset_name), f"{subject_id}{session}_labels.mat")

def ensure_dir(path: str):
    """确保目录存在"""
    os.makedirs(path, exist_ok=True)