# 开发指南

本文档说明如何扩展本项目的功能，包括添加新数据集、新特征提取器和新分类器。

---

## 1. 添加新数据集

### 1.1 目录结构

在 `data/` 下新建以数据集命名的目录，结构与现有数据集保持一致：

```
data/
└── <new_dataset>/
    ├── config.json      ← 实验参数配置（必需）
    ├── raw/             ← 原始数据文件
    ├── epochs/          ← 预处理后 epochs
    ├── label/           ← 标签文件
    ├── feature/         ← 特征矩阵 + 提取器
    ├── classifier/      ← 训练好的分类器
    └── result/          ← 评估结果
```

### 1.2 配置 config.json

复制 `example_config.json` 并调整参数：

```json
{
  "default": {
    "eog_channels": [...],
    "tmin": 0.0,
    "tmax": 3.0,
    "mi_event_mapping": {"<event_id>": <class_label>, ...},
    "expected_trials": <数量>,
    ...
  },
  "<subject_id>": {
    "T": { ... },
    "E": { ... }
  }
}
```

### 1.3 添加数据加载支持

若新数据集的原始文件格式不是 `.gdf`，需在 `BCIDataLoader.load()` 中添加对应的读取逻辑：

```python
# src/data_preparation/data_loader.py
def load(self, filepath: str, verbose: bool = False):
    if filepath.endswith('.gdf'):
        raw = mne.io.read_raw_gdf(...)
    elif filepath.endswith('.edf'):
        raw = mne.io.read_raw_edf(...)
    # ...
```

### 1.4 更新路径配置

若目录结构与现有不同，在 `config.py` 中添加对应的路径函数。

---

## 2. 添加新特征提取器

### 2.1 实现接口

新特征提取器需实现以下方法，参考 `OVOCspFeatureExtractor`：

```python
class NewFeatureExtractor:
    def __init__(self, **params):
        # 初始化参数
        ...

    def fit(self, eeg_epochs_array, event_labels, verbose=True):
        """
        训练特征提取器

        Args:
            eeg_epochs_array: ndarray, shape (n_trials, n_channels, n_times)
            event_labels: ndarray, shape (n_trials,)
        """
        ...
        return self

    def transform(self, eeg_epochs_array):
        """
        对新数据提取特征

        Returns:
            features: ndarray, shape (n_trials, n_features)
        """
        ...

    def fit_transform(self, eeg_epochs_array, event_labels, verbose=True):
        """训练 + 转换"""
        self.fit(eeg_epochs_array, event_labels, verbose=verbose)
        return self.transform(eeg_epochs_array)

    def save(self, filepath: str):
        """持久化模型"""
        ...

    @staticmethod
    def load(filepath: str):
        """加载模型"""
        ...

    def get_params(self) -> dict:
        """返回当前参数（用于结果记录）"""
        ...
```

### 2.2 集成到训练脚本

在 `scripts/train.py` 的 `run_feature_extraction()` 函数中替换 extractor 实例化逻辑：

```python
# scripts/train.py
def run_feature_extraction(epoch_path, label_path, feature_dir, cfg, ...):
    extractor = NewFeatureExtractor(...)  # 替换 OVOCspFeatureExtractor
    features = extractor.fit_transform(X, y, verbose=verbose)
    ...
```

### 2.3 更新评估脚本

在 `scripts/evaluate.py` 中添加对新特征提取器的加载和调用逻辑。

---

## 3. 添加新分类器

### 3.1 sklearn 兼容接口

新分类器需继承 `BaseEstimator` 和 `ClassifierMixin`，实现 `fit()` 和 `predict()`：

```python
from sklearn.base import BaseEstimator, ClassifierMixin

class NewClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, **params):
        # 参数必须在 __init__ 中设置（sklearn 约定）
        ...

    def fit(self, X, y):
        """
        训练分类器

        Args:
            X: ndarray, shape (n_samples, n_features)
            y: ndarray, shape (n_samples,)
        """
        self.classes_ = np.unique(y)
        # 训练逻辑
        ...
        return self

    def predict(self, X):
        """
        预测类别

        Returns:
            predictions: ndarray, shape (n_samples,)
        """
        ...
```

### 3.2 可选：predict_proba

若需输出概率，实现 `predict_proba()`：

```python
def predict_proba(self, X):
    """返回类别概率，shape (n_samples, n_classes)"""
    ...
```

### 3.3 持久化方法

实现 `save()` 和 `load()` 方法，参考 `BayesianClassifier`：

```python
def save(self, filepath: str):
    joblib.dump({...}, filepath)

@staticmethod
def load(filepath: str):
    state = joblib.load(filepath)
    clf = NewClassifier()
    # 恢复状态
    ...
    return clf
```

### 3.4 集成到训练脚本

调用 `run_classification()` 时传入新分类器：

```python
run_classification(..., classifier_class=NewClassifier)
```

---

## 4. 代码规范

- **类型注解**：所有 public 方法应标注参数类型和返回类型
- **Docstring**：采用 Google 或 NumPy 风格，包含 Args、Returns、Raises
- **命名**：变量和函数用 `snake_case`，类用 `PascalCase`
- **日志**：当前使用 `print()`，未来计划迁移到 `logging` 模块

---

## 5. 单元测试

### 5.1 目录结构

```
tests/
├── conftest.py                    # 共享夹具（真实/模拟 config.json）
├── utils/
│   ├── test_session_config.py     # SessionConfig 测试
│   └── test_constants.py          # 常量测试
├── data_preparation/              # BCIDataLoader / EEGPreprocessor / EpochProcessor
├── feature_extraction/            # OVOCspFeatureExtractor
├── classification/                # BayesianClassifier
├── evaluation/                    # BCIEvaluator
├── pipeline/                      # 端到端组装测试（@pytest.mark.slow）
└── scripts/                       # 脚本聚合逻辑 + 搜索集成测试
```

测试文件与 `src/` 一一对应。新增 `src/utils/xxx.py` 时，应同步添加 `tests/utils/test_xxx.py`。

### 5.2 运行

```bash
pip install pytest
python -m pytest tests/ -v
```

### 5.3 编写规范

- 使用 `pytest` 框架
- 测试类按功能分组（`TestFromJsonFile`、`TestSave` 等）
- 通过 `tests/conftest.py` 中的 fixture 获取真实/模拟 config.json 路径
- save() 测试必须使用 `temp_config_path` fixture，避免污染真实文件
- 关键覆盖项：默认配置加载、被试覆盖合并、属性/字典访问、错误路径、diff 保存逻辑
