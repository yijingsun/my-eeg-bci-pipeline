# 架构概览

## 整体数据流

```
raw/*.gdf
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  DataPipeline (preprocess_pipeline.py)                      │
│                                                             │
│  BCIDataLoader          加载 .gdf，标记 EOG，设置 montage    │
│       ↓                                                     │
│  EEGPreprocessor        降采样 → 坏道修复 → 重参考           │
│                          → ICA滤波/去伪迹 → MI频段滤波        │
│       ↓                                                     │
│  EpochProcessor         事件提取 → 筛选MI → 重映射 → 分段    │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
epochs/*.fif  +  label/*_labels.npy
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  TrainOVOCspFeaturePipeline (feature_pipeline.py)           │
│                                                             │
│  OVOCspFeatureExtractor                                     │
│    1. 每对类别训练 CSP → 空间滤波                             │
│    2. log方差特征 → 拼接所有配对                              │
│    3. (可选) StandardScaler 标准化                            │
│    4. (可选) LDA 降维                                        │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
feature/*_ovocsp_features.npy  +  *_ovocsp_extractor.joblib
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  TrainClassifierPipeline (classify_pipeline.py)             │
│                                                             │
│  BayesianClassifier     全量训练 + 预测                      │
│  BCIEvaluator           K-Fold 交叉验证 (Accuracy + Kappa)   │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
classifier/*_bayesian_clf.joblib  +  result/*_train_results.json
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  batch_evaluate.py（测试集评估）                              │
│                                                             │
│  加载 T 集训练好的 extractor + classifier                    │
│  → 对 E 集 epochs 做 transform → predict                    │
│  → 与官方真实标签对比，计算 Kappa / Accuracy                  │
└─────────────────────────────────────────────────────────────┘
```

---

## 模块职责

| 模块 | 文件 | 职责 | 依赖 |
|------|------|------|------|
| `data_preparation` | `data_loader.py` | 原始数据加载、通道标记 | `mne`, `config` |
| | `pre_processor.py` | 预处理全流程（降采样/滤波/ICA/重参考） | `mne`, `numpy` |
| | `epoch_processor.py` | 事件提取、MI 筛选、分段 | `mne`, `numpy` |
| `feature_extraction` | `ovocsp_feature_extractor.py` | OVO-CSP 特征提取 + LDA 降维 | `mne`, `sklearn`, `joblib` |
| `classification` | `bayesian_classifier.py` | 贝叶斯分类器（sklearn 兼容） | `numpy`, `sklearn`, `joblib` |
| `evaluation` | `evaluator.py` | K-Fold 交叉验证评估 | `sklearn` |
| `pipeline` | `preprocess_pipeline.py` | 编排预处理全流程 | `data_preparation`, `utils` |
| | `feature_pipeline.py` | 编排特征提取流程 | `feature_extraction`, `utils` |
| | `classify_pipeline.py` | 编排分类训练流程 | `classification`, `evaluation`, `utils` |
| `utils` | `constants.py` | 通道映射、montage 等常量 | — |
| | `session_config.py` | 被试级配置读写（config.json） | `config` |
| 根目录 | `config.py` | 全局路径配置（纯函数） | `os` |

---

## 类关系

```
DataPipeline ──uses──▶ BCIDataLoader
     │           ──uses──▶ EEGPreprocessor
     │           ──uses──▶ EpochProcessor
     │           ──uses──▶ SessionConfig

TrainOVOCspFeaturePipeline ──uses──▶ OVOCspFeatureExtractor
     │                       ──uses──▶ SessionConfig

TrainClassifierPipeline ──uses──▶ BayesianClassifier
     │                    ──uses──▶ BCIEvaluator
     │                    ──uses──▶ SessionConfig

SessionConfig ──reads──▶ config.json
```

---

## 配置系统

```
config.py（路径配置，纯函数）
    │
    ▼
data/<dataset>/config.json（实验参数）
    │
    ├── "default": { ... }        ← 所有被试的默认参数
    ├── "A01": { "T": {...}, "E": {...} }  ← 被试级覆盖
    │
    ▼
SessionConfig（运行时合并 default + 覆盖，支持属性访问和 .save()）
```
