# 架构概览

## 整体数据流

```
raw/*.gdf
    │
    ▼
┌──────────────────────────────────────────────────────┐
│  scripts/train.py                                    │
│                                                      │
│  run_preprocessing()                                 │
│    BCIDataLoader → EEGPreprocessor → EpochProcessor  │
│       ↓                                              │
│  run_feature_extraction()                            │
│    OVOCspFeatureExtractor                            │
│       ↓                                              │
│  run_classification()                                │
│    BayesianClassifier + BCIEvaluator                 │
└──────────────────────────────────────────────────────┘
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
| `utils` | `constants.py` | 通道映射、montage 等常量 | — |
| | `layout.py` | 路径封装 dataclass | `os` |
| | `session_config.py` | 被试级配置读写（config.json） | `json`, `os` |
| 根目录 | `config.py` | 全局路径配置（纯函数） | `os` |

---

## 执行入口

```
scripts/train.py
    ├── run_preprocessing()    ← 组装 BCIDataLoader + EEGPreprocessor + EpochProcessor
    ├── run_feature_extraction() ← 组装 OVOCspFeatureExtractor
    └── run_classification()   ← 组装 BayesianClassifier + BCIEvaluator

scripts/evaluate.py  → 复用预处理逻辑，加载训练好的模型评估测试集
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
