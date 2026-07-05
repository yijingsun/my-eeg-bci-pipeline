# my-eeg-bci-pipeline

模块化的 EEG-BCI 分类流水线，支持预处理、特征提取、分类训练与评估的完整流程。

> BCI 领域知识笔记见 [docs/bci-domain-notes.md](./docs/bci-domain-notes.md)
>
> 架构概览见 [docs/ARCHITECTURE.md](./docs/ARCHITECTURE.md) | 扩展开发指南见 [docs/DEVELOPMENT.md](./docs/DEVELOPMENT.md)

---

## 环境准备

- Python 3.10
- 所有依赖见 `pyproject.toml`

```bash
git clone https://github.com/yijingsun/my-eeg-bci-pipeline.git
cd my-eeg-bci-pipeline

python3.10 -m venv .myvenv
source .myvenv/bin/activate

pip install --upgrade pip
pip install -e .
```

---

## 项目结构

```
my-eeg-bci-pipeline/
├── config.py                          # 全局路径配置（只管理目录，不含实验参数）
├── example_config.json                # 实验参数配置模板
├── main.py
├── pyproject.toml                     # 依赖声明 + 包配置
│
├── src/                               # 核心源码
│   ├── data_preparation/
│   │   ├── data_loader.py             # 原始数据加载（GDF → MNE Raw）
│   │   ├── pre_processor.py           # 预处理器（降采样/滤波/ICA/重参考）
│   │   └── epoch_processor.py         # 事件提取与分段
│   ├── feature_extraction/
│   │   └── ovocsp_feature_extractor.py  # OVO-CSP 特征提取器
│   ├── classification/
│   │   └── bayesian_classifier.py     # 贝叶斯分类器（兼容 sklearn 接口）
│   ├── evaluation/
│   │   └── evaluator.py               # K-Fold 交叉验证评估器
│   └── utils/
│       ├── constants.py               # 通道映射等常量
│       └── session_config.py          # 被试级配置管理（config.json 读写）
│
├── scripts/                           # 运行入口脚本
│   ├── train.py                       # 训练（单被试 + --batch 批量）
│   ├── evaluate.py                    # 测试集评估（单被试 + --batch 批量）
│   ├── convert_mat_labels.py          # .mat 标签转 .npy
│   ├── search_best_params.py          # 特征+分类器参数网格搜索
│   ├── params_search_full.py          # 含分段窗口的全参数搜索
│   └── build_package.py               # 打包构建
│
├── tests/                             # 单元测试（与 src/ 一一对应）
│   ├── conftest.py                    # 共享夹具
│   ├── utils/
│   │   ├── test_session_config.py     # SessionConfig 测试
│   │   └── test_constants.py          # 常量测试
│   ├── data_preparation/              # BCIDataLoader / EEGPreprocessor / EpochProcessor
│   ├── feature_extraction/            # OVOCspFeatureExtractor
│   ├── classification/                # BayesianClassifier
│   ├── evaluation/                    # BCIEvaluator
│   ├── pipeline/                      # 端到端组装测试（@pytest.mark.slow）
│   └── scripts/                       # 脚本聚合逻辑 + 搜索集成测试
│
├── data/                              # 数据目录
│   └── BCICIV_2a/
│       ├── config.json                # 被试级实验参数配置
│       ├── raw/                       # 原始 .gdf 文件
│       ├── epochs/                    # 预处理后 .fif epochs
│       ├── label/                     # 标签 .npy / .mat
│       ├── feature/                   # CSP 特征 .npy + 提取器 .joblib
│       ├── classifier/                # 训练好的分类器 .joblib
│       └── result/                    # 交叉验证结果 .json
│
├── notebook/                          # Jupyter 分析笔记
├── draft/                             # 草稿代码
└── docs/                              # 项目文档
    ├── ARCHITECTURE.md                # 架构概览与数据流
    ├── DEVELOPMENT.md                 # 扩展开发指南
    └── bci-domain-notes.md            # BCI 领域知识笔记
```

---

## 使用流程

### 1. 准备数据

将原始 `.gdf` 文件放入 `data/BCICIV_2a/raw/`，并复制配置模板：

```bash
cp example_config.json data/BCICIV_2a/config.json
```

### 2. 训练

```bash
# 单被试全流程（默认 A01，训练集 T）
python scripts/train.py

# 指定被试
python scripts/train.py --subject A03

# 仅运行特征提取
python scripts/train.py --step feature

# 批量模式（A01 / A02 / A03 ...）
python scripts/train.py --batch
```

### 3. 测试集评估

```bash
python scripts/evaluate.py              # 单被试 A01
python scripts/evaluate.py --subject A03
python scripts/evaluate.py --batch     # 批量评估

---

## 配置系统

实验参数通过 `data/<dataset>/config.json` 管理，采用 **default + 被试覆盖** 的两层结构：

```json
{
  "default": {
    "resample_freq": null,
    "filter_ica": [8, 30],
    "tmin": 1.0,
    "tmax": 4.0,
    "csp_n_components": 6,
    ...
  },
  "A01": {
    "T": {},
    "E": { "tmin": 3.0, "tmax": 6.0 }
  }
}
```

- `default`：所有被试的默认参数
- `A01.T` / `A01.E`：针对特定被试/会话的覆盖，只写与 default 不同的字段
- 运行时由 `SessionConfig` 自动合并，支持属性式访问和 `.save()` 持久化

完整参数说明见 `example_config.json` 中的字段。

---

## 参数搜索

```bash
# 特征提取 + 分类器参数网格搜索（固定预处理）
python scripts/search_best_params.py

# 含分段窗口（tmin/tmax）的全参数搜索
python scripts/params_search_full.py
```

搜索完成后可将最优参数写回 `config.json` 对应被试条目。

---

## 测试

```bash
# 安装测试依赖
pip install pytest

# 运行全部测试
python -m pytest tests/ -v

# 运行特定模块
python -m pytest tests/utils/test_session_config.py -v
```

测试文件与 `src/` 一一对应，覆盖：
- `SessionConfig`：config.json 加载、默认值合并、被试覆盖、属性/字典访问、save() diff
- `BayesianClassifier`：fit/predict/predict_proba、sklearn 兼容、持久化
- `OVOCspFeatureExtractor`：fit/transform/fit_transform、LDA、持久化
- `BCIEvaluator`：交叉验证评估
- `BCIDataLoader` / `EEGPreprocessor` / `EpochProcessor`：各预处理步骤
- Pipeline 组装逻辑（slow）：端到端预处理→特征→分类
- Scripts 聚合逻辑：批量汇总统计、argparse 解析