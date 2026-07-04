# my-eeg-bci-pipeline

模块化的 EEG-BCI 分类流水线，支持预处理、特征提取、分类训练与评估的完整流程。

> BCI 领域知识笔记见 [bci-notes.md](./bci-notes.md)

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
│   ├── pipeline/
│   │   ├── preprocess_pipeline.py     # 预处理管道（加载→预处理→分段→保存）
│   │   ├── feature_pipeline.py        # 特征提取管道（加载epochs→CSP→保存）
│   │   └── classify_pipeline.py       # 分类管道（加载特征→训练→CV→保存）
│   └── utils/
│       ├── constants.py               # 通道映射等常量
│       └── session_config.py          # 被试级配置管理（config.json 读写）
│
├── scripts/                           # 运行入口脚本
│   ├── run_preprocessing.py           # 单被试预处理
│   ├── run_ovocsp_feature.py          # 单被试特征提取
│   ├── run_classifier.py              # 单被试分类训练
│   ├── training.py                    # 单被试全流程（预处理+特征+分类）
│   ├── batch_training.py              # 批量训练（A01~A09）
│   ├── batch_evaluate.py              # 批量测试集评估
│   ├── convert_mat_labels.py          # .mat 标签转 .npy
│   ├── search_best_params.py          # 特征+分类器参数网格搜索
│   └── params_search_full.py          # 含分段窗口的全参数搜索
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
└── draft/                             # 草稿代码
```

---

## 使用流程

### 1. 准备数据

将原始 `.gdf` 文件放入 `data/BCICIV_2a/raw/`，并复制配置模板：

```bash
cp example_config.json data/BCICIV_2a/config.json
```

### 2. 预处理

```bash
python scripts/run_preprocessing.py
```

输出：`data/BCICIV_2a/epochs/A01T_epo.fif`

可在脚本中修改 `SUBJECT_ID` 和 `SESSION` 处理其他被试，或取消注释循环批量处理。

### 3. 特征提取

```bash
python scripts/run_ovocsp_feature.py
```

输出：
- `data/BCICIV_2a/feature/A01T_ovocsp_features.npy`
- `data/BCICIV_2a/feature/A01T_ovocsp_extractor.joblib`

### 4. 分类训练

```bash
python scripts/run_classifier.py
```

输出：
- `data/BCICIV_2a/classifier/A01T_bayesian_clf.joblib`
- `data/BCICIV_2a/result/A01T_bayesian_train_results_*.json`

### 5. 单被试全流程（2-4 合并）

```bash
python scripts/training.py
```

### 6. 批量训练（A01~A09）

```bash
python scripts/batch_training.py
```

可通过顶部的 `DO_PREPROCESS` / `DO_FEATURE` 开关跳过已完成的步骤。

### 7. 测试集评估

```bash
python scripts/batch_evaluate.py
```

加载 T 集训练好的特征提取器和分类器，对 E 集进行预测，计算 Kappa 和 Accuracy。

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