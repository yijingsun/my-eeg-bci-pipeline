# EEG-BCI 知识笔记

> 本项目涉及的 BCI 领域知识整理，包括数据集、预处理流程、特征提取与分类方法、评估指标。

---

## 1. Datasets

### 1.1 Motor Imagery（运动想象）

**BCI Competition IV Dataset IIa** (`BCICIV_2a`)
- 4-class motor imagery: left hand, right hand, both feet, tongue
- Data Type: EEG（头皮脑电图）
- EOG Estimate: 1-eyes open, 2-eyes closed, 3-eyes movement
- [Download](https://www.bbci.de/competition/download/competition_iv/BCICIV_2a_gdf.zip)
- [Description](https://www.bbci.de/competition/iv/desc_2a.pdf)
- [True Label](https://www.bbci.de/competition/iv/results/ds2a/true_labels.zip)

### 1.2 Other Datasets（待扩展）

| Task | Dataset | Status |
|------|---------|--------|
| Cognitive Task（认知任务） | BCI Competition IV Dataset IIb – P300 Speller | TODO |
| Sleep Stage Classification（睡眠分期） | Sleep-EDF | TODO |
| Emotion Recognition（情绪识别） | DEAP / SEED | TODO |

---

## 2. Data Cleaning（数据预处理）

### 2.1 Reading Data
- `.gdf`（General Data Format）→ `mne.io.read_raw_gdf()`

### 2.2 Preprocessing Pipeline

#### Down Sampling（降采样）
- e.g. 250Hz → 128Hz

#### Channel Selection（通道选择）
- Bad Channel: Drop or Interpolate

#### Filtering（滤波）
- **Method**: FIR（Finite Impulse Response）band-pass filter
- **Frequency Range**（For Motor Imagery）:
  - Coarse filtering（粗滤波，ICA 前）: 0.5 Hz – 40 Hz
  - Fine filtering（精滤波，MI 特征提取前）: 8 Hz – 30 Hz（Mu rhythm & Beta rhythm）

#### Re-referencing（重参考）
- Average（全通道平均）
- Mastoids（乳突参考）

#### Artifact Removal（伪迹去除）
- **Type**: Eye EOG / Muscle EMG / Heart ECG / Line Noise / Channel Noise
- **Method**: ICA（Independent Component Analysis）
  - [UCSD Labeling Tutorial](https://labeling.ucsd.edu/tutorial/labels)

### 2.3 Epoching（分段）
- Segment continuous signals into labeled epochs
- Pick MI event epochs (trials)
  - Typical window: 1s – 4s after cue（项目中针对每个被试搜索了最佳的分割时间段）
- Drop bad/EOG channels
- Drop bad trials
- Save as `.fif`

---

## 3. Training Models

### 3.1 Feature Extraction（特征提取）

| Domain | Method | Output |
|--------|--------|--------|
| Spatial（空间域） | CSP（Common Spatial Pattern） | Projection matrix `Wcsp: (n_channels, 2 * csp_n_components)`, e.g. (22, 8) |
| Time（时间域） | ERP（Event-Related Potential） | — |
| Frequency（频率域） | PSD（Power Spectral Density） | — |

**本项目采用**: OVO-CSP（One-vs-One CSP），每对类别训练一个 CSP，拼接所有配对特征。

### 3.2 Dimensionality Reduction（降维）

| Method | Description |
|--------|-------------|
| LDA | Projection matrix `Wlda: (2 * csp_n_components, lda_n_components)`，最多降至 `n_classes - 1` 维 |
| PCA | Principal Component Analysis |

### 3.3 Classification Models

**Traditional Machine Learning**
- LDA: Most widely used in BCI systems
- SVM（RBF kernel）
- Logistic Regression / Random Forest / K-Nearest Neighbors
- Bayesian Classifier（共享协方差矩阵，与 LDA 等价）

**Deep Learning**
- EEGNet
- EEG-Transformer

---

## 4. Performance Evaluation（评估指标）

| Metric | Description |
|--------|-------------|
| Accuracy | Overall classification accuracy |
| Kappa Coefficient | BCI Competition 官方指标，排除随机一致性 |
| Confusion Matrix | 查看各类别分类详情 |
| F1-score | Precision 与 Recall 的调和均值 |

---

## 5. Useful Links

- [MNE-Python API Reference](https://mne.tools/stable/api/python_reference.html)
