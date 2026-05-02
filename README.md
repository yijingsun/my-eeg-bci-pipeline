# My-EEG-BCI-Baseline
## Dev Environment
- Python 3.10.20
    - `python3.10 -m venv .myvenv`
    - `source .myvenv/bin/activate`
- requirements.txt
    `python -m pip install --upgrade pip -r requirements.txt`
## Datasets
- Motor Imagery (MI, 运动想象)
    - BCI Competition IV Dataset IIa (BCICIV_2a_gdf)
        - Imagination of movement (4-class)：1-left hand, 2-right hand, 3-both feet, 4-tongue (EEG, Electroencephalogram, 脑电图)
        - EOG Estimate: 1-eyes open, 2-eyes closed, 3-eyes movement (EOG, Electroencephalogram, 眼电图)
            - Download:`wget https://www.bbci.de/competition/download/competition_iv/BCICIV_2a_gdf.zip`
            - Description:`https://www.bbci.de/competition/iv/desc_2a.pdf`
            - True Label:`https://www.bbci.de/competition/iv/results/ds2a/true_labels.zip`
- Emotion Recognition (情绪识别)
- Cognitive Task (认知任务)
    - BCI Competition IV Dataset IIb – P300 Speller (BCICIV_2b_gdf)
        - Spelling Task (拼写任务):
- Sleep Stage Classification (睡眠分析)
## Data Cleaning
1. Reading data
    - .gdf (General Data Format)
        - `mne.io.read_raw_gdf()`
2. Preprocessing
    - Filtering
        - Filter
            - FIR (Finite Impulse Response) band-pass filter
        - Frequency
            - For Motor Imagery: 0.5 Hz – 40 Hz / 8 Hz – 30 Hz
    - Re-referencing
    - Artifact Removal
        - ICA (Independent Component Analysis)
            - https://labeling.ucsd.edu/tutorial/labels
3. Epoching (Segment continuous signals into labeled epochs)
## Training Models
### Traditional Machine Learning
1. Feature Extraction
    - CSP (Common Spatial Pattern)
    - PSD (Power Spectral Density)
2. Feature Selection
    - LDA (Linear Discriminant Analysis)
    - PCA (Principal Component Analysis)
3. Classification Models
    - LDA: Most widely used in BCI systems
    - SVM (RBF kernel)
    - Logistic Regression, Random Forest, K-Nearest Neighbors
### Deep Learning
- Architectures
    - EEGNet
    - EEG-Transformer
## Performance Evaluation
- Overall Accuracy
- Kappa coefficient (official competition metric)
- Confusion Matrix
- F1-score