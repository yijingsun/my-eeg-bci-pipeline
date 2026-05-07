# My-EEG-BCI-Baseline
A modular pipeline for EEG classification.

## Getting Started

### Prerequisites
- Python 3.10
- All dependencies are listed in `requirements.txt`

### Setup

```bash
# Clone the repo
git clone <your-repo-url>
cd my-eeg-bci-pipeline

# Create and activate virtual environment
python3.10 -m venv .myvenv
source .myvenv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Quick Start: Run Preprocessing

1. Place the raw .gdf files in data/BCICIV_2a/raw/.
2. (Optional) Adjust parameters in data/BCICIV_2a/config.json.
   The default settings work for the BCI Competition IV 2a dataset.
3. Execute:

```bash
cp example_config.json data/BCICIV_2a/config.json
python scripts/run_preprocessing.py
```
This will process subject A01T and save the cleaned epochs to data/BCICIV_2a/epochs/A01T_epo.fif

## Project Structure

```
my-eeg-bci-pipeline/
├── config.py                  # Global path configuration
├── src/                       # Source code (utils, data_preparation, ...)
├── scripts/                   # Entry points for running pipelines
├── data/                      # Datasets (config)
│   └── BCICIV_2a/
│       └── config.json
├── requirements.txt
└── README.md
```

## Datasets

1. Motor Imagery (MI, 运动想象)
    - BCI Competition IV Dataset IIa (BCICIV_2a_gdf)
        - Description: Imagination of movement (4-class), 1-left hand, 2-right hand, 3-both feet, 4-tongue
        - Data Type: EEG (Electroencephalogram, 头皮脑电图)
        - EOG Estimate: 1-eyes open, 2-eyes closed, 3-eyes movement (EOG, Electroencephalogram, 眼电图)
            - [Download](https://www.bbci.de/competition/download/competition_iv/BCICIV_2a_gdf.zip) or use `wget`
            - [Description](https://www.bbci.de/competition/iv/desc_2a.pdf)
            - [True Label](https://www.bbci.de/competition/iv/results/ds2a/true_labels.zip)

2. Cognitive Task (认知任务)
    - TODO Spelling Task (拼写任务) BCI Competition IV Dataset IIb – P300 Speller (BCICIV_2b_gdf)

3. Sleep Stage Classification (睡眠分期)
    - TODO Sleep-EDF

4. Emotion Recognition (情绪识别)
    - TODO DEAP / SEED

## Data Cleaning

1. Reading data
    - .gdf (General Data Format)
        - `mne.io.read_raw_gdf()`

2. Preprocessing
    - Down Sampling (Resample) 
        - eg. 250Hz to 128Hz
    - Channel Selection
        - Bad Channel: Drop or Interpolate
    - Filtering (Coarse filtering​ & Fine filtering​)
        - Method
            - FIR (Finite Impulse Response) band-pass filter
        - Frequency Range
            - For Motor Imagery: 0.5 Hz – 40 Hz / 8 Hz – 30 Hz (Mu rhythm & Beta rhythm)
    - Re-referencing
        - Average
        - Mastoids
    - Artifact Removal
        - Type
            - Eye EOG / Muscle EMG / Heart ECG / Line Noise / Channel Noise
        - Method 
            - ICA (Independent Component Analysis) [UCSD Labeling Tutorial](https://labeling.ucsd.edu/tutorial/labels)

3. Epoching (Segment continuous signals into labeled epochs)
    - Pick MI event epochs (trials)
        - typical window duration: 1s - 4s after cue on imagery (eg. for dataset bciciv 2a)
    - Drop bad/eog/... channels
    - Drop bad trials
    - Save epochs data .fif

## Training Models

### Traditional Machine Learning

1. Feature Extraction
    - Spatial Domain
        - CSP (Common Spatial Pattern)  -> CSP projectotion matrix: `Wcsp : (n_channels, 2 * csp_n_components)`, eg. (22, 2*4)
    - Time Domain
        - ERP (Event-Related Potential)
    - Frequency Domain
        - PSD (Power Spectral Density)

2. Dimensionality Reduction
    - LDA (Linear Discriminant Analysis)  -> LDA projection matrix: `Wlda : (2 * csp_n_components, lda_n_components)`
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

***
## Useful Links
- [MNE API reference](https://mne.tools/stable/api/python_reference.html)