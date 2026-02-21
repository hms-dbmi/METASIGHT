# METASIGHT: Pan-Cancer Metastasis Prediction System

Official implementation of METASIGHT, a deep learning system for predicting distant metastasis from whole-slide histology images.

## Overview

METASIGHT consists of two complementary prediction modules:
- **Metastasis Status Prediction Module**: Binary classification (presence/absence of distant metastasis)
- **Future Trajectory Prediction Module**: Multi-class classification for patient outcome at specific time horizons (1, 2, 3 years): stable disease vs. locoregional recurrence vs. distant metastasis

Both modules use attention-based Multiple Instance Learning (MIL) to aggregate tile-level foundation model features. Features are pre-extracted from whole-slide images using foundation models (CHIEF, UNI, GIGAPATH, or VIRCHOW2).

---

## Installation

### Requirements

- Python >= 3.8
- PyTorch >= 1.10.0
- CUDA (recommended for GPU acceleration)

### Install Dependencies

```bash
# Clone repository
git clone <repository_url>
cd Metastasis_STpath_github

# Install requirements
pip install -r requirements.txt
```

### Required Python Packages

```
torch>=1.10.0
torchvision>=0.11.0
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
torchmetrics>=0.6.0
scipy>=1.7.0
statsmodels>=0.13.0
lifelines>=0.27.0
h5py>=3.6.0
anndata>=0.8.0
matplotlib>=3.5.0
seaborn>=0.11.0
tqdm>=4.62.0
pyyaml>=6.0
```

---

## Data Preparation

### Foundation Model Feature Dimensions

| Foundation Model | Feature Dimension |
|-----------------|-------------------|
| CHIEF | 768 |
| UNI | 1024 |
| GIGAPATH | 1536 |
| VIRCHOW2 | 2560 |

### Required Data Files

#### 1. Histology Features (Pre-extracted)

- **Format**: PyTorch tensor files (`.pt`)
- **Shape**: `[n_tiles, feature_dim]` where `feature_dim` depends on foundation model
- **Content**: Pre-extracted features from whole-slide images using foundation models
- **Naming**: `{slide_id}.pt` (e.g., `TCGA-A1-A0SB-01Z-00-DX1.pt`)

#### 2. Clinical Data

**File**: `clinical_for_ipcw.csv` (used by Trajectory Prediction for IPCW computation)

**Location**: `{clinical_root}/clinical_for_ipcw.csv`

**Required columns**:
- `case_submitter_id`: Patient identifier
- `project_id`: Cancer type identifier (optional if using cancer-specific directories)
- `age_at_diagnosis`: Age at diagnosis in years
- `gender`: Patient gender (e.g., MALE, FEMALE)
- `race`: Patient race category
- `ethnicity`: Patient ethnicity
- `ajcc_pathologic_stage`: AJCC pathologic stage (e.g., Stage IA, Stage IIB)
- `ajcc_pathologic_t`: T stage (tumor size/extent)
- `ajcc_pathologic_n`: N stage (lymph node involvement)
- `ajcc_pathologic_m`: M stage (distant metastasis)

**Optional**: Can include additional covariates (e.g., BMI, biomarkers) - all available columns are automatically used for IPCW

**Note**: Status prediction does NOT require clinical files - it only needs labels with `folder_id`

---

### Required Column Names Summary

| File | Column Name | Type | Description | Required For |
|------|-------------|------|-------------|--------------|
| **metastasis_status_label.csv** | `case_submitter_id` | string | Patient identifier | Status |
| | `folder_id` | string | Slide identifier (matches feature filename) | Status |
| | `project_id` | string | Cancer type identifier | Status |
| | `metastasis_label` | int | Binary label: 0 or 1 | Status |
| **future_trajectory_label.csv** | `case_submitter_id` | string | Patient identifier | Trajectory |
| | `folder_id` | string | Slide identifier (matches feature filename) | Trajectory |
| | `project_id` | string | Cancer type identifier | Trajectory |
| | `new_tumor_event_type` | string | Event type (exact match) | Trajectory |
| | `days` | numeric | Days from diagnosis to event/censoring | Trajectory |
| **clinical_for_ipcw.csv** | `case_submitter_id` | string | Patient identifier | Trajectory |
| | `project_id` | string | Cancer type (optional if using subdirs) | Trajectory |
| | `age_at_diagnosis` | numeric | Age in years | Trajectory |
| | `gender` | string | Patient gender | Trajectory |
| | `race` | string | Race category | Trajectory |
| | `ethnicity` | string | Ethnicity category | Trajectory |
| | `ajcc_pathologic_stage` | string | AJCC stage (e.g., "Stage IIA") | Trajectory |
| | `ajcc_pathologic_t` | string | T stage | Trajectory |
| | `ajcc_pathologic_n` | string | N stage | Trajectory |
| | `ajcc_pathologic_m` | string | M stage | Trajectory |

**Important Notes:**
- All column names are **case-sensitive** and must match exactly
- `folder_id` must match feature file names (without `.pt` extension)
- `new_tumor_event_type` values must be exact strings: `"No Meta No Recur"`, `"Locoregional Recurrence"`, or `"Distant Metastasis"`
- Additional columns in `clinical_for_ipcw.csv` are automatically used for IPCW computation

---

#### 3. Label Files

For **Status Prediction**:
- **File**: `metastasis_status_label.csv`
- **Required columns**:
  - `case_submitter_id`: Patient identifier
  - `folder_id`: Slide identifier (links to feature files)
  - `project_id`: Cancer type identifier
  - `metastasis_label`: Binary label (0 = no metastasis, 1 = metastasis)

For **Trajectory Prediction**:
- **File**: `future_trajectory_label.csv`
- **Required columns**:
  - `case_submitter_id`: Patient identifier
  - `folder_id`: Slide identifier (links to feature files)
  - `project_id`: Cancer type identifier
  - `new_tumor_event_type`: Event type (exact string match required)
    - `"No Meta No Recur"`: No event occurred
    - `"Locoregional Recurrence"`: Local recurrence
    - `"Distant Metastasis"`: Distant metastasis
  - `days`: Time from diagnosis to event or last follow-up (numeric, in days)

### Directory Structure

```
/path/to/data/
├── foundation_model_features/
│   └── WSI_features/
│       ├── {CANCER_TYPE}-FS/              # Frozen section slides
│       │   ├── {FOUNDATION_MODEL_1}/      # e.g., TCGA-BRCA-FS or CUSTOM_CANCER-FS
│       │   │   └── 20X/
│       │   │       └── features/
│       │   │           ├── slide_001.pt  # [n_tiles, feature_dim]
│       │   │           ├── slide_002.pt
│       │   │           └── ...
│       │   ├── {FOUNDATION_MODEL_2}/
│       │   │   └── 20X/
│       │   │       └── features/
│       │   │           └── ...
│       │   └── {FOUNDATION_MODEL_3}/
│       │       └── 20X/
│       │           └── features/
│       │               └── ...
│       ├── {CANCER_TYPE}-PM/              # Permanent section slides  
│       │   └── {FOUNDATION_MODEL}/
│       │       └── 20X/
│       │           └── features/
│       │               └── ...
│       ├── {ANOTHER_CANCER}-FS/
│       │   └── {FOUNDATION_MODEL}/
│       │       └── 20X/
│       │           └── features/
│       │               └── ...
│       └── ...
│
├── clinical_gdc/
│   └── clinical_for_ipcw.csv              # Clinical covariates for IPCW (all cancers)
│
└── labels/
    ├── metastasis_status_label.csv        # Status labels (binary)
    └── future_trajectory_label.csv        # Trajectory labels (outcome at time horizons)
```

**Note**: For TCGA data, use `TCGA-{CANCER}-{SLIDE_TYPE}` format (e.g., `TCGA-BRCA-FS`). For custom datasets, use `{CANCER}-{SLIDE_TYPE}` format (e.g., `MyDataset-FS`). The code automatically handles both formats.

### Minimum Sample Size Requirements

For reliable model training and evaluation, the following minimum sample sizes are recommended:

| Purpose | Minimum Samples | Rationale |
|---------|----------------|-----------|
| **Training set** | 30 slides | Sufficient for model convergence |
| **Validation set** | 10 slides | Early stopping assessment |
| **Test set (per cancer)** | 30 slides | Reliable per-cancer performance evaluation |
| **Positive class (per fold)** | 5 samples | Prevent class collapse |

**Important Notes:**
- These requirements apply per cancer type when training on multiple cancers
- Cancers with fewer than 30 test samples will receive a warning but can still be included
- Class imbalance is handled via class weighting and IPCW (trajectory module)
- Cross-validation uses patient-level splitting to prevent data leakage

### Data Validation

Validate your dataset before training:

```bash
# Validate status prediction data
python data/validation.py \
  --cancer BRCA \
  --foundation_model CHIEF \
  --slide_type FS \
  --task status \
  --feature_root /path/to/foundation_model_features/WSI_features/ \
  --clinical_root /path/to/clinical_gdc \
  --label_file /path/to/labels/metastasis_status_label.csv

# Validate trajectory prediction data
python data/validation.py \
  --cancer BRCA \
  --foundation_model CHIEF \
  --slide_type FS \
  --task trajectory \
  --feature_root /path/to/foundation_model_features/WSI_features/ \
  --clinical_root /path/to/clinical_gdc \
  --label_file /path/to/labels/future_trajectory_label.csv
```

---

## Usage

### 1. Metastasis Status Prediction

Train binary classifier to predict presence/absence of metastasis. Specify one or more cancer types in `--cancer_list`:

```bash
# Single cancer example
python scripts/train_status_prediction.py \
  --foundation_model CHIEF \
  --slide_type FS \
  --batch_size 32 \
  --learning_rate 1e-4 \
  --dropout 0.3 \
  --num_epochs 50 \
  --fold_n 3 \
  --loss_type combined \
  --scheduler_type cosine \
  --cancer_list BRCA \
  --feature_root /path/to/foundation_model_features/WSI_features/ \
  --clinical_root /path/to/clinical_gdc \
  --label_file /path/to/labels/metastasis_status_label.csv \
  --output_dir results/status_brca_chief_fs

# Multiple cancers example
python scripts/train_status_prediction.py \
  --foundation_model CHIEF \
  --slide_type FS \
  --batch_size 32 \
  --learning_rate 1e-4 \
  --dropout 0.3 \
  --num_epochs 50 \
  --fold_n 3 \
  --loss_type combined \
  --scheduler_type cosine \
  --cancer_list BRCA LUAD KIRC COAD STAD \
  --feature_root /path/to/foundation_model_features/WSI_features/ \
  --clinical_root /path/to/clinical_gdc \
  --label_file /path/to/labels/metastasis_status_label.csv \
  --output_dir results/status_multi_chief_fs
```

**Note**: The `--cancer_list` parameter accepts any cancer type identifier. For TCGA data, use standard abbreviations (e.g., BRCA, LUAD). For custom datasets, use any identifier matching your feature directory structure (e.g., `CUSTOM_CANCER-FS` or `MyDataset-PM`).

### 2. Future Trajectory Prediction

Train multi-class classifier for outcome prediction at time horizons with IPCW. Specify one or more cancer types in `--cancer_list`:

```bash
# Single cancer example
python scripts/train_trajectory_prediction.py \
  --foundation_model CHIEF \
  --slide_type FS \
  --batch_size 16 \
  --learning_rate 5e-5 \
  --dropout 0.2 \
  --hidden_dim 128 \
  --num_epochs 50 \
  --fold_n 4 \
  --loss_type ce \
  --cutoffs 365 730 1095 \
  --use_cross_fit_ipcw \
  --use_class_weight \
  --stratified_cv \
  --cancer_list BRCA \
  --feature_root /path/to/foundation_model_features/WSI_features/ \
  --clinical_root /path/to/clinical_gdc \
  --label_file /path/to/labels/future_trajectory_label.csv \
  --output_dir results/trajectory_brca_chief_fs

# Multiple cancers with multiple time horizons
python scripts/train_trajectory_prediction.py \
  --foundation_model CHIEF \
  --slide_type FS \
  --batch_size 16 \
  --learning_rate 5e-5 \
  --dropout 0.2 \
  --hidden_dim 128 \
  --num_epochs 50 \
  --fold_n 4 \
  --loss_type ce \
  --cutoffs 365 730 1095 \
  --use_cross_fit_ipcw \
  --use_class_weight \
  --stratified_cv \
  --cancer_list BRCA LUAD KIRC COAD STAD HNSC \
  --feature_root /path/to/foundation_model_features/WSI_features/ \
  --clinical_root /path/to/clinical_gdc \
  --label_file /path/to/labels/future_trajectory_label.csv \
  --output_dir results/trajectory_multi_chief_fs
```

### 3. Inference on New Slides

Run predictions on new whole-slide images using trained models:

```bash
# Single slide
python scripts/inference.py \
  --checkpoint results/status_brca_chief_fs/checkpoints/fold0_best.pt \
  --module status \
  --feature_file /path/to/new_slide_features.pt \
  --output_file predictions/new_slide_predictions.csv

# Multiple slides from directory
python scripts/inference.py \
  --checkpoint results/trajectory_brca_chief_fs/cutoff_365days/checkpoints/fold0_best.pt \
  --module trajectory \
  --feature_dir /path/to/test_slides/ \
  --output_file predictions/test_cohort_predictions.csv \
  --output_format csv

# Batch inference with slide list
python scripts/inference.py \
  --checkpoint results/status_multi_chief_fs/checkpoints/fold0_best.pt \
  --module status \
  --feature_dir /path/to/features/ \
  --slide_list cohort_slide_ids.csv \
  --output_file predictions/cohort_predictions.csv
```

---

## Hyperparameters

### Recommended Settings

| Parameter | Status Prediction | Trajectory Prediction |
|-----------|-------------------|----------------------|
| **Batch Size** | 32 | 16 |
| **Learning Rate** | 1e-4 | 5e-5 |
| **Dropout** | 0.3 | 0.2 |
| **Hidden Dim** | 128 | 128 |
| **Loss Function** | combined (CE + TPR) | ce (with IPCW) |
| **Scheduler** | cosine | none |
| **Gradient Clipping** | 1.0 | 1.0 |
| **Early Stopping Patience** | 7 epochs | 7 epochs |
| **CV Folds** | 3 (status) | 4 (trajectory) |

### Loss Functions

- **ce**: Cross-entropy with class weights
- **focal**: Focal loss (addresses class imbalance)
- **combined**: CE + TPR loss (improves recall for metastasis class)

### Learning Rate Schedulers

- **none**: Constant learning rate
- **cosine**: Cosine annealing with warm restarts
- **plateau**: Reduce LR on validation plateau

---

## Output Structure

After training, results are organized as follows:

```
output_dir/
├── checkpoints/
│   ├── fold0_best.pt              # Best model for fold 0
│   ├── fold1_best.pt
│   └── fold2_best.pt
├── predictions/
│   ├── fold0_predictions.csv      # Per-fold predictions
│   ├── fold1_predictions.csv
│   └── fold2_predictions.csv
└── results_summary.json           # Aggregated metrics

# For trajectory prediction:
output_dir/
├── cutoff_365days/
│   ├── checkpoints/
│   ├── predictions/
│   └── results_summary.json
├── cutoff_730days/
│   ├── checkpoints/
│   ├── predictions/
│   └── results_summary.json
└── cutoff_1095days/
    ├── checkpoints/
    ├── predictions/
    └── results_summary.json
```

### Prediction Files

**Status Prediction** (`predictions/fold0_predictions.csv`):
```csv
patient_id,slide_id,fold_id,true_label,pred_prob_metastasis,cancer_type
TCGA-A1-A0SB,TCGA-A1-A0SB-01Z-00-DX1,0,1,0.8234,BRCA
```

**Trajectory Prediction** (`predictions/fold0_predictions.csv`):
```csv
patient_id,slide_id,fold_id,true_label,pred_prob_class_0,pred_prob_class_1,pred_prob_class_2,cancer_type
TCGA-A1-A0SB,TCGA-A1-A0SB-01Z-00-DX1,0,2,0.1234,0.2341,0.6425,BRCA
```

Where:
- Class 0: Stable disease (no event)
- Class 1: Locoregional recurrence
- Class 2: Distant metastasis

### Results Summary (`results_summary.json`):
```json
{
  "cancer": "BRCA",
  "foundation_model": "CHIEF",
  "slide_type": "FS",
  "n_folds": 3,
  "mean_auroc": 0.8234,
  "std_auroc": 0.0456,
  "mean_auprc": 0.7891,
  "std_auprc": 0.0523,
  "hyperparameters": { ... }
}
```

---

## Evaluation Metrics

### Metastasis Status Prediction Module
- **AUROC**: Area under ROC curve (primary metric)
- **AUPRC**: Area under precision-recall curve
- **Brier Score**: Calibration metric (lower is better)
- **Per-Cancer Performance**: Evaluated separately for each cancer type

### Future Trajectory Prediction Module  
- **AUROC (Macro)**: Multi-class one-vs-rest AUROC (primary metric)
- **AUROC (Per-Class)**: Separate AUROC for each outcome class
- **Brier Score**: Multi-class calibration metric
- **Per-Cancer Performance**: Evaluated per cancer when training on multiple cancers

---

## Repository Structure

```
METASIGHT/
├── configs/                        # Configuration files
│   ├── status_prediction.yaml
│   └── trajectory_prediction.yaml
├── data/                          # Data loading and processing
│   ├── __init__.py
│   ├── dataset.py                 # WSI dataset class
│   ├── preprocessing.py           # Feature and clinical loaders
│   ├── ipcw.py                    # IPCW computation
│   └── validation.py              # Data validation utilities
├── models/                        # Neural network architectures
│   ├── __init__.py
│   ├── architectures.py           # MIL networks
│   └── losses.py                  # Loss functions
├── training/                      # Training utilities
│   ├── __init__.py
│   └── schedulers.py              # Learning rate schedulers
├── evaluation/                    # Evaluation metrics
│   ├── __init__.py
│   └── metrics.py                 # Performance metrics
├── scripts/                       # Training and inference scripts
│   ├── train_status_prediction.py
│   ├── train_trajectory_prediction.py
│   └── inference.py               # Standalone inference
├── agentic_ensembling/            # Evolved ensemble programs
│   ├── metastasis_status_prediction/
│   └── future_trajectory_prediction/
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Citation

If you use METASIGHT in your research, please cite:

Kao TW. et al., Agentic AI-Powered Pathology Evaluation Reveals Conserved Morphologic Signatures of Metastatic Progression across 23 Cancer Types (under review).



---

## License

This project is licensed under the GNU Affero General Public License v3.0 - see the [LICENSE](LICENSE) file for details.


---

## Acknowledgments

This work uses pre-extracted features from the following foundation models:
- **CHIEF**: Wang, X., Zhao, J., Marostica, E. et al. A pathology foundation model for cancer diagnosis and prognosis prediction. Nature 634, 970–978 (2024). https://doi.org/10.1038/s41586-024-07894-z
- **UNI**: Chen, R.J., Ding, T., Lu, M.Y. et al. Towards a general-purpose foundation model for computational pathology. Nat Med 30, 850–862 (2024). https://doi.org/10.1038/s41591-024-02857-3
- **GIGAPATH**: Xu, H., Usuyama, N., Bagga, J. et al. A whole-slide foundation model for digital pathology from real-world data. Nature 630, 181–188 (2024). https://doi.org/10.1038/s41586-024-07441-w
- **VIRCHOW2**: Vorontsov, E., Bozkurt, A., Casson, A. et al. A foundation model for clinical-grade computational pathology and rare cancers detection. Nat Med 30, 2924–2935 (2024). https://doi.org/10.1038/s41591-024-03141-0

