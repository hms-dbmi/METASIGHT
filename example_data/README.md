# Example Data for Testing METASIGHT

This directory contains deidentified real data from TCGA-BRCA for testing the METASIGHT pipeline.

## Contents

### Features (`features/BRCA-FS/CHIEF/20X/pt_files(stain_norm)/`)
- **19 deidentified whole-slide image feature files**
- Extracted using CHIEF foundation model (768 dimensions)
- Frozen section slides (FS)
- Files renamed to `SAMPLE_XXX_SLIDE_01.pt` for deidentification

### Labels

#### `labels/metastasis_status_label.csv`
Binary metastasis status labels for Status Prediction Module
- 19 samples total
- 14 M0 (no metastasis: 10 stable + 4 LRR)
- 5 M1 (distant metastasis)

**Columns:**
- `case_submitter_id`: Deidentified patient ID (SAMPLE_001 - SAMPLE_019)
- `folder_id`: Slide identifier matching feature filenames
- `project_id`: Cancer type (BRCA)
- `metastasis_label`: Binary label (0=M0, 1=M1)

#### `labels/future_trajectory_label.csv`
Patient outcome labels for Trajectory Prediction Module
- 19 samples with time-to-event data
- **10 "No Meta No Recur"** (stable disease)
- **4 "Locoregional Recurrence"** (local recurrence)
- **5 "Distant Metastasis"** (distant metastasis)

**Columns:**
- `case_submitter_id`: Deidentified patient ID
- `folder_id`: Slide identifier matching feature filenames
- `project_id`: Cancer type (BRCA)
- `new_tumor_event_type`: Outcome type
- `days`: Time from diagnosis to event or last follow-up

**Important Note on Trajectory Prediction:**
- At 365-day cutoff: Only 1-2 samples per class have events → AUROC cannot be computed
- At 730-day cutoff: Similar issue with class imbalance
- Brier score is still computed and valid
- For production use, need larger datasets with more events at each time horizon

### Clinical Data

#### `clinical/clinical_for_ipcw.csv`
Clinical covariates for IPCW (Inverse Probability of Censoring Weighting)
- Used by Trajectory Prediction Module
- Synthetic clinical variables generated for deidentification

**Columns:**
- `case_submitter_id`: Deidentified patient ID
- `project_id`: Cancer type
- `age_at_diagnosis`, `gender`, `race`, `ethnicity`
- `ajcc_pathologic_stage`, `ajcc_pathologic_t`, `ajcc_pathologic_n`, `ajcc_pathologic_m`

## Running Tests

```bash
# Quick test (from repository root)
cd /n/data2/hms/dbmi/kyu/lab/tik161/Metastasis_STpath_github
bash test_run/run_test.sh

# Or test individual modules:

# Status Prediction
python scripts/train_status_prediction.py \
  --foundation_model CHIEF \
  --slide_type FS \
  --cancer_list BRCA \
  --feature_root example_data/features/ \
  --label_file example_data/labels/metastasis_status.csv \
  --output_dir test_results/status

# Trajectory Prediction
python scripts/train_trajectory_prediction.py \
  --foundation_model CHIEF \
  --slide_type FS \
  --cancer_list BRCA \
  --feature_root example_data/features/ \
  --label_file example_data/labels/future_trajectory_label.csv \
  --clinical_root example_data/clinical \
  --output_dir test_results/trajectory
```

## Expected Test Results

### Status Prediction
- **AUROC: ~0.77 ± 0.07** - Good performance with 19 samples
- All 3 folds complete successfully
- Results demonstrate binary classification capability

### Trajectory Prediction
- **Brier Score: ~0.21 ± 0.02** - Valid calibration metric
- **AUROC: N/A** - Cannot compute due to insufficient samples per class at time cutoffs
- IPCW weighting computed successfully
- All 3 folds complete without errors

**Why AUROC is N/A:**
- Multi-class AUROC requires ≥2 samples per class in each test fold
- At 365-day cutoff: Only 1 DM sample with event → insufficient for AUROC
- At 730-day cutoff: Only 2-3 samples per class → still insufficient with 3-fold CV
- This is a limitation of the small test dataset, not the pipeline

**For Production:**
- Use larger cohorts (≥30 samples per class recommended)
- Or reduce CV folds (e.g., 2-fold instead of 3-fold for small datasets)
- AUROC will be computable with adequate sample sizes

## Data Source

All feature files and labels are derived from deidentified TCGA-BRCA data:
- Original features extracted from TCGA whole-slide images
- Labels derived from TCGA clinical follow-up data
- File names and patient IDs replaced with generic identifiers
- No PHI (Protected Health Information) included

## Notes

- These are real foundation model features, not synthetic data
- Small sample size (n=19) is for testing pipeline functionality only
- Not suitable for scientific analysis or model evaluation
- For actual research, use larger cohorts with proper train/validation/test splits
- The pipeline is designed to work with any cancer type and feature extraction method

## Pooled Multi-Cancer Dataset (Update: Dec 2025)

The example data now demonstrates **pooled training across multiple cancer types**:

- **8 HNSC samples** - Rich in Locoregional Recurrence events
- **8 LUAD samples** - Rich in Distant Metastasis events  
- **7 BRCA samples** - Stable cases (No Meta No Recur)
- **7 STAD samples** - Stable cases (No Meta No Recur)

This balanced multi-cancer dataset enables:
- Robust AUROC calculation (all classes represented)
- Demonstration of pooled training capability
- Better generalization across cancer types

**Test Results:**
- Status Prediction: AUROC 0.65 ± 0.17
- Trajectory Prediction (1095-day): AUROC 0.66 ± 0.04
