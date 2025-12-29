# METASIGHT Example Data Structure

This directory demonstrates the expected data organization for METASIGHT training and inference.

## Overview

The example structure shows:
1. How foundation model features should be organized
2. Required clinical data files and format
3. Label file structure for both modules

**Note**: This directory contains only example file structures and formats. Actual data must be obtained from TCGA or your own dataset.

---

## Directory Structure

```
example_data/
├── features/
│   └── CANCER_TYPE-SLIDE_TYPE/
│       └── FOUNDATION_MODEL/
│           └── 20X/
│               └── features/
│                   ├── example_slide_001.pt
│                   ├── example_slide_002.pt
│                   └── example_slide_003.pt
├── clinical/
│   └── nationwidechildrens.org_clinical_patient_brca.txt_BRCA/
│       ├── clinical_information.pkl
│       ├── clinical_for_ipcw.csv
│       └── README.txt
└── labels/
    ├── metastasis_status_label.csv
    ├── future_trajectory_label.csv
    └── README.txt
```

---

## 1. Feature Files

### Location
`features/{CANCER_TYPE}-{SLIDE_TYPE}/{FOUNDATION_MODEL}/20X/features/`

### Format
- **File type**: PyTorch tensor (`.pt`)
- **Content**: Pre-extracted patch-level features from whole-slide images
- **Naming convention**: `{slide_id}.pt`
  - Example: `TCGA-A1-A0SB-01Z-00-DX1.pt`

### Tensor Shape
```python
import torch

# Load feature file
features = torch.load('example_slide_001.pt')

# Expected shape: [n_patches, feature_dim]
# Example: [2048, 768]
# - 2048 patches from the whole-slide image
# - 768 features per patch (foundation model dimension)

print(features.shape)  # torch.Size([2048, 768])
```

### Foundation Model Dimensions
| Model | Feature Dimension |
|-------|-------------------|
| CHIEF | 768 |
| UNI | 1024 |
| GIGAPATH | 1536 |
| VIRCHOW2 | 2560 |

### Quality Requirements
- No NaN or Inf values
- Minimum 50 patches per slide (typical: 1000-5000)
- Consistent feature dimension across all files for same foundation model

---

## 2. Clinical Data

### clinical_information.pkl

Pickle file containing patient and slide metadata.

**Required Columns:**
```python
{
    'case_submitter_id': 'PATIENT_001',  # Patient ID
    'folder_id': 'PATIENT_001_SLIDE_01',  # Slide ID
    'project_id': 'CANCER_TYPE',
    'gender': 'FEMALE',
    'race': 'WHITE',
    'ethnicity': 'NOT HISPANIC OR LATINO',
    'age_at_diagnosis': 45,
    'ajcc_pathologic_stage': 'Stage IIA',
    'ajcc_pathologic_t': 'T2',
    'ajcc_pathologic_n': 'N0',
    'ajcc_pathologic_m': 'M0'
}
```

**Example Code to Create:**
```python
import pandas as pd

clinical_df = pd.DataFrame([
    {
    'case_submitter_id': 'PATIENT_001',
    'folder_id': 'PATIENT_001_SLIDE_01',
    'project_id': 'CANCER_TYPE',
        'gender': 'FEMALE',
        'race': 'WHITE',
        'ethnicity': 'NOT HISPANIC OR LATINO',
        'age_at_diagnosis': 45,
        'ajcc_pathologic_stage': 'Stage IIA',
        'ajcc_pathologic_t': 'T2',
        'ajcc_pathologic_n': 'N0',
        'ajcc_pathologic_m': 'M0'
    },
    # ... more patients
])

clinical_df.to_pickle('clinical_information.pkl')
```

### clinical_for_ipcw.csv

CSV file with covariates for IPCW computation (trajectory module).

**Format:**
```csv
case_submitter_id,age_at_diagnosis,gender,race,ethnicity,ajcc_pathologic_stage,ajcc_pathologic_t,ajcc_pathologic_n,ajcc_pathologic_m
TCGA-A1-A0SB,45,FEMALE,WHITE,NOT HISPANIC OR LATINO,Stage IIA,T2,N0,M0
TCGA-A1-A0SC,52,FEMALE,BLACK OR AFRICAN AMERICAN,NOT HISPANIC OR LATINO,Stage IIIA,T3,N1,M0
TCGA-A1-A0SD,38,FEMALE,ASIAN,NOT HISPANIC OR LATINO,Stage IA,T1,N0,M0
```

**Notes:**
- Must have same patients as in clinical_information.pkl
- Used for computing censoring weights (IPCW)
- Missing values are handled via multiple imputation

---

## 3. Label Files

### metastasis_status_label.csv (Status Prediction)

Labels for binary metastasis prediction.

**Format:**
```csv
case_submitter_id,project_id,gender,race,ajcc_pathologic_stage,metastasis_label
PATIENT_001,PATIENT_001_SLIDE_01,CANCER_TYPE,0
PATIENT_002,PATIENT_002_SLIDE_01,CANCER_TYPE,1
PATIENT_003,PATIENT_003_SLIDE_01,CANCER_TYPE,0
TCGA-A2-A0CM,TCGA-LUAD,MALE,WHITE,Stage IB,0
TCGA-A2-A0CN,TCGA-LUAD,FEMALE,WHITE,Stage IIIB,1
```

**Column Descriptions:**
- `case_submitter_id`: Patient identifier (must match clinical data)
- `project_id`: Cancer type identifier (e.g., BRCA, LUAD, COAD)
- Demographics: Must match clinical data for merging
- `metastasis_label`: 
  - `0` = No distant metastasis
  - `1` = Distant metastasis occurred

### future_trajectory_label.csv (Trajectory Prediction)

Event data for time-to-event analysis.

**Format:**
```csv
case_submitter_id,project_id,new_tumor_event_type,days
PATIENT_001,PATIENT_001_SLIDE_01,CANCER_TYPE,No Meta No Recur,1825
PATIENT_002,PATIENT_002_SLIDE_01,CANCER_TYPE,Distant Metastasis,456
PATIENT_003,PATIENT_003_SLIDE_01,CANCER_TYPE,Locoregional Recurrence,892
TCGA-A2-A0CM,TCGA-LUAD,No Meta No Recur,2190
TCGA-A2-A0CN,TCGA-LUAD,Distant Metastasis,234
```

**Column Descriptions:**
- `case_submitter_id`: Patient identifier
- `project_id`: TCGA cancer type
- `new_tumor_event_type`: Event category
  - `"No Meta No Recur"`: No event (censored or event-free)
  - `"Locoregional Recurrence"`: Local/regional recurrence
  - `"Distant Metastasis"`: Distant metastasis
- `days`: Time from diagnosis to event or last follow-up

**Label Assignment at Cutoff (e.g., 365 days):**
```python
# At 365-day cutoff:
# - If event occurs <= 365 days:
#   - Distant Metastasis = Class 2
#   - Locoregional Recurrence = Class 1
#   - No event = Class 0
# - If days > 365: Class 0 (event-free at this horizon)
# - If censored before 365: Use IPCW to handle
```

---

## Validation Checklist

Before training, validate your data:

### Feature Files
- [ ] All .pt files load without errors
- [ ] Tensors have correct shape [n_patches, feature_dim]
- [ ] Feature dimension matches foundation model
- [ ] No NaN or Inf values
- [ ] Minimum 50 patches per slide

### Clinical Data
- [ ] `clinical_information.pkl` loads successfully
- [ ] All required columns present
- [ ] Patient IDs match between files
- [ ] Slide IDs (folder_id) match feature filenames

### Labels
- [ ] Label files load as CSV
- [ ] Patient IDs match clinical data
- [ ] Labels are valid (0/1 for status, valid events for trajectory)
- [ ] Sufficient samples per class (min 5 per fold)

### Integration
- [ ] Feature files exist for all slides in clinical data
- [ ] Clinical data exists for all slides with features
- [ ] No duplicate patient IDs
- [ ] Meets minimum sample size requirements

### Run Validation Script
```bash
python data/validation.py \
  --cancer BRCA \
  --foundation_model YOUR_MODEL \
  --slide_type FS \
  --task status \
  --feature_root example_data/features/ \
  --clinical_root example_data/clinical/ \
  --label_file example_data/labels/metastasis_status_label.csv
```

---

## Creating Your Dataset

### Step 1: Extract Foundation Model Features

Use a foundation model (e.g., CHIEF, UNI, GIGAPATH, VIRCHOW2, or custom models) to extract patch-level features from your whole-slide images. This is typically done using the foundation model's official implementation.

```python
# Pseudo-code for feature extraction
import torch
from foundation_model import load_model, extract_features

model = load_model('YOUR_FOUNDATION_MODEL')
features = extract_features(model, 'path/to/slide.svs')
torch.save(features, 'slide_id.pt')
```

### Step 2: Prepare Clinical Data

Collect patient metadata from TCGA clinical files or your institution's database.

```python
import pandas as pd

# Combine slide IDs with patient metadata
clinical_df = pd.read_csv('tcga_clinical.csv')
slides_df = pd.read_csv('slide_mapping.csv')

merged = pd.merge(clinical_df, slides_df, on='case_submitter_id')
merged.to_pickle('clinical_information.pkl')
```

### Step 3: Prepare Labels

**For Status Prediction:**
Extract metastasis status from patient follow-up data.

**For Trajectory Prediction:**
Extract time-to-event data including event type and time.

### Step 4: Validate

Run data validation to ensure everything is correct before training.

---

## Example File Sizes

Typical file sizes for reference:

| File Type | Example Size | Notes |
|-----------|-------------|-------|
| Feature file (.pt) | 6-12 MB | For 2000 patches × 768 dims |
| clinical_information.pkl | 1-5 MB | For 500-1000 patients |
| clinical_for_ipcw.csv | 50-200 KB | Sparse clinical matrix |
| metastasis_status_label.csv | 100-500 KB | Pan-cancer labels |
| future_trajectory_label.csv | 50-200 KB | Event data |

---

## Need Help?

- Review the main [README.md](../README.md) for full documentation
- Check the data validation script: `data/validation.py`
- Review example training scripts in `scripts/`
- Open an issue on GitHub for support

---

## IMPORTANT: Data Availability

This repository does NOT include actual TCGA data. To use METASIGHT:

1. **TCGA Data**: Request access from [GDC Data Portal](https://portal.gdc.cancer.gov/)
2. **Foundation Model Features**: Extract using publicly available foundation models
3. **Clinical Data**: Download from TCGA's clinical data repository
4. **Labels**: Derive from TCGA follow-up and pathology reports

Please respect data use agreements and patient privacy when working with clinical data.

