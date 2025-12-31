# Clinical Data Files

This directory contains clinical covariates for IPCW computation (trajectory prediction module only).

## File

**clinical_for_ipcw.csv** - Clinical covariates for all patients

## Required Columns

- `case_submitter_id`: Patient identifier
- `project_id`: Cancer type identifier (e.g., 'BRCA', 'LUAD')
- `age_at_diagnosis`: Age in years (numeric)
- `gender`: Patient gender (e.g., 'MALE', 'FEMALE')
- `race`: Race category
- `ethnicity`: Ethnicity category
- `ajcc_pathologic_stage`: AJCC pathologic stage (e.g., 'Stage IIA')
- `ajcc_pathologic_t`: T stage (e.g., 'T2')
- `ajcc_pathologic_n`: N stage (e.g., 'N0')
- `ajcc_pathologic_m`: M stage (e.g., 'M0')

## Optional Columns

Any additional clinical variables can be included (e.g., BMI, biomarkers, treatment). All available columns are automatically used for IPCW computation.

## Directory Organization

### Structure:
```
clinical_root/
└── clinical_for_ipcw.csv              # Single file for all cancers
```


## Usage

- **Status Prediction**: Does NOT use this file (only needs labels with folder_id)
- **Trajectory Prediction**: Uses this file for computing IPCW weights to handle censored data

## Creating Clinical Files

```python
import pandas as pd

# Example clinical data
clinical_df = pd.DataFrame([
    {
        'case_submitter_id': 'PATIENT_001',
        'project_id': 'BRCA',
        'age_at_diagnosis': 45,
        'gender': 'FEMALE',
        'race': 'WHITE',
        'ethnicity': 'NOT HISPANIC OR LATINO',
        'ajcc_pathologic_stage': 'Stage IIA',
        'ajcc_pathologic_t': 'T2',
        'ajcc_pathologic_n': 'N0',
        'ajcc_pathologic_m': 'M0'
    },
    # ... more patients
])

# Save as CSV
clinical_df.to_csv('clinical_for_ipcw.csv', index=False)
```

## Important Notes

- Missing values in IPCW covariates are handled via Bayesian multiple imputation (MICE)
- All available columns (beyond the required ones) are automatically included in IPCW computation
- This file is only needed for trajectory prediction, not status prediction
- Patients must match those in the trajectory label file (future_trajectory_label.csv)

## Example Data

This directory contains synthetic example data with 3 sample patients for demonstration purposes only. Use real clinical data for actual model training.
