# Label Files

This directory contains outcome labels for model training.

## Files

1. **metastasis_status_label.csv**: Binary metastasis labels
   - Used by: Status Prediction Module
   - Required columns:
     - case_submitter_id: Patient ID
     - folder_id: Slide ID (links to feature files)
     - project_id: Cancer type
     - metastasis_label: 0 (no metastasis), 1 (metastasis)
   
2. **future_trajectory_label.csv**: Patient outcome labels at specific time horizons
   - Used by: Trajectory Prediction Module
   - Required columns:
     - case_submitter_id: Patient ID
     - folder_id: Slide ID (links to feature files)
     - project_id: Cancer type
     - new_tumor_event_type: No Meta No Recur, Locoregional Recurrence, Distant Metastasis
     - days: Time from diagnosis to event or last follow-up

## Important Notes

- **folder_id must match feature file names** (without .pt extension)
- These are EXAMPLE files with synthetic data
- Real labels must be derived from your actual clinical follow-up data
- folder_id links slides to their corresponding feature files

