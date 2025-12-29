# Future Trajectory Prediction Module - Clarification

## Correct Description

The Future Trajectory Prediction Module is a **multi-class classification task**, NOT a survival/time-to-event analysis.

### What It Does

Predicts patient outcome at specific time horizons (1, 2, 3 years):
- **Class 0**: Stable disease (no event by the time horizon)
- **Class 1**: Locoregional recurrence (occurred by the time horizon)
- **Class 2**: Distant metastasis (occurred by the time horizon)

### Key Points

1. **Classification, Not Survival Analysis**:
   - Predicts discrete outcome classes at fixed time points
   - Does NOT predict time-to-event or hazard ratios
   - Does NOT use Cox proportional hazards or survival curves

2. **IPCW Role**:
   - IPCW (Inverse Probability of Censoring Weighting) handles censored patients
   - Censored patients are those who were lost to follow-up before the time horizon
   - IPCW reweights uncensored samples to account for missing data
   - This is NOT the same as survival analysis - it's a weighting scheme for classification

3. **Evaluation Metrics**:
   - AUROC (Area Under ROC Curve) - multi-class
   - Brier Score - calibration metric
   - NO C-Index (that's for survival analysis)
   - NO time-dependent AUROC (that's for survival analysis)

### Implementation Details

From `train_trajectory_prediction.py`:
```python
# Multi-class classification for predicting patient outcome at time horizons:
# - Class 0: Stable disease (no event)
# - Class 1: Locoregional recurrence  
# - Class 2: Distant metastasis
```

From `load_trajectory_clinical_data()`:
```python
# Labels patients at a specific time cutoff:
# - Class 0: Stable disease (no event by cutoff)
# - Class 1: Locoregional recurrence
# - Class 2: Distant metastasis
```

### Incorrect Terminology (Fixed)

❌ "Time-to-metastasis survival analysis"
❌ "Time-to-event data"
❌ "Survival analysis"
❌ "Time-to-event module"

✅ "Multi-class classification for patient outcome at time horizons"
✅ "Outcome labels at specific time points"
✅ "Multi-class outcome prediction"
✅ "Classification with censored data handling"

