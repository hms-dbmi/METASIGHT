# Repository Corrections Summary

## 1. Input Data Terminology

### Corrected: Tile-Level Foundation Model Features

**Previous (Incorrect)**: "whole-slide images as input"
**Current (Correct)**: "tile-level foundation model features as input"

### What Changed:
- Clarified that models use **pre-extracted tile features**, not raw images
- Added foundation model dimensions: CHIEF (768), UNI (1024), GIGAPATH (1536), VIRCHOW2 (2560)
- Updated all docstrings and comments to reflect tile-level processing
- Created `TERMINOLOGY_CORRECTION.md` for reference

### Files Updated:
- `data/dataset.py` - Dataset and collate function docstrings
- `models/architectures.py` - Model forward pass comments
- `README.md` - Added "Input Data" section
- `REPOSITORY_SUMMARY.md` - Added foundation model details

---

## 2. Future Trajectory Prediction Module Description

### Corrected: Multi-Class Classification (NOT Survival Analysis)

**Previous (Incorrect)**: "Time-to-metastasis survival analysis"
**Current (Correct)**: "Multi-class classification for patient outcome at time horizons"

### What It Actually Does:
- Predicts discrete outcome classes at fixed time points (1, 2, 3 years)
- Class 0: Stable disease
- Class 1: Locoregional recurrence  
- Class 2: Distant metastasis

### Key Clarifications:
- **NOT** a time-to-event or survival analysis model
- **NOT** using Cox proportional hazards
- **DOES** use IPCW to handle censored data in classification
- Evaluation: AUROC and Brier score (NOT C-Index)

### Files Updated:
- `README.md` - Module description and overview
- `configs/trajectory_prediction.yaml` - Comment about time horizons
- `data/ipcw.py` - Docstring clarification
- `example_data/labels/README.txt` - Label description
- Created `TRAJECTORY_MODULE_CLARIFICATION.md` for reference

### Verification:
```bash
# All incorrect terms removed (0 matches):
grep -ri "survival analysis\|time-to-event\|time-to-metastasis" \
  --include="*.md" --include="*.py" --include="*.yaml" --include="*.txt" . \
  | grep -v "TRAJECTORY_MODULE_CLARIFICATION.md" | wc -l
# Result: 0
```

---

## 3. Patch vs. Tile Terminology

### Standardized to "Tile"

**Previous**: Mixed use of "patch" and "tile"
**Current**: Consistent use of "tile" in user-facing documentation

**Note**: Internal variable names (e.g., `n_patches`, `max_patches`) remain unchanged in code to avoid breaking functionality. Only user-facing documentation updated.

---

## Summary

All terminology now accurately reflects:
1. **Input**: Tile-level foundation model features (pre-extracted)
2. **Status Module**: Binary classification
3. **Trajectory Module**: Multi-class classification at time horizons
4. **IPCW**: Weighting scheme for handling censored data in classification
5. **Metrics**: AUROC and Brier score (no survival metrics)

Repository is now professionally accurate and ready for publication.
