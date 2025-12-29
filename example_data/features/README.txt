# Feature Files

This directory should contain pre-extracted foundation model features.

## Expected Structure

```
features/
└── {CANCER_TYPE}-{SLIDE_TYPE}/
    └── {FOUNDATION_MODEL}/
        └── {MAGNIFICATION}/
            └── features/
                ├── {slide_id_1}.pt
                ├── {slide_id_2}.pt
                └── ...
```

### Path Components:
- **CANCER_TYPE**: Cancer type identifier (e.g., BRCA, LUAD, COAD)
- **SLIDE_TYPE**: Slide type - FS (frozen section) or PM (FFPE/permanent)
- **FOUNDATION_MODEL**: Model name (e.g., CHIEF, UNI, GIGAPATH, VIRCHOW2)
- **MAGNIFICATION**: Magnification level (e.g., 20X, 40X)
- **features**: Directory containing .pt feature files

### Example:
```
features/
└── BRCA-FS/
    └── CHIEF/
        └── 20X/
            └── features/
                ├── PATIENT001-SLIDE01.pt
                ├── PATIENT002-SLIDE01.pt
                └── PATIENT003-SLIDE01.pt
```

## File Format

Each `.pt` file contains a PyTorch tensor with shape: `[n_patches, feature_dim]`

Where:
- **n_patches**: Number of tissue patches from the whole-slide image (typically 1000-5000)
- **feature_dim**: Dimension of foundation model features
  - CHIEF: 768
  - UNI: 1024
  - GIGAPATH: 1536
  - VIRCHOW2: 2560

## Creating Feature Files

Features must be extracted using foundation models. Example:

```python
import torch

# After running foundation model feature extraction
# features shape: [n_patches, 768] for CHIEF
features = extract_features_from_wsi('slide.svs', model='CHIEF')

# Save with slide identifier as filename
torch.save(features, 'PATIENT001-SLIDE01.pt')
```

## Important Notes

- Plsease extract features using desired foundation models
- Ensure feature dimension matches the foundation model you're using
- All feature files for a given model must have the same dimension
- Feature extraction is done ONCE before training, not during training
- **Slide IDs in filenames must match the `folder_id` in label files**

## Quality Checks

Before training, validate that:
1. All .pt files load without errors
2. Tensors have correct shape [n_patches, feature_dim]
3. No NaN or Inf values
4. Minimum 50 patches per slide
5. Feature dimension matches foundation model
6. Slide IDs match those in label files

Use the validation script:
```bash
python data/validation.py --cancer BRCA --foundation_model CHIEF --slide_type FS
```
